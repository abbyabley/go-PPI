import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from glob import glob
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score

# ================= 🔧 路径与超参配置 =================
BASE_DIR = '/home/dongshali/fasta_data'
TSV_PATH = f'{BASE_DIR}/pdb_3di_train_filtered.tsv'
ESM_DIR  = f'{BASE_DIR}/esm2_features'
PPI_DIR  = f'{BASE_DIR}/ppi_prior_features_single'
GO_MAP   = f'{BASE_DIR}/meta_data/go_mapping.pkl'

# 🚀 优化 1：利用 A6000 庞大显存，将 Batch Size 暴增
BATCH_SIZE = 128 
LR = 1e-4
EPOCHS = 20
NUM_GO_CLASSES = 150
# ====================================================

# ----------------- 1. 数据集定义 -----------------
class PPIAwareGODatasetSingle(Dataset):
    def __init__(self, tsv_path, esm_root, ppi_dir, go_mapping, num_classes=150):
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.ppi_dir = ppi_dir
        self.num_classes = num_classes
        
        self.esm_map = {os.path.basename(f).replace('.pt', ''): f 
                        for f in glob(os.path.join(esm_root, '**', '*.pt'), recursive=True)}
        
        with open(go_mapping, 'rb') as f:
            self.go_map = pickle.load(f)['mapping'] 

        self.data_list = []
        for idx, row in self.df.iterrows():
            pdb = str(row['pdb_name'])
            chain = str(row['chain'])
            t_key_full = f"{pdb}_{chain}"
            t_key_simple = pdb 
            
            esm_p = self.esm_map.get(t_key_full) or self.esm_map.get(t_key_simple)
            if not esm_p: continue
            
            ppi_p = os.path.join(ppi_dir, f"{t_key_full}.pt")
            
            if os.path.exists(ppi_p):
                self.data_list.append({"pdb_name": pdb, "esm_path": esm_p, "ppi_path": ppi_p})
                
        print(f"✅ Valid Sequence Samples: {len(self.data_list)}")

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        try:
            esm = torch.load(item['esm_path']).float()
            ppi = torch.load(item['ppi_path']).float()
        except: return None

        min_L = min(esm.shape[0], ppi.shape[0])
        esm, ppi = esm[:min_L], ppi[:min_L]
        
        go_labels = torch.zeros(self.num_classes)
        for fid in self.go_map.get(item['pdb_name'], []):
            if fid < self.num_classes: go_labels[fid] = 1.0

        return {"esm": esm, "ppi_prior": ppi, "go_labels": go_labels, "mask": torch.ones(min_L)}

def collate_fn_single(batch):
    batch = [b for b in batch if b is not None and b['mask'].sum() > 0]
    if not batch: return None
    
    max_len = max([b['esm'].shape[0] for b in batch])
    B, D, NUM_GO = len(batch), batch[0]['esm'].shape[1], batch[0]['go_labels'].shape[0]
    
    pad_esm = torch.zeros(B, max_len, D)
    pad_ppi = torch.zeros(B, max_len)
    pad_msk = torch.zeros(B, max_len)
    go_labels = torch.zeros(B, NUM_GO)

    for i, b in enumerate(batch):
        L = b['esm'].shape[0]
        pad_esm[i, :L] = b['esm']
        pad_ppi[i, :L], pad_msk[i, :L] = b['ppi_prior'], 1.0
        go_labels[i] = b['go_labels']

    return {"esm": pad_esm, "ppi_prior": pad_ppi, "go_labels": go_labels, "mask": pad_msk}

# ----------------- 2. 纯序列 PPI 引导功能预测网络 -----------------
class SinglePPIAwareGOPredictor(nn.Module):
    def __init__(self, esm_dim=1280, num_classes=150, hidden_dim=512):
        super().__init__()
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, esm, ppi_prior, mask):
        # 序列特征降维
        h_esm = self.esm_proj(esm) 
        
        # 1. 获取全局特征 (Global Context)
        mask_weights = mask.unsqueeze(-1)
        global_rep = (h_esm * mask_weights).sum(dim=1) / (mask_weights.sum(dim=1) + 1e-6) 
        
        # 2. 获取局部互作特征 (PPI-Guided Local Focus)
        ppi_weights = (ppi_prior * mask).unsqueeze(-1) 
        local_rep = (h_esm * ppi_weights).sum(dim=1) / (ppi_weights.sum(dim=1) + 1e-6) 
        
        # 3. 融合特征：全局大局观 + 局部互作先验
        final_rep = global_rep + local_rep
        
        # 4. 功能分类
        return self.classifier(final_rep)

# ----------------- 3. 训练主循环 -----------------
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> Using device: {device}")

    full_ds = PPIAwareGODatasetSingle(TSV_PATH, ESM_DIR, PPI_DIR, GO_MAP, num_classes=NUM_GO_CLASSES)
    train_len = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_len, len(full_ds) - train_len])
    
    # 🚀 优化 2：增加 num_workers 提升多线程读取速度，开启 pin_memory 加速显存拷贝
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              collate_fn=collate_fn_single, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            collate_fn=collate_fn_single, num_workers=8, pin_memory=True)

    model = SinglePPIAwareGOPredictor(num_classes=NUM_GO_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    
    # 🚀 优化 3：引入混合精度训练 (AMP) Scaler
    scaler = torch.cuda.amp.GradScaler()
    
    best_auprc = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]")
        
        for batch in loop:
            if batch is None: continue 
            
            # non_blocking=True 配合 pin_memory 实现数据传输与计算并行
            esm = batch['esm'].to(device, non_blocking=True)
            ppi = batch['ppi_prior'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            labels = batch['go_labels'].to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 使用混合精度上下文
            with torch.cuda.amp.autocast():
                logits = model(esm, ppi, mask)
                loss = criterion(logits, labels)
            
            # Scaler 缩放梯度并反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        # --- 验证阶段 ---
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Val]"):
                if batch is None: continue
                esm = batch['esm'].to(device, non_blocking=True)
                ppi = batch['ppi_prior'].to(device, non_blocking=True)
                mask = batch['mask'].to(device, non_blocking=True)
                labels = batch['go_labels'].to(device, non_blocking=True)
                
                with torch.cuda.amp.autocast():
                    logits = model(esm, ppi, mask)
                
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        try:
            auprc = average_precision_score(all_labels, all_preds, average='micro')
        except ValueError:
            auprc = 0.0
            
        print(f"\nEpoch {epoch+1} | Val Micro-AUPRC: {auprc:.4f}")
        
        if auprc > best_auprc:
            best_auprc = auprc
            # 保存双路融合模型的最终权重
            torch.save(model.state_dict(), "best_go_dual_predictor.pth")
            print(f"🌟 New Best Model Saved! (AUPRC: {auprc:.4f})")

if __name__ == '__main__':
    train()
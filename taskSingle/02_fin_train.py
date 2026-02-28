import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from glob import glob
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import average_precision_score

# ================= 🔧 随机种子固定 =================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ================= 🔧 路径配置 =================
BASE_DIR = '/home/dongshali/fasta_data'
TRAIN_CSV = './Task_Single_dataset/Task_DeepPPISP/352_name_seq_label.csv'
TEST_CSV  = './Task_Single_dataset/Task_DeepPPISP/Test_70.csv'
ESM_DIR   = f'{BASE_DIR}/esm2_features'
GO_MAP    = f'{BASE_DIR}/meta_data/go_mapping.pkl'

# ----------------- 1. Dataset 定义 -----------------
class DeepPPISP_OracleDataset(Dataset):
    def __init__(self, csv_path, esm_root, go_mapping):
        self.df = pd.read_csv(csv_path)
        with open(go_mapping, 'rb') as f:
            go_data = pickle.load(f)
            self.go_map = go_data['mapping']
            self.num_classes = go_data.get('num_classes', 150)
            
        self.esm_files = {os.path.basename(f).replace('.pt', ''): f 
                          for f in glob(os.path.join(esm_root, '**', '*.pt'), recursive=True)}

        self.data_list = []
        for idx, row in self.df.iterrows():
            raw_name = str(row.iloc[0]).strip().lstrip('>')
            if raw_name not in self.go_map: continue
            esm_p = self.esm_files.get(raw_name)
            if not esm_p: continue
                
            label_str = str(row['label']).strip()
            ppi_tensor = torch.tensor([float(c) for c in label_str])
            self.data_list.append({"name": raw_name, "esm_path": esm_p, "ppi_true": ppi_tensor})
            
        print(f"✅ {csv_path} 加载完毕 | 有效样本: {len(self.data_list)}")

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        esm = torch.load(item['esm_path']).float()
        ppi = item['ppi_true'].float()
        min_L = min(esm.shape[0], ppi.shape[0])
        esm, ppi = esm[:min_L], ppi[:min_L]
        
        go_labels = torch.zeros(self.num_classes)
        for fid in self.go_map.get(item['name'], []):
            if fid < self.num_classes: go_labels[fid] = 1.0
        return {"esm": esm, "ppi_prior": ppi, "go_labels": go_labels, "mask": torch.ones(min_L)}

def collate_fn_oracle(batch):
    batch = [b for b in batch if b is not None]
    max_len = max([b['esm'].shape[0] for b in batch])
    B, D, NUM_GO = len(batch), batch[0]['esm'].shape[1], batch[0]['go_labels'].shape[0]
    pad_esm, pad_ppi, pad_msk = torch.zeros(B, max_len, D), torch.zeros(B, max_len), torch.zeros(B, max_len)
    go_labels = torch.zeros(B, NUM_GO)
    for i, b in enumerate(batch):
        L = b['esm'].shape[0]
        pad_esm[i, :L], pad_ppi[i, :L], pad_msk[i, :L] = b['esm'], b['ppi_prior'], 1.0
        go_labels[i] = b['go_labels']
    return {"esm": pad_esm, "ppi_prior": pad_ppi, "go_labels": go_labels, "mask": pad_msk}

# ----------------- 2. 终极高阶网络定义 (AdvancedFlexibleGOPredictor) -----------------
class AdvancedFlexibleGOPredictor(nn.Module):
    def __init__(self, esm_dim=1280, num_classes=150, hidden_dim=512, is_baseline=False):
        super().__init__()
        self.is_baseline = is_baseline
        
        # 1. 基础投影层
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU()
        )
        
        # 2. 多尺度局部基序卷积 (Multi-Scale Motif CNN)
        self.conv_k3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2)
        self.conv_norm = nn.LayerNorm(hidden_dim)
        
        # 3. 自适应注意力聚合网络 (Adaptive Attention)
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # 4. 门控跨模态融合机制 (Gating Mechanism)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 5. 高阶分类器 (增加 BatchNorm 防止协变量偏移)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, esm, ppi_prior, mask):
        B, L, _ = esm.size()
        
        # --- A. 基础降维 ---
        h_esm = self.esm_proj(esm) # [B, L, 512]
        
        # --- B. 多尺度序列基序提取 ---
        h_esm_t = h_esm.transpose(1, 2) # [B, 512, L]
        feat_k3 = torch.relu(self.conv_k3(h_esm_t)) # [B, 256, L]
        feat_k5 = torch.relu(self.conv_k5(h_esm_t)) # [B, 256, L]
        h_conv = torch.cat([feat_k3, feat_k5], dim=1).transpose(1, 2) # [B, L, 512]
        
        # 残差连接与归一化
        h_encoded = self.conv_norm(h_esm + h_conv) 
        mask_weights = mask.unsqueeze(-1)
        
        # --- C. 全局分支：屏蔽无效区域的均值池化 ---
        global_rep = (h_encoded * mask_weights).sum(dim=1) / (mask_weights.sum(dim=1) + 1e-6) # [B, 512]
        
        if self.is_baseline:
            # 🟢 Baseline 模式：退化为全局宏观特征
            final_rep = global_rep
        else:
            # --- D. 局部分支：基于 PPI 先验偏置的自适应注意力 ---
            raw_attn = self.attn_net(h_encoded) # [B, L, 1]
            ppi_bias = ppi_prior.unsqueeze(-1) * 5.0 # 强制放缩 PPI 权重引导注意力
            biased_attn = raw_attn + ppi_bias
            
            # Mask 处理与 Softmax 归一化
            biased_attn = biased_attn.masked_fill(mask_weights == 0, -1e9)
            attn_weights = torch.softmax(biased_attn, dim=1) 
            
            # 注意力加权池化
            local_rep = (h_encoded * attn_weights).sum(dim=1) # [B, 512]
            
            # --- E. 门控跨模态融合 ---
            concat_rep = torch.cat([global_rep, local_rep], dim=-1) # [B, 1024]
            gate = self.fusion_gate(concat_rep) # [B, 512]
            
            # 动态门控融合：取长补短
            final_rep = gate * global_rep + (1 - gate) * local_rep 
            
        return self.classifier(final_rep)

# ----------------- 3. 训练主循环 -----------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 启动高阶实验 | 模式: {'BASELINE (Global CNN Only)' if args.is_baseline else 'PROPOSED (Advanced PPI-Guided Fusion)'}")

    train_ds = DeepPPISP_OracleDataset(TRAIN_CSV, ESM_DIR, GO_MAP)
    val_ds = DeepPPISP_OracleDataset(TEST_CSV, ESM_DIR, GO_MAP)
    
    model = AdvancedFlexibleGOPredictor(num_classes=train_ds.num_classes, is_baseline=args.is_baseline).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_oracle, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_oracle, num_workers=4)

    best_auprc = 0.0
    save_name = "best_go_advanced_baseline.pth" if args.is_baseline else "best_go_advanced_proposed.pth"

    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            esm, ppi, mask, labels = batch['esm'].to(device), batch['ppi_prior'].to(device), batch['mask'].to(device), batch['go_labels'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(esm, ppi, mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                esm, ppi, mask, labels = batch['esm'].to(device), batch['ppi_prior'].to(device), batch['mask'].to(device), batch['go_labels'].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(esm, ppi, mask)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        auprc = average_precision_score(np.vstack(all_labels), np.vstack(all_preds), average='micro')
        print(f"Epoch {epoch+1} | Val Micro-AUPRC: {auprc:.4f}")
        
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(model.state_dict(), save_name)
            print(f"🌟 已保存最佳模型 ({save_name})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--is_baseline', action='store_true', help='是否运行 Baseline 模式')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200) # 🔴 提高到30轮，让高阶模型充分收敛
    args = parser.parse_args()
    main(args)
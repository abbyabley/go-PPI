import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import average_precision_score

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

BASE_DIR = '/home/dongshali/fasta_data'
GO_MAP   = f'{BASE_DIR}/meta_data/go_mapping.pkl'
BATCH_SIZE, LR, EPOCHS, NUM_GO_CLASSES = 128, 1e-4, 20, 150

class StaticGODatasetWithPPI(Dataset):
    def __init__(self, tsv_path, go_mapping, num_classes=150):
        self.df = pd.read_csv(tsv_path, sep='\t')
        self.num_classes = num_classes
        with open(go_mapping, 'rb') as f:
            self.go_map = pickle.load(f)['mapping'] 

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            esm = torch.load(row['esm_path']).float()
            # 🟢 Proposed 模型：读取真实的 PPI 概率文件
            ppi = torch.load(row['ppi_path']).float()
        except: return None
        
        min_L = min(esm.shape[0], ppi.shape[0])
        esm, ppi = esm[:min_L], ppi[:min_L]
        
        go_labels = torch.zeros(self.num_classes)
        for fid in self.go_map.get(row['pdb_name'], []):
            if fid < self.num_classes: go_labels[fid] = 1.0
            
        return {"esm": esm, "ppi_prior": ppi, "go_labels": go_labels, "mask": torch.ones(min_L)}

def collate_fn_single(batch):
    batch = [b for b in batch if b is not None]
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

class DualPPIAwareGOPredictor(nn.Module):
    def __init__(self, esm_dim=1280, num_classes=150, hidden_dim=512):
        super().__init__()
        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, num_classes))

    def forward(self, esm, ppi_prior, mask):
        h_esm = self.esm_proj(esm) 
        
        # 🟢 Proposed：双路特征融合机制 (全局大局观 + PPI 局部聚焦)
        mask_weights = mask.unsqueeze(-1)
        global_rep = (h_esm * mask_weights).sum(dim=1) / (mask_weights.sum(dim=1) + 1e-6) 
        
        ppi_weights = (ppi_prior * mask).unsqueeze(-1) 
        local_rep = (h_esm * ppi_weights).sum(dim=1) / (ppi_weights.sum(dim=1) + 1e-6) 
        
        final_rep = global_rep + local_rep
        return self.classifier(final_rep)

def train_dual():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(">>> 正在训练 Proposed 双路模型 (序列 + PPI)...")

    train_ds = StaticGODatasetWithPPI(f'{BASE_DIR}/go_train_cleaned.tsv', GO_MAP, NUM_GO_CLASSES)
    val_ds = StaticGODatasetWithPPI(f'{BASE_DIR}/go_val_cleaned.tsv', GO_MAP, NUM_GO_CLASSES)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_single, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_single, num_workers=8, pin_memory=True)

    model = DualPPIAwareGOPredictor(num_classes=NUM_GO_CLASSES).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scaler = torch.amp.GradScaler('cuda')
    
    best_auprc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            if batch is None: continue 
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
                if batch is None: continue
                esm, ppi, mask, labels = batch['esm'].to(device), batch['ppi_prior'].to(device), batch['mask'].to(device), batch['go_labels'].to(device)
                with torch.amp.autocast('cuda'): logits = model(esm, ppi, mask)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        auprc = average_precision_score(np.vstack(all_labels), np.vstack(all_preds), average='micro')
        print(f"Epoch {epoch+1} | Dual AUPRC: {auprc:.4f}")
        if auprc > best_auprc:
            best_auprc = auprc
            torch.save(model.state_dict(), "best_go_dual.pth")

if __name__ == '__main__':
    train_dual()
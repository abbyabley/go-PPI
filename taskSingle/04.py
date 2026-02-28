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
import matplotlib.pyplot as plt
import seaborn as sns

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
ADJ_DIR   = f'{BASE_DIR}/graph_adj_features' 
GO_MAP    = f'{BASE_DIR}/meta_data/go_mapping.pkl'

# ================= 📊 核心新增 1：多标签 Fmax 计算 =================
def calculate_fmax(y_true, y_pred, thresholds=np.linspace(0.01, 0.99, 100)):
    best_fmax = 0.0
    for t in thresholds:
        pred_bin = (y_pred >= t).astype(int)
        tp = np.sum((pred_bin == 1) & (y_true == 1))
        fp = np.sum((pred_bin == 1) & (y_true == 0))
        fn = np.sum((pred_bin == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > best_fmax:
                best_fmax = f1
    return best_fmax

# ================= 📊 核心新增 2：过拟合自动化判定 =================
def detect_overfitting(train_losses, val_losses, val_auprcs):
    epochs = len(train_losses)
    if epochs < 6: return "训练轮数过少，无法准确判定"
        
    for i in range(5, epochs):
        val_rises = all(val_losses[j] < val_losses[j+1] for j in range(i-5, i))
        train_drops = (train_losses[i-5] - train_losses[i]) / train_losses[i-5] > 0.2
        if val_rises and train_drops:
            return f"⚠️ 严重过拟合 (依据: 第 {i-4}~{i+1} 轮 Val Loss 连续上升，且 Train Loss 暴跌)"
            
    best_auprc_idx = np.argmax(val_auprcs)
    if epochs - best_auprc_idx >= 4: 
        for i in range(best_auprc_idx + 3, epochs):
            auprc_drops = all(val_auprcs[j] > val_auprcs[j+1] for j in range(i-3, i))
            train_drops = train_losses[i] < train_losses[i-3]
            if auprc_drops and train_drops:
                return f"⚠️ 轻微过拟合 (依据: 第 {i-2}~{i+1} 轮 Val AUPRC 连续下降，已越过最佳泛化点)"
                
    return "✅ 无过拟合，模型收敛良好 (依据: Train/Val 趋势一致，性能指标稳定)"

# ================= 📊 核心新增 3：论文级训练曲线绘制 (加入动态传参) =================
def plot_training_curves(train_losses, val_losses, val_auprcs, val_fmaxs, is_baseline, exp_name, pdf_path, png_path):
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial' 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    
    title_prefix = "Baseline Mode" if is_baseline else "Proposed Multimodal (PPI-Guided)"
    # 🔴 图表大标题自动加入实验标识
    fig.suptitle(f"{title_prefix} Training Dynamics | Exp: {exp_name}", fontsize=16, fontweight='bold', y=1.05)

    # --- 子图 1: Loss 曲线 ---
    ax = axes[0]
    ax.plot(epochs, train_losses, color='#d32f2f', linestyle='-', marker='o', markersize=4, label='Train Loss')
    ax.plot(epochs, val_losses, color='#1976d2', linestyle='-', marker='s', markersize=4, label='Val Loss')
    ax.set_title("Epoch-Loss Curve", fontsize=14, fontweight='bold')
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight='bold')
    ax.set_ylabel("Loss Value", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    best_val_loss_epoch = np.argmin(val_losses) + 1
    if best_val_loss_epoch < len(epochs) - 3:
        ax.axvline(x=best_val_loss_epoch, color='red', linestyle='--', alpha=0.6)
        ax.text(best_val_loss_epoch + 0.5, np.mean(val_losses), 'Divergence Point', color='red')

    # --- 子图 2: Micro-AUPRC 曲线 ---
    ax = axes[1]
    ax.plot(epochs, val_auprcs, color='#2e7d32', linestyle='-', marker='^', markersize=4, label='Val Micro-AUPRC')
    best_auprc_epoch = np.argmax(val_auprcs) + 1
    best_auprc_val = np.max(val_auprcs)
    ax.scatter(best_auprc_epoch, best_auprc_val, color='red', s=100, zorder=5)
    ax.annotate(f'Best: {best_auprc_val:.4f}', (best_auprc_epoch, best_auprc_val), 
                xytext=(10, -15), textcoords='offset points', color='red', fontweight='bold')
    ax.set_title("Epoch - Micro-AUPRC", fontsize=14, fontweight='bold')
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight='bold')
    ax.set_ylabel("Micro-AUPRC", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    if len(val_auprcs) - best_auprc_epoch >= 5:
        if all(val_auprcs[i] > val_auprcs[i+1] for i in range(best_auprc_epoch-1, best_auprc_epoch+4)):
            ax.text(best_auprc_epoch + 1, best_auprc_val * 0.9, '可能过拟合', color='darkred')

    # --- 子图 3: Fmax 曲线 ---
    ax = axes[2]
    ax.plot(epochs, val_fmaxs, color='#7b1fa2', linestyle='-', marker='D', markersize=4, label='Val Fmax')
    best_fmax_epoch = np.argmax(val_fmaxs) + 1
    best_fmax_val = np.max(val_fmaxs)
    ax.scatter(best_fmax_epoch, best_fmax_val, color='red', s=100, zorder=5)
    ax.annotate(f'Best: {best_fmax_val:.4f}', (best_fmax_epoch, best_fmax_val), 
                xytext=(10, -15), textcoords='offset points', color='red', fontweight='bold')
    ax.set_title("Epoch - Fmax Coefficient", fontsize=14, fontweight='bold')
    ax.set_xlabel("Training Epochs", fontsize=12, fontweight='bold')
    ax.set_ylabel("Fmax", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    if np.corrcoef(val_auprcs, val_fmaxs)[0, 1] < 0.5:
        ax.text(epochs[len(epochs)//2], np.mean(val_fmaxs), '需调整分类阈值', color='darkorange')

    plt.tight_layout()
    # 🔴 动态保存路径
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    print(f"✅ 论文级训练曲线已生成: {pdf_path} / {png_path}")


# ----------------- 1. Dataset 定义 -----------------
class DeepPPISP_MultimodalDataset(Dataset):
    def __init__(self, csv_path, esm_root, adj_root, go_mapping):
        self.df = pd.read_csv(csv_path)
        with open(go_mapping, 'rb') as f:
            go_data = pickle.load(f)
            self.go_map = go_data['mapping']
            self.num_classes = go_data.get('num_classes', 150)
            
        self.esm_files = {os.path.basename(f).replace('.pt', ''): f 
                          for f in glob(os.path.join(esm_root, '**', '*.pt'), recursive=True)}
        self.adj_files = {os.path.basename(f).replace('.pt', ''): f 
                          for f in glob(os.path.join(adj_root, '*.pt'))}

        self.data_list = []
        for idx, row in self.df.iterrows():
            raw_name = str(row.iloc[0]).strip().lstrip('>')
            if raw_name not in self.go_map: continue
            
            esm_p = self.esm_files.get(raw_name)
            adj_p = self.adj_files.get(raw_name)
            if not esm_p or not adj_p: continue
                
            label_str = str(row['label']).strip()
            ppi_tensor = torch.tensor([float(c) for c in label_str])
            self.data_list.append({
                "name": raw_name, "esm_path": esm_p, 
                "adj_path": adj_p, "ppi_true": ppi_tensor
            })
            
        print(f"✅ {csv_path} 加载完毕 | 多模态有效样本: {len(self.data_list)}")

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        esm = torch.load(item['esm_path']).float()
        adj = torch.load(item['adj_path']).float()
        ppi = item['ppi_true'].float()
        
        min_L = min(esm.shape[0], ppi.shape[0], adj.shape[0])
        esm, ppi = esm[:min_L], ppi[:min_L]
        adj = adj[:min_L, :min_L]
        
        go_labels = torch.zeros(self.num_classes)
        for fid in self.go_map.get(item['name'], []):
            if fid < self.num_classes: go_labels[fid] = 1.0
            
        return {"esm": esm, "adj": adj, "ppi_prior": ppi, "go_labels": go_labels, "mask": torch.ones(min_L)}

def collate_fn_multimodal(batch):
    batch = [b for b in batch if b is not None]
    max_len = max([b['esm'].shape[0] for b in batch])
    B, D, NUM_GO = len(batch), batch[0]['esm'].shape[1], batch[0]['go_labels'].shape[0]
    
    pad_esm = torch.zeros(B, max_len, D)
    pad_adj = torch.zeros(B, max_len, max_len)
    pad_ppi = torch.zeros(B, max_len)
    pad_msk = torch.zeros(B, max_len)
    go_labels = torch.zeros(B, NUM_GO)
    
    for i, b in enumerate(batch):
        L = b['esm'].shape[0]
        pad_esm[i, :L], pad_ppi[i, :L], pad_msk[i, :L] = b['esm'], b['ppi_prior'], 1.0
        pad_adj[i, :L, :L] = b['adj']
        go_labels[i] = b['go_labels']
        
    return {"esm": pad_esm, "adj": pad_adj, "ppi_prior": pad_ppi, "go_labels": go_labels, "mask": pad_msk}

# ----------------- 2. 空间 GCN 层定义 -----------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, adj):
        out = self.proj(x)
        out = torch.bmm(adj, out) 
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        out = out / deg
        return torch.relu(out)

# ----------------- 3. 终极多模态网络定义 (针对过拟合与特征冗余优化的瘦身版) -----------------
class AdvancedFlexibleGOPredictor(nn.Module):
    def __init__(self, esm_dim=1280, num_classes=150, hidden_dim=512, is_baseline=False):
        super().__init__()
        self.is_baseline = is_baseline
        
        # 1. 基础投影层
        self.esm_proj = nn.Sequential(nn.Linear(esm_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        
        # 2. 多尺度局部基序卷积
        self.conv_k3 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.conv_k5 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=5, padding=2)
        
        # 3. 双层空间 GCN
        self.gcn1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim)
        
        self.feature_norm = nn.LayerNorm(hidden_dim)
        
        # 🔴 优化点 1：特征层正则化，防止 GCN/CNN 提取的特征在 335 个样本上死记硬背
        self.feat_dropout = nn.Dropout(0.3)
        
        # 4. 自适应注意力池化 (去掉了臃肿的 fusion_gate)
        self.attn_net = nn.Sequential(nn.Linear(hidden_dim, 128), nn.Tanh(), nn.Linear(128, 1))
        
        # 🔴 优化点 2：极其优雅的可学习 PPI 先验权重 (Learnable Prior Bias)
        # 初始值为 0.5，让网络通过反向传播自动决定 PPI 信息的权重大小
        self.ppi_alpha = nn.Parameter(torch.tensor(0.5))
        
        # 5. 高阶分类器 (保持强正则化)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.6), nn.Linear(256, num_classes)
        )

    def forward(self, esm, adj, ppi_prior, mask):
        B, L, _ = esm.size()
        
        # --- A. 基础投影 & 多尺度 CNN ---
        h_esm = self.esm_proj(esm) 
        h_esm_t = h_esm.transpose(1, 2)
        feat_k3 = torch.relu(self.conv_k3(h_esm_t)) 
        feat_k5 = torch.relu(self.conv_k5(h_esm_t)) 
        h_conv = torch.cat([feat_k3, feat_k5], dim=1).transpose(1, 2) 
        
        # --- B. 空间 GCN ---
        h_gcn = self.gcn1(h_esm, adj)
        h_gcn = self.gcn2(h_gcn, adj) 
        
        # --- C. 特征融合与扰动 ---
        h_encoded = self.feature_norm(h_esm + h_conv + h_gcn) 
        h_encoded = self.feat_dropout(h_encoded) # 施加特征级 Dropout
        mask_weights = mask.unsqueeze(-1)
        
        if self.is_baseline:
            # Baseline: 纯粹的全局均值池化
            final_rep = (h_encoded * mask_weights).sum(dim=1) / (mask_weights.sum(dim=1) + 1e-6) 
        else:
            # 🔴 优化点 3：柔和残差注意力池化 (Soft Residual Attention)
            # 让网络自己提取全局语义 (raw_attn)，同时用 PPI 作为柔和的引导偏置
            raw_attn = self.attn_net(h_encoded)
            
            # 使用可学习的 ppi_alpha，不再死板地放大 5 倍
            ppi_bias = ppi_prior.unsqueeze(-1) * self.ppi_alpha
            biased_attn = raw_attn + ppi_bias
            
            # Mask 与 归一化
            biased_attn = biased_attn.masked_fill(mask_weights == 0, -1e9)
            attn_weights = torch.softmax(biased_attn, dim=1) 
            
            # 直接输出最终特征，彻底抛弃容易过拟合的拼接和门控网络！
            final_rep = (h_encoded * attn_weights).sum(dim=1) 
            
        return self.classifier(final_rep)
# ----------------- 4. 训练主循环 -----------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 🔴 动态生成基础文件名前缀
    mode_str = "baseline" if args.is_baseline else "proposed"
    base_file_prefix = f"{args.exp_name}_{mode_str}_lr{args.lr}"
    
    print(f"\n🚀 启动多模态实验 | 模式: {mode_str.upper()} | 实验名: {args.exp_name}")

    train_ds = DeepPPISP_MultimodalDataset(TRAIN_CSV, ESM_DIR, ADJ_DIR, GO_MAP)
    val_ds = DeepPPISP_MultimodalDataset(TEST_CSV, ESM_DIR, ADJ_DIR, GO_MAP)
    
    model = AdvancedFlexibleGOPredictor(num_classes=train_ds.num_classes, is_baseline=args.is_baseline).to(device)
    # 采用加强版权重衰减，避免复杂模型过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-2)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_multimodal, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_multimodal, num_workers=4)

    train_losses, val_losses, val_auprcs, val_fmaxs = [], [], [], []
    best_auprc = 0.0
    
    # 🔴 动态分配保存路径
    save_model_path = f"best_model_{base_file_prefix}.pth"
    save_csv_path = f"training_log_{base_file_prefix}.csv"
    save_pdf_path = f"training_curves_{base_file_prefix}.pdf"
    save_png_path = f"training_curves_{base_file_prefix}.png"

    for epoch in range(args.epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Ep {epoch+1} [Train]"):
            esm, adj, ppi, mask, labels = batch['esm'].to(device), batch['adj'].to(device), batch['ppi_prior'].to(device), batch['mask'].to(device), batch['go_labels'].to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(esm, adj, ppi, mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_train_loss += loss.item()
            
        avg_train_loss = epoch_train_loss / len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                esm, adj, ppi, mask, labels = batch['esm'].to(device), batch['adj'].to(device), batch['ppi_prior'].to(device), batch['mask'].to(device), batch['go_labels'].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(esm, adj, ppi, mask)
                    v_loss = criterion(logits, labels)
                epoch_val_loss += v_loss.item()
                
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        if len(all_preds) > 0:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            try:
                auprc = average_precision_score(all_labels, all_preds, average='micro')
                fmax = calculate_fmax(all_labels, all_preds)
            except ValueError:
                auprc, fmax = 0.0, 0.0
                
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            val_auprcs.append(auprc)
            val_fmaxs.append(fmax)
            
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | AUPRC: {auprc:.4f} | Fmax: {fmax:.4f}")
            
            if auprc > best_auprc:
                best_auprc = auprc
                torch.save(model.state_dict(), save_model_path)
                print(f"🌟 已保存最佳模型 -> {save_model_path}")

    print("\n" + "="*50)
    print("📈 训练结束，正在执行自动化指标分析与绘图...")
    
    log_df = pd.DataFrame({
        'Epoch': range(1, args.epochs + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Micro_AUPRC': val_auprcs,
        'Fmax': val_fmaxs
    })
    log_df.to_csv(save_csv_path, index=False)
    print(f"✅ 日志已导出至: {save_csv_path}")
    
    # 🔴 传入动态文件名进行绘图
    plot_training_curves(train_losses, val_losses, val_auprcs, val_fmaxs, args.is_baseline, args.exp_name, save_pdf_path, save_png_path)
    
    diagnosis = detect_overfitting(train_losses, val_losses, val_auprcs)
    print("\n🔍 【模型健康度诊断报告】")
    print(diagnosis)
    print("="*50 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 🔴 新增实验名称参数
    parser.add_argument('--exp_name', type=str, default='v1', help='实验名称标识，用于区分输出文件')
    parser.add_argument('--is_baseline', action='store_true', help='是否运行 Baseline 模式')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    main(args)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 使用你的 1 号显卡

import torch
import pandas as pd
from tqdm import tqdm
import esm

# ================= 🔧 路径配置 =================
CSV_FILES = [
    './Task_Single_dataset/Task_DeepPPISP/352_name_seq_label.csv',
    './Task_Single_dataset/Task_DeepPPISP/Test_70.csv',
]
SAVE_DIR = '/home/dongshali/fasta_data/esm2_features'
os.makedirs(SAVE_DIR, exist_ok=True)

# 加载 ESM-2 模型 (使用 650M 参数版本，适合大多数科研需求)
model_name = "esm2_t33_650M_UR50D"
print(f">>> 正在加载模型 {model_name}...")
model, alphabet = esm.pretrained.load_model_and_alphabet_hub(model_name)
batch_converter = alphabet.get_batch_converter()
model.eval().cuda()
# ===============================================

def extract_features():
    # 1. 汇总所有需要提取的序列
    all_tasks = []
    for f in CSV_FILES:
        df = pd.read_csv(f)
        name_col = df.columns[0]
        for _, row in df.iterrows():
            name = str(row[name_col]).strip().lstrip('>')
            seq = str(row['sequence']).strip()
            all_tasks.append((name, seq))
    
    # 去重，防止重复计算
    all_tasks = list(set(all_tasks))
    print(f"✅ 待处理唯一序列总数: {len(all_tasks)}")

    # 2. 逐条提取并保存
    with torch.no_grad():
        for name, seq in tqdm(all_tasks, desc="Extracting ESM-2"):
            save_path = os.path.join(SAVE_DIR, f"{name}.pt")
            
            # 如果已经存在，直接跳过（断点续传）
            if os.path.exists(save_path):
                continue
                
            # 处理输入
            data = [(name, seq)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.cuda()
            
            # 推理
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
            token_representations = results["representations"][33]
            
            # 去掉 [CLS] 和 [SEP] 标记，只保留残基本身的特征 (L, 1280)
            # batch_tokens 的第 0 位是 <cls>，最后一位是 <eos>
            sequence_representation = token_representations[0, 1 : len(seq) + 1]
            
            # 转存到 CPU 并保存
            torch.save(sequence_representation.cpu(), save_path)

    print(f"🎉 特征提取全部完成！文件保存在: {SAVE_DIR}")

if __name__ == "__main__":
    extract_features()
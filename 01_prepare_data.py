import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
import pickle

# ================= 配置路径 =================
BASE_DIR = '/home/dongshali/fasta_data'
TSV_PATH = f'{BASE_DIR}/pdb_3di_train_filtered.tsv'
ESM_DIR  = f'{BASE_DIR}/esm2_features'
PPI_DIR  = f'{BASE_DIR}/ppi_prior_features_single'
GO_MAP   = f'{BASE_DIR}/meta_data/go_mapping.pkl'
# ============================================

def clean_and_split_data():
    print(">>> 1. 正在读取原始数据...")
    df = pd.read_csv(TSV_PATH, sep='\t')
    print(f"原始数据总数: {len(df)}")
    
    print(">>> 2. 正在扫描特征文件进行严格对齐与清洗...")
    # 扫描真实的特征文件列表
    esm_files = {os.path.basename(f).replace('.pt', ''): f 
                 for f in glob(os.path.join(ESM_DIR, '**', '*.pt'), recursive=True)}
    
    with open(GO_MAP, 'rb') as f:
        go_map = pickle.load(f)['mapping']

    valid_rows = []
    # 遍历每一行，检查是否同时具备 ESM 特征、PPI 概率特征 和 GO 标签
    for idx, row in df.iterrows():
        pdb = str(row['pdb_name'])
        chain = str(row['chain'])
        t_key_full = f"{pdb}_{chain}"
        t_key_simple = pdb 
        
        # 匹配 ESM 文件
        esm_p = esm_files.get(t_key_full) or esm_files.get(t_key_simple)
        if not esm_p: continue
        
        # 匹配 PPI 先验文件
        ppi_p = os.path.join(PPI_DIR, f"{t_key_full}.pt")
        if not os.path.exists(ppi_p): continue
        
        # 将合法的路径直接存入 DataFrame，后续读取速度更快
        valid_rows.append({
            'pdb_name': pdb,
            'chain': chain,
            'esm_path': esm_p,
            'ppi_path': ppi_p
        })
        
    df_clean = pd.DataFrame(valid_rows)
    print(f">>> 清洗完毕！有效且特征对齐的样本数: {len(df_clean)}")
    
    print(">>> 3. 正在固定随机种子，划分训练集与验证集 (9:1)...")
    # 强制固定 random_state=42，保证每次切分绝对一致
    df_train, df_val = train_test_split(df_clean, test_size=0.1, random_state=42)
    
    train_save_path = f'{BASE_DIR}/go_train_cleaned.tsv'
    val_save_path = f'{BASE_DIR}/go_val_cleaned.tsv'
    
    df_train.to_csv(train_save_path, sep='\t', index=False)
    df_val.to_csv(val_save_path, sep='\t', index=False)
    
    print(f"✅ 训练集保存至: {train_save_path} (样本数: {len(df_train)})")
    print(f"✅ 验证集保存至: {val_save_path} (样本数: {len(df_val)})")
    print("数据清洗与划分已完成！可以进入下一步训练了。")

if __name__ == '__main__':
    clean_and_split_data()
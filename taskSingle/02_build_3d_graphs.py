import os
import pandas as pd
import requests
import time
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance_matrix
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning

# 忽略 Biopython 解析非标准 PDB 时产生的警告
warnings.simplefilter('ignore', BiopythonWarning)

# ================= 🔧 路径配置 =================
BASE_DIR = '/home/dongshali/fasta_data'
CSV_FILES = [
    './Task_Single_dataset/Task_DeepPPISP/352_name_seq_label.csv',
    './Task_Single_dataset/Task_DeepPPISP/Test_70.csv'
]
PDB_DIR = f'{BASE_DIR}/pdb_files'
ADJ_DIR = f'{BASE_DIR}/graph_adj_features' # 保存构建好的邻接矩阵
os.makedirs(PDB_DIR, exist_ok=True)
os.makedirs(ADJ_DIR, exist_ok=True)

def download_pdb(pdb_id, save_path, retries=3):
    """从 RCSB PDB 下载文件，包含重试机制"""
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                with open(save_path, 'w') as f:
                    f.write(resp.text)
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    return False

def extract_ca_adjacency(pdb_path, threshold=8.0):
    """解析 PDB，提取第一条链的 CA 原子坐标，构建邻接矩阵"""
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)
    
    # 默认提取第一个 Model 的第一条 Chain
    first_model = next(structure.get_models())
    first_chain = next(first_model.get_chains())
    
    ca_coords = []
    for residue in first_chain:
        # 跳过水分子等非标准氨基酸
        if residue.id[0] != ' ': continue 
        if 'CA' in residue:
            ca_coords.append(residue['CA'].get_coord())
            
    if not ca_coords:
        return None
        
    coords_array = np.array(ca_coords)
    # 计算所有 CA 原子两两之间的欧氏距离
    dist_mat = distance_matrix(coords_array, coords_array)
    # 构建邻接矩阵：距离 < 8Å 为 1（包含自环），否则为 0
    adj_mat = (dist_mat < threshold).astype(np.float32)
    return torch.tensor(adj_mat)

def main():
    print(">>> 1. 正在读取蛋白质序列列表...")
    raw_names = set()
    for f in CSV_FILES:
        if not os.path.exists(f): continue
        df = pd.read_csv(f)
        names = df.iloc[:, 0].apply(lambda x: str(x).strip().lstrip('>')).tolist()
        raw_names.update(names)
        
    print(f"✅ 共获取到 {len(raw_names)} 个待处理蛋白质。")
    
    success_count = 0
    for name in tqdm(raw_names, desc="Building 3D Graphs"):
        pdb_id = name[:4] # 提取前 4 位作为标准 PDB ID
        pdb_path = os.path.join(PDB_DIR, f"{name}.pdb")
        adj_path = os.path.join(ADJ_DIR, f"{name}.pt")
        
        # 1. 下载 PDB (如果不存在)
        if not os.path.exists(pdb_path):
            if not download_pdb(pdb_id, pdb_path):
                continue # 下载失败，跳过
                
        # 2. 构建图邻接矩阵并保存 (如果不存在)
        if not os.path.exists(adj_path):
            try:
                adj_tensor = extract_ca_adjacency(pdb_path)
                if adj_tensor is not None:
                    torch.save(adj_tensor, adj_path)
                    success_count += 1
            except Exception as e:
                continue # 格式损坏或解析失败，跳过
        else:
            success_count += 1
            
    print(f"\n🎉 空间邻接矩阵构建完成！成功处理: {success_count} / {len(raw_names)}")
    print(f"📁 矩阵文件保存在: {ADJ_DIR}")

if __name__ == '__main__':
    main()
import pandas as pd
import requests
import pickle
import os
import time
from tqdm import tqdm
from collections import Counter, defaultdict

# ================= 🔧 路径配置 =================
# 准确使用你划分好的 DeepPPISP 数据集
CSV_FILES = [
    './Task_Single_dataset/Task_DeepPPISP/352_name_seq_label.csv',
    './Task_Single_dataset/Task_DeepPPISP/Test_70.csv',
]

BASE_DIR = '/home/dongshali/fasta_data'
os.makedirs(f'{BASE_DIR}/meta_data', exist_ok=True)
GO_MAP_SAVE = f'{BASE_DIR}/meta_data/go_mapping.pkl'
NUM_GO_CLASSES = 150 
# ===============================================

def clean_to_pdb_id(raw_name):
    """
    【核心修复】：不管是有下划线(1F60_A)、连写(1F60A)还是破折号(1F60-A)，
    只要去掉 > 之后，PDB ID 永远是前 4 个字符！
    """
    name_str = str(raw_name).strip().lstrip('>')
    return name_str[:4].upper()

def get_go_terms_from_rcsb(pdb_id):
    """通过 RCSB PDB 官方 REST API 获取 GO 注释"""
    go_terms = set()
    entry_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        resp = requests.get(entry_url, timeout=10)
        if resp.status_code != 200:
            return go_terms
            
        data = resp.json()
        entity_ids = data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])
    except Exception:
        return go_terms

    for ent_id in entity_ids:
        entity_url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{ent_id}"
        try:
            ent_resp = requests.get(entity_url, timeout=10)
            if ent_resp.status_code == 200:
                annotations = ent_resp.json().get('rcsb_polymer_entity_annotation', [])
                for anno in annotations:
                    if anno.get('type') == 'GO':
                        go_id = anno.get('annotation_id')
                        if go_id:
                            go_terms.add(go_id)
        except Exception:
            continue

    return go_terms

def main():
    print(">>> 1. 正在读取你划分好的 DeepPPISP 数据集...")
    raw_names = set()
    for f in CSV_FILES:
        if not os.path.exists(f):
            print(f"⚠️ 找不到文件: {f}")
            continue
        df = pd.read_csv(f)
        # 获取第一列的所有蛋白质名称
        protein_col = df.columns[0] 
        names = df[protein_col].tolist()
        raw_names.update(names)
        
    raw_names = list(raw_names)
    print(f"✅ 共获取到 {len(raw_names)} 个唯一的蛋白质实体（包含不同链）。")
    
    print(f"\n>>> 2. 正在通过 RCSB PDB 官方 API 扒取真实 GO 标签...")
    raw_go_dict = defaultdict(set)
    
    for raw_name in tqdm(raw_names, desc="Fetching GO"):
        # 暴力截取前 4 位作为 PDB ID
        pdb_id = clean_to_pdb_id(raw_name)
        
        go_set = get_go_terms_from_rcsb(pdb_id)
        if go_set:
            # 注意：存入字典的键必须是原始名称（去掉 > ），以保证后续 Dataset 能精准匹配
            clean_raw_name = str(raw_name).strip().lstrip('>')
            raw_go_dict[clean_raw_name].update(go_set)
            
        time.sleep(0.1)

    print(f"✅ 成功获取 {len(raw_go_dict)} 个 PDB 结构的 GO 标签！")

    if len(raw_go_dict) == 0:
        print("❌ 未获取到任何数据，请检查网络。")
        return

    print("\n>>> 3. 正在统计并筛选高频 GO 功能...")
    all_go_terms = []
    for go_set in raw_go_dict.values():
        all_go_terms.extend(list(go_set))
        
    go_counter = Counter(all_go_terms)
    actual_num_classes = min(NUM_GO_CLASSES, len(go_counter))
    top_go = [go for go, count in go_counter.most_common(actual_num_classes)]
    
    go_to_idx = {go_id: idx for idx, go_id in enumerate(top_go)}
    
    final_mapping = {}
    for clean_raw_name, go_set in raw_go_dict.items():
        indices = [go_to_idx[go] for go in go_set if go in go_to_idx]
        if indices: 
            final_mapping[clean_raw_name] = indices
            
    save_data = {
        'go_to_idx': go_to_idx,
        'mapping': final_mapping,
        'num_classes': actual_num_classes
    }
    with open(GO_MAP_SAVE, 'wb') as f:
        pickle.dump(save_data, f)
        
    print(f"✅ Top-{actual_num_classes} GO 映射字典已保存至: {GO_MAP_SAVE}")
    print("🎉 第 0 步数据准备完美收官！不需要再重新切分数据，直接拿你的 CSV 进 DataLoader 即可！")

if __name__ == '__main__':
    main()
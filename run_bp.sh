#!/bin/bash
#SBATCH --job-name=protformer-site           # 任务名区分开
#SBATCH --output=%j.out          # 日志分开存
#SBATCH --error=%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --nodelist=gpu-1-8          # 你指定的节点
#SBATCH --gres=gpu:1                # 申请 1 张卡
#SBATCH --cpus-per-task=8           # 建议多给点 CPU 做数据加载

# 激活环境
source ~/miniconda3/bin/activate esm

# 运行 Python 脚本 (传入 BP 参数)
python 03_train_dual.py
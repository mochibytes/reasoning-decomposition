#!/bin/bash
#SBATCH --job-name=inv-mlp-patch
#SBATCH --partition=PARTITION
#SBATCH --account=ACCOUNT
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/inv-mlp-patch_%j.out
#SBATCH --error=logs/inv-mlp-patch_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

srun python3 train.py \
  --dataset inverse \
  --model mlp-patch \
  --rank 20 \
  --batch_size 256 \
  --data-workers 8 \
  --patch_baseline True \
  --patch_size 10

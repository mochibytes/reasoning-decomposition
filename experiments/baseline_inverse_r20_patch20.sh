#!/bin/bash
#SBATCH --job-name=baseline_inverse_r20_patch20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/baseline_inverse_r20_patch20_%j.out
#SBATCH --error=logs/baseline_inverse_r20_patch20_%j.err

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
  --patch_size 20

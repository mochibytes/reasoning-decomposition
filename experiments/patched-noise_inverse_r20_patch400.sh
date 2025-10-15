#!/bin/bash
#SBATCH --job-name=patched-noise_inverse_r20_patch400
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/patched-noise_inverse_r20_patch400_%j.out
#SBATCH --error=logs/patched-noise_inverse_r20_patch400_%j.err

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
  --patch_baseline False \
  --patch_size 400 \
  --train_num_steps 10000 \

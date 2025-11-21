#!/bin/bash
#SBATCH --job-name=attn_r20_patch20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/attn_heads2_test_%j.out
#SBATCH --error=logs/attn_heads2_test_%j.err

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
  --patch_size 20 \
  --train_num_steps 150000 \
  --num_heads 2 \
  --results_filename ~/reasoning-decomposition/results/patch_attn/attn_r20_p20_150k_heads2.csv

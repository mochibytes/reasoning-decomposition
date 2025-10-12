#!/bin/bash
#SBATCH --job-name=baseline_inverse_r20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/baseline_inverse_r20_ood_%j.out
#SBATCH --error=logs/baseline_inverse_r20_ood_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

srun python3 train.py \
  --dataset inverse \
  --model mlp \
  --rank 20 \
  --batch_size 256 \
  --data-workers 8 \
  --ood \
  --train_num_steps 150000 \
  --results_filename results/baseline_inverse_r20_ood.csv

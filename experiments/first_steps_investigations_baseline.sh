#!/bin/bash
#SBATCH --job-name=firststeps
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/baseline_inverse_r20_patch40_%j.out
#SBATCH --error=logs/baseline_inverse_r20_patch40_%j.err

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
  --save_and_sample_every 1 \
  --train_num_steps 1000 \
  --results_filename results/first_1000_steps_investigations_baseline.csv

#!/bin/bash
#SBATCH --job-name=baseline_sudoku
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/baseline_sudoku_%j.out
#SBATCH --error=logs/baseline_sudoku_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

srun python3 train.py \
  --dataset sudoku \
  --model sudoku \
  --batch_size 64 \
  --data-workers 4 \
  --save_and_sample_every 500 \
  --train_num_steps 50000 \
  --results_filename results/baseline_sudoku.csv

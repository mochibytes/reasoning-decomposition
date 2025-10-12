#!/bin/bash
#SBATCH --job-name=patched-noise_sudoku_patch27
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=120:00:00
#SBATCH --output=logs/patched-noise_sudoku_patch27_%j.out
#SBATCH --error=logs/patched-noise_sudoku_patch27_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

srun python3 train.py \
  --dataset sudoku \
  --model sudoku \
  --batch_size 64 \
  --data-workers 4 \
  --patch_size 27 \
  --save_and_sample_every 500 \
  --train_num_steps 50000 \
  --results_filename results/patched-noise_sudoku_patch27.csv

#!/bin/bash
#SBATCH --job-name=patched-noise_sudoku_patch27
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/patched-noise_sudoku_patch27_%j.out
#SBATCH --error=logs/patched-noise_sudoku_patch27_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

python3 ~/reasoning-decomposition/train.py \
  --dataset sudoku \
  --batch_size 64 \
  --model sudoku-patch \
  --patch_size 27 \
  --cond_mask True \
  --supervise-energy-landscape True \
  --use-innerloop-opt True \
  --results_filename ~/reasoning-decomposition/results/sudoku_patched-noise_patch27.csv \
  --save_and_sample_every 500 \
  --train_num_steps 50000

#!/bin/bash
#SBATCH --job-name=transformer_r20_patch20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/transformer_r20_patch20_%j.out
#SBATCH --error=logs/transformer_r20_patch20_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

srun python3 train.py \
  --dataset inverse \
  --model mlp-patch-t \
  --rank 20 \
  --batch_size 256 \
  --data-workers 8 \
  --patch_size 20 \
  --train_num_steps 150000 \
  --results_filename ~/reasoning-decomposition/results/patch_transformer/transformer_r20_p20_150k.csv

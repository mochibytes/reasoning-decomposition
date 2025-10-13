#!/bin/bash
#SBATCH --job-name=landscapes-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/landscapes-20_inverse_r20_patch20_%j.out
#SBATCH --error=logs/landscapes-20_inverse_r20_patch20_%j.err

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
  --patch_size 20 \
  --train_num_steps 50000 \
  --diffusion_steps 20 \
  --results_filename ~/reasoning-decomposition/results/simple-patched_inverse_r20_patch20_50000-steps_diffusion20.csv
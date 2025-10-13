#!/bin/bash
#SBATCH --job-name=uniformt_sharpness-sweep_inverse_r20_patch40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=logs/uniformt_sharpness-sweep_inverse_r20_patch40_%j.out
#SBATCH --error=logs/uniformt_sharpness-sweep_inverse_r20_patch40_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

# Sweep over different sharpness values
for sharpness in 2.0 1.0
do
  echo "Running with sharpness=${sharpness}"
  srun python3 train.py \
    --dataset inverse \
    --model mlp-patch \
    --rank 20 \
    --batch_size 256 \
    --cond_mask False \
    --data-workers 8 \
    --patch_baseline False \
    --patch_size 40 \
    --noising_scheme uniform-t \
    --sharpness ${sharpness} \
    --train_num_steps 100000 \
    --results_filename ~/reasoning-decomposition/results/uniform_t/sharpness_test_inverse_r20_patch40_sharp${sharpness}.csv
done

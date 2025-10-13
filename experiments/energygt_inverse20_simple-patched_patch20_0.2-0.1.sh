#!/bin/bash
#SBATCH --job-name=energygt_inverse20_simple-patched_patch40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/energygt_inverse20_simple-patched_patch40_%j.out
#SBATCH --error=logs/energygt_inverse20_simple-patched_patch40_%j.err

set -euo pipefail

mkdir -p logs

source ~/ired_env/bin/activate
cd ~/reasoning-decomposition

# Sweep over different sharpness values
for energy_gt in 0.2 0.1
do
  echo "Running with energy_gt=${energy_gt}"
  srun python3 train.py \
    --dataset inverse \
    --model mlp-patch \
    --rank 20 \
    --batch_size 256 \
    --cond_mask False \
    --data-workers 8 \
    --patch_baseline False \
    --patch_size 20 \
    --noising_scheme random \
    --energy_weight_gt ${energy_gt} \
    --train_num_steps 50000 \
    --results_filename ~/reasoning-decomposition/results/energy_gt/inverse20_simple-patched_patch20_energygt${energy_gt}.csv
done

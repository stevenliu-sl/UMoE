#!/bin/bash
#SBATCH --job-name=umoe-sweep
#SBATCH --account=coc
#SBATCH --gres=gpu:<TYPE>:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/storage/ice1/2/4/sliu736/umoe/logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your_gt_username@gatech.edu

source ~/UMoE/env.sh
bash ~/UMoE/pace/batch_sweep.sh "${CONFIG:-small-vanilla-UMoE-1024}" "${SIZES:-4 8 16 32 64}"

#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=finetune
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH -n 1



dataset="DongfuTingle/FeTaQA"
model="google/flan-t5-small"
model_type="t5"

# For offline finetuning
export WANDB_DISABLED=True

python finetune.py \
        --model_type $model_type \
        --dataset $dataset \
        --model $model 

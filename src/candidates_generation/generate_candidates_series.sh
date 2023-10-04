#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -n 1

CMD="sbatch"

# models=("google/flan-t5-small" "google/flan-t5-base" "google/flan-t5-large" "google/flan-t5-xl" "google/flan-t5-xxl")
# models=("lmsys/vicuna-33b-v1.3" "lmsys/vicuna-13b-v1.3" "lmsys/vicuna-7b-v1.3") # vicuna
models=("lmsys/vicuna-33b-v1.3") # vicuna-33b-v1.3 need two gpus
# models=("lmsys/vicuna-13b-v1.3" "lmsys/vicuna-7b-v1.3") # vicuna
# model_type="t5"
model_type="llama"
dataset="din0s/asqa"
dataset="DongfuTingle/FeTaQA"
# dataset="cosmos_qa"
# dataset="eli5"
set="test"
output_max_length=512
for model in "${models[@]}"; do
    ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"
done
# data_path=""
# python generate_candidates_by_gpt.py \
#     --task "long-form QA" \
#     --data_path $data_path \
#     --dataset $dataset \
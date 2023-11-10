#!/bin/bash
#SBATCH --job-name=generate_candidates_by_gpt
#SBATCH --time=24:00:00
#SBATCH --output=../../jobs/%j.out


# datasets=("GAIR/lima" "tatsu-lab/alpaca_farm:alpaca_instructions" "HuggingFaceH4/oasst1_en" "JosephusCheung/GuanacoDataset" "databricks/databricks-dolly-15k")
dataset=$1
task=$2
data_path=""
python generate_candidates_by_gpt.py \
    --task $task \
    --data_path $data_path \
    --dataset $dataset \
    --source_max_length 512 \
    --overwrite "False" 
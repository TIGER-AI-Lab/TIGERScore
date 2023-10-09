#!/bin/bash
#SBATCH --job-name=synthesis_distill_data
#SBATCH --time=48:00:00
#SBATCH --output=../../jobs/synthesis_distill_data/%j.out

xgptscore_mode="paraphrase"
version_key="distill"
model_name="gpt-4"
if [ ${model_name} == "gpt-4" ]; then
    export OPENAI_API_KEY=
    export OPENAI_API_BASE=""
    export OPENAI_API_TYPE="azure"
    export OPENAI_API_VERSION="2023-07-01-preview"
fi

IFS=$'\n'
# tasks=("translation" "long-form QA" "summarization" "data2text" "mathQA" "instruction-following")
tasks=("translation")
for task in ${tasks[@]}; do
    input_file="../../data/synthesis/${task}/train_data.json"
    echo task: $task
    python generate_synthesis_distill_data.py \
        --task $task \
        --xgptscore_mode $xgptscore_mode \
        --version_key $version_key \
        --model_name $model_name \
        --input_file $input_file \
        --source_max_length 512 \
        --overwrite "False" \
        --shuffle_file True \
        --max_size 0.15 \
        
done
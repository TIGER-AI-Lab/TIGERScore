#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --hint=memory_bound
#SBATCH --mem=60G
#SBATCH --gres=gpu:a6000:2
#SBATCH --qos=normal
#SBATCH -n 1

nvidia-smi
# candidates will be saved in ../../data/${dataset}/candidates/${decoding_method}/${model}.json
dataset=$1
set=$2
model_type=$3
model=$4
output_max_length=$5
no_instruction=$6
input_max_length=$7
decoding_method=$8
image2text=$9
start_idx=${10}
end_idx=${11}
data_dir="../../data"
dtype="float16"
num_candidates=5
num_beams=$num_candidates
num_beam_groups=$num_candidates
overwrite=False
inference_bs=1


if [ -z "$start_idx" ] && [ -z "$end_idx" ]; then
    echo "start_idx and end_idx are not provided, set to None"
else
    echo "start_idx: $start_idx"
    echo "end_idx: $end_idx"
fi
if [ -z "$output_max_length" ]; then
    output_max_length=300
    echo "output_max_length is not provided, set to $output_max_length"
else
    echo "output_max_length: $output_max_length"
fi

if [ -z "$input_max_length" ]; then
    input_max_length=300
    echo "input_max_length is not provided, set to $input_max_length"
else
    echo "input_max_length: $input_max_length"
fi

if [ -z "$image2text" ]; then
    image2text=False
    echo "image2text is not provided, set to $image2text"
else
    echo "image2text: $image2text"
fi
if [ -z "$no_instruction" ]; then
    no_instruction=False
    echo "no_instruction is not provided, set to $no_instruction"
else
    echo "no_instruction: $no_instruction"
fi
if [ -z "$decoding_method" ]; then
    decoding_method="top_p_sampling"
    echo "decoding_method is not provided, set to $decoding_method"
else
    echo "decoding_method: $decoding_method"
fi
python ./generate_candidates.py \
    --model_type $model_type \
    --model $model \
    --data_dir $data_dir \
    --dataset $dataset \
    --set $set \
    --num_return_sequences $num_candidates \
    --decoding_method $decoding_method \
    --inference_bs $inference_bs \
    --prompt_max_length $input_max_length \
    --output_max_length $output_max_length \
    --dtype $dtype \
    --num_beams $num_beams \
    --num_beam_groups $num_beam_groups \
    --no_repeat_ngram_size 3 \
    --start_idx "$start_idx" \
    --end_idx "$end_idx" \
    --overwrite $overwrite \
    --image2text "$image2text" \
    --no_instruction "$no_instruction" \
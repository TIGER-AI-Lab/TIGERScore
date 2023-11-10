#!/bin/bash
#SBATCH --job-name=test_llama
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --output=../../jobs/test_llama/%j.out
nvidia-smi

model_name="meta-llama/Llama-2-7b-hf"
outputs_dir=""

# outputs_dir="../../outputs"
checkpoint_name="model_len_1024_mix_real_world"
checkpoint_path="${outputs_dir}/${model_name}/${checkpoint_name}/checkpoint-532"

# human_score_names="rank"
# data_path="../../data_bak/lfqa/test.json"
# output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
# python test_llama_vllm.py \
#     --model_name_or_path ${checkpoint_path} \
#     --task "long-form QA" \
#     --data_path ${data_path} \
#     --output_path ${output_path} \
#     --batch_size 60 \
#     --human_score_names ${human_score_names} \
#     --overwrite

# task="instruction-following"
# human_score_names="gpt_rank_score"
# data_path="../../data_bak/llm-blender/mix-instruct/test_data_prepared_300.json"
# output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
# python test_llama_vllm.py \
#     --model_name_or_path ${checkpoint_path} \
#     --task ${task} \
#     --data_path ${data_path} \
#     --output_path ${output_path} \
#     --batch_size 60 \
#     --human_score_names ${human_score_names} \
#     --overwrite

# task="mathQA"
# human_score_names="accuracy"
# data_path="../../data_bak/mathqa/gsm8k_test_output_prepared.json"
# output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
# python test_llama_vllm.py \
#     --model_name_or_path ${checkpoint_path} \
#     --task ${task} \
#     --data_path ${data_path} \
#     --output_path ${output_path} \
#     --batch_size 60 \
#     --human_score_names ${human_score_names} \
#     --overwrite


# mtme test mqm
task="translation"
human_score_names="mqm"
data_path="../../data/wmt22/zh-en/eval_data.random_2.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite

# # sum test relevance
# task="summarization"
# human_score_names="coherence,consistency,fluency,relevance"
# output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
# python test_llama_vllm.py \
#     --model_name_or_path ${checkpoint_path} \
#     --task ${task} \
#     --data_path ${data_path} \
#     --output_path ${output_path} \
#     --batch_size 60 \
#     --human_score_names ${human_score_names} \
#     --overwrite

# # d2t test Correctness
# task="data2text"
# human_score_names="Correctness,DataCoverage,Fluency,Relevance,TextStructure"
# data_path="../../data_bak/webnlg/webnlg2020_gen_with_scores.json"
# output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
# python test_llama_vllm.py \
#     --model_name_or_path ${checkpoint_path} \
#     --task ${task} \
#     --data_path ${data_path} \
#     --output_path ${output_path} \
#     --batch_size 60 \
#     --human_score_names ${human_score_names} \
#     --overwrite

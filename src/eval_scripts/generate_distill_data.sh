#!/bin/bash
#SBATCH --job-name=generate_distill_data
#SBATCH -c 2
#SBATCH --time=24:00:00
#SBATCH --mem=10G
#SBATCH --output=../../jobs/%x/%j.out

version_key="distill"
overwrite=True
model_name="ChatGPT"
if [ ${model_name} == "gpt-4" ]; then
    export OPENAI_API_KEY=
    export OPENAI_API_BASE=""
    export OPENAI_API_TYPE="azure"
    export OPENAI_API_VERSION="2023-07-01-preview"
fi

# task='translation'
# xgptscore_mode="wmt_mqm"
# input_file="../../data/synthesis_min/translation/train_data.kb_txt.distill.syn_cand.json"
# python generate_distill_data.py \
#     --task ${task} \
#     --input_file ${input_file} \
#     --xgptscore_mode ${xgptscore_mode} \
#     --version_key ${version_key} \
#     --model_name ${model_name} \
#     --overwrite ${overwrite} \

# task='summarization'
# xgptscore_mode="align_score"
# input_file="../../data/synthesis_min/summarization/train_data.kb_txt.distill.syn_cand.json"
# python generate_distill_data.py \
#     --task ${task} \
#     --input_file ${input_file} \
#     --xgptscore_mode ${xgptscore_mode} \
#     --version_key ${version_key} \
#     --model_name ${model_name} \
#     --overwrite ${overwrite} \

# task='data2text'
# xgptscore_mode="d2t"
# input_file="../../data/synthesis_min/data2text/train_data.kb_txt.distill.syn_cand.json"
# python generate_distill_data.py \
#     --task ${task} \
#     --input_file ${input_file} \
#     --xgptscore_mode ${xgptscore_mode} \
#     --version_key ${version_key} \
#     --model_name ${model_name} \
#     --overwrite ${overwrite} \

# task='instruction-following'
# xgptscore_mode="instruction_following"
# input_file="../../data/synthesis_min/instruction-following/train_data.kb_txt.distill.syn_cand.json"
# python generate_distill_data.py \
#     --task ${task} \
#     --input_file ${input_file} \
#     --xgptscore_mode ${xgptscore_mode} \
#     --version_key ${version_key} \
#     --model_name ${model_name} \
#     --overwrite ${overwrite} \

task='long-form QA'
xgptscore_mode="longform_qa"
input_file="../../data/synthesis_min/long-form QA/train_data.kb_txt.distill.syn_cand.json"
python generate_distill_data.py \
    --task "${task}" \
    --input_file "${input_file}" \
    --xgptscore_mode ${xgptscore_mode} \
    --version_key ${version_key} \
    --model_name ${model_name} \
    --overwrite ${overwrite} \


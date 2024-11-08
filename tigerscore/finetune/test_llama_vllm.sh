#!/bin/bash
#SBATCH --job-name=test_llama
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --output=../../jobs/test_llama/%j.out
nvidia-smi


## Note
# please download the data in the working directory as indicated in the Data Preparation section in the read me
# quick command: gdown https://drive.google.com/uc?id=1DAjvig-A_57CuBvENLg8A2PycOaz9ZkT
## 

model_name="meta-llama/Llama-2-7b-hf"
outputs_dir=""

# outputs_dir="../../outputs"
checkpoint_name="ref"
# checkpoint_path="${outputs_dir}/${model_name}/${checkpoint_name}/checkpoint-532"
checkpoint_path="TIGER-Lab/TIGERScore-13B"

human_score_names="gpt_rank_score"
data_path="../../data/evaluation/lfqa/test_data_prepared.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task "long-form QA" \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite

task="instruction-following"
human_score_names="gpt_rank_score"
data_path="../../data/evaluation/instruct/just-eval-instruct/test_data_prepared.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite

task="mathQA"
human_score_names="accuracy"
data_path="../../data/evaluation/mathqa/gsm8k/test_data_prepared.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite


# mtme test mqm
task="translation"
human_score_names="mqm"
data_path="../../data/evaluation/translation/wmt22/zh-en/eval_data.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite

# sum test relevance
task="summarization"
human_score_names="coherence,consistency,fluency,relevance"
data_path="../../data/evaluation/summarization/summeval/test_data_prepared.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite

# d2t test Correctness
task="data2text"
human_score_names="Correctness,DataCoverage,Fluency,Relevance,TextStructure"
data_path="../../data/evaluation/d2t/webnlg_2020/test_data_prepared.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite


# storygen test human
task="storygen"
human_score_names="human"
data_path="../../data/evaluation/storygen/test_data_prepared.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}_test.output"
python test_llama_vllm.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --batch_size 60 \
    --human_score_names ${human_score_names} \
    --overwrite

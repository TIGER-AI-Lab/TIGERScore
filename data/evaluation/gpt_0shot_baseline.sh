#!/bin/bash
#SBATCH --job-name=test_llama
#SBATCH --time=24:00:00
#SBATCH --output=../../jobs/test_llama/%j.out


data_path="./lfqa/test_data_prepared.json"
python test_0shot_gpt.py \
    --task "long-form QA" \
    --data_path ${data_path}

task="instruction-following"
data_path="./instruct/just-eval-instruct/test_data_prepared.json"
python test_0shot_gpt.py \
    --task ${task} \
    --data_path ${data_path}

task="mathQA"
data_path="./mathqa/gsm8k/test_data_prepared.json"
python test_0shot_gpt.py \
    --task ${task} \
    --data_path ${data_path}


# mtme test mqm
task="translation"
data_path="./translation/wmt22/zh-en/eval_data.json"
python test_0shot_gpt.py \
    --task ${task} \
    --data_path ${data_path}

# sum test relevance
task="summarization"
data_path="./summarization/summaeval/test_data_prepared.json"
python test_0shot_gpt.py \
    --task ${task} \
    --data_path ${data_path}

# # d2t test Correctness
task="data2text"
data_path="./d2t/webnlg_2020/test_data_prepared.json"
python test_0shot_gpt.py \
    --task ${task} \
    --data_path ${data_path}

#!/bin/bash
#SBATCH --job-name=test_llama
#SBATCH --time=24:00:00
#SBATCH --output=../../jobs/test_llama/%j.out

human_score_names="gpt_rank_score"
data_path="./lfqa/test_data_prepared_ref.eval.json"
python gpt_0shot_baseline.py \
    --task "long-form QA" \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}

human_score_names="relevance,coherence,factuality,depth,engagement,safety"
task="instruction-following"
data_path="./instruct/just-eval-instruct/test_data_prepared_ref.eval.json"
python gpt_0shot_baseline.py \
    --task ${task} \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}

human_score_names="accuracy"
task="mathQA"
data_path="./mathqa/gsm8k/test_data_prepared_ref.eval.json"
python gpt_0shot_baseline.py \
    --task ${task} \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}


# mtme test mqm
human_score_names="mqm"
task="translation"
data_path="./translation/wmt22/zh-en/eval_data_ref.eval.json"
python gpt_0shot_baseline.py \
    --task ${task} \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}

# sum test relevance
task="summarization"
human_score_names="coherence,consistency,fluency,relevance"
data_path="./summarization/summeval/test_data_prepared_ref.eval.json"
python gpt_0shot_baseline.py \
    --task ${task} \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}


# # d2t test Correctness
task="data2text"
human_score_names="Correctness,DataCoverage,Fluency,Relevance,TextStructure"
data_path="./d2t/webnlg_2020/test_data_prepared_ref.eval.json"
python gpt_0shot_baseline.py \
    --task ${task} \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}


task="story_generation"
human_score_names="human"
data_path="./storygen/test_data_prepared_ref.eval.json"
python gpt_0shot_baseline.py \
    --task ${task} \
    --data_path ${data_path} \
    --human_score_names ${human_score_names}

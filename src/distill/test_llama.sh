#!/bin/bash
#SBATCH --job-name=test_llama
#SBATCH --gres=gpu:6000:1
#SBATCH --time=24:00:00
#SBATCH --output=../../jobs/test_llama/%j.out
nvidia-smi

model_name="meta-llama/Llama-2-7b-hf"
outputs_dir=""

# outputs_dir="../../outputs"
checkpoint_name="model_len_1024_mix_v2"
checkpoint_path="${outputs_dir}/${model_name}/${checkpoint_name}/checkpoint-best"
# task="translation"
# # finetune test
# data_path="/home//WorkSpace/ExplainableGPTScore/finetune_data/${task}/test.json" 

# BARTScore test
# data_path="/home//WorkSpace/ExplainableGPTScore/BARTScore/WMT/zh-en/final_p_with_xgptscore.test_llama_new.json"

# mtme test mqm
# task="translation"
# human_score_names="mqm,da"
# data_path="../../data/wmt22/zh-en/eval_data.random_2.json"

# sum test relevance
# task="summarization"
# human_score_names="coherence,consistency,fluency,relevance"
# data_path="../../BARTScore/SUM/SummEval/final_p_with_xgptscore.json"

# d2t test Correctness
# task="data2text"
# human_score_names="Correctness,DataCoverage,Fluency,Relevance,TextStructure"
# data_path="/home//WorkSpace/ExplainableGPTScore_bak/data/webnlg/webnlg2020_gen_with_scores.json"

# instruction-following
# rank
# data_path="/home//WorkSpace/ExplainableGPTScore_bak/data/databricks/databricks-dolly-15k/rank_eval_mid.json"

# task="instruction-following"
# human_score_names="gpt_rank_score"
# data_path="/home//WorkSpace/ExplainableGPTScore_bak/data/llm-blender/mix-instruct/test_data_prepared_300.json"

# long-form QA
### ATTENTION the space in the task name is not allowed,you need use --task "long-form QA" instead of --task ${task}
# task="long-form QA"
# human_score_names="rank"
# data_path="/home//WorkSpace/ExplainableGPTScore_bak/data/lfqa/test.json"

# Math QA
# accuracy
# task="mathQA"
# human_score_names="accuracy"
# data_path="/home//WorkSpace/ExplainableGPTScore_bak/gsm8k-ScRel/data/test_acc.json"

output_path="${data_path}.llama_2_7b_${checkpoint_name}.output"

# seems batch_size=1 is faster than batch_size=2 or higher
python test_llama.py \
    --model_name_or_path ${checkpoint_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --torch_dtype "bfloat16" \
    --batch_size 1 \
    --human_score_names ${human_score_names} \
    --model_max_length 1024 \
    --max_eval_input_length 512 \
    --max_eval_hyp_length 512 \
    --max_eval_output_length 1024 \
    --overwrite True \
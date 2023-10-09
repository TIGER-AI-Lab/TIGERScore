#!/bin/bash
#SBATCH --job-name=test_llama
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=2:00:00
#SBATCH --output=../../jobs/test_llama/%j.out

model_name="meta-llama/Llama-2-7b-hf"
outputs_dir="../../outputs"
checkpoint_name="model_len_768_lora_qv_old_data"
lora_path="${outputs_dir}/${model_name}/${checkpoint_name}/checkpoint-best"
task="summarization"
# # finetune test
# data_path="../../WorkSpace/ExplainableGPTScore/finetune_data/${task}/test.json" 

# BARTScore test
# data_path="../../WorkSpace/ExplainableGPTScore/BARTScore/WMT/zh-en/final_p_with_xgptscore.test_llama_new.json"

# mtme test
data_path="../../WorkSpace/ExplainableGPTScore_bak/BARTScore/SUM/SummEval/final_p_with_xgptscore.json"
output_path="${data_path}.llama_2_7b_${checkpoint_name}.output"

../../.conda/envs/llm_reranker/bin/python test_llama.py \
    --model_name_or_path ${model_name} \
    --load_lora ${lora_path} \
    --task ${task} \
    --data_path ${data_path} \
    --output_path ${output_path} \
    --torch_dtype "bfloat16" \
    --batch_size 2 \
    --model_max_length 1024 \
    --max_eval_input_length 256 \
    --max_eval_hyp_length 256 \
    --max_eval_output_length 1024 \
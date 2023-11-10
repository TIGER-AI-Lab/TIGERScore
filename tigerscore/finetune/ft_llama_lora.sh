#!/bin/bash
#SBATCH --job-name=ft_llama_lora
#SBATCH --gres=gpu:a6000:1
#SBATCH --time=24:00:00
#SBATCH --qos=general
#SBATCH --output=../../jobs/llama_finetune/%j.out

MASTER_PORT=4635
MODEL_DIR="meta-llama/Llama-2-7b-hf" # 13b
run_name="model_len_1024_lora_debug" # change this every time you run a new experiment
output_dir="../../outputs/${MODEL_DIR}/${run_name}"
# train_data_path="../../data/wmt/train_data.wmt_mqm.distill.format.json"
train_data_path="../../WorkSpace/ExplainableGPTScore/finetune_data/translation/train.json"
# train_data_path="../../WorkSpace/ExplainableGPTScore/finetune_data/translation/train/wmt18_zh-en.json"
mkdir -p ${output_dir}

# slurm system gpus can't connect to each other by default
# set the following environment variables to enable nccl
export NCCL_IB_DISABLE=1;
export NCCL_P2P_DISABLE=1;

export NCCL_DEBUG=INFO;
export NCCL_SOCKET_IFNAME=en,eth,em,bond;
export CXX=g++;

# batch_size = train_batch_size * gradient_accumulation_steps * num_gpus = 128
# epoch size: alpaca using 3 epochs for 52k data
# epoch size: translation data size, only 8k

../../.conda/envs/llm_reranker/bin/deepspeed \
    --num_gpus 1 \
    --num_nodes 1 \
    --master_port ${MASTER_PORT} \
    train.py \
    --model_name_or_path ${MODEL_DIR} \
    --train_data_path ${train_data_path} \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --model_max_length 1024 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 2 \
    --tf32 True \
    --deepspeed ds_llama_config.json \
    --run_name ${run_name} \
    --seed 42 \
    --is_lora True \

# lora Config
# lr: 3e-4
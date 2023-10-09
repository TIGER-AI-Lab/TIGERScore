#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=finetune
#SBATCH --output ../../jobs/finetune_base_models/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH --nodes=1
#SBATCH -n 1

model_type="t5"
model_name_or_path="google/flan-t5-large"
data_dir="../../data"
dataset="cosmos_qa"
train_file="${data_dir}/${dataset}/finetune_data.json"
eval_file="${data_dir}/${dataset}/validation_data.json"
with_instruction=True
run_name="ft_${dataset}"
learning_rate=1e-4
num_train_epochs=10
per_device_train_batch_size=2
per_device_eval_batch_size=8
gradient_accumulation_steps=16
max_grad_norm=1
input_max_length=512
output_max_length=256
optim="adafactor"
lr_scheduler_type="linear"
warmup_ratio=0.1
fp16=False
output_dir="../../finetuned_models/${model_name_or_path}/${run_name}"
cache_dir="../../hf_models"
localhost=$RANDOM # random port number
n_gpu=1
torchrun \
    --rdzv_backend=c10d \
    --rdzv_endpoint="localhost:${localhost}" \
    --nnodes 1 \
    --nproc_per_node ${n_gpu} \
    finetune_base_model.py \
    --model_type $model_type \
    --model_name_or_path $model_name_or_path \
    --data_dir $data_dir \
    --train_file $train_file \
    --eval_file $eval_file \
    --with_instruction $with_instruction \
    --run_name $run_name \
    --learning_rate $learning_rate \
    --optim $optim \
    --fp16 $fp16 \
    --lr_scheduler_type $lr_scheduler_type \
    --num_train_epochs $num_train_epochs \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --max_grad_norm $max_grad_norm \
    --input_max_length $input_max_length \
    --output_max_length $output_max_length \
    --output_dir $output_dir \
    --cache_dir $cache_dir \
    --report_to "wandb" \
    --logging_steps 2 \


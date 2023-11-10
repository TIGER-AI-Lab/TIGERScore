#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=eval_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:2080:1
#SBATCH --nodes=1
#SBATCH -n 2

data_dir="../../data"
# dataset="samsum,xsum,newsroom" # summarization
# dataset="wmt16/cs-en,wmt16/de-en,wmt16/tr-en,wmt17/fi-en,wmt18/zh-en" # translation
# dataset="totto,kasnerz/wikitabletext" # data2text
dataset="din0s/asqa,DongfuTingle/FeTaQA,cosmos_qa,eli5" # long-form QA 
# dataset="databricks/databricks-dolly-15k" 
# dataset="gsm8k:main,math_qa"

# dataset="common_gen,vicgalle/alpaca-gpt4,xnli/en,knkarthick/dialogsum"
set="test"
num_workers=1
metrics="bleu,rouge,bart_score,bart_score_cnn"
overwrite="True"
echo "dataset: $dataset"
echo "set: $set"
python eval_candidates.py \
    --data_dir $data_dir \
    --dataset $dataset \
    --set $set \
    --num_workers $num_workers \
    --metrics $metrics \
    --overwrite $overwrite
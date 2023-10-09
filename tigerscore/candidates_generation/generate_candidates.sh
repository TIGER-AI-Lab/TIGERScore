#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=generate_candidates
#SBATCH --output ../../jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH -n 1

CMD="sbatch"

# # # # <===================== Summarization =====================>

# model_type="pegasus"
# model="google/pegasus-gigaword"
# dataset="gigaword"
# set="train"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="pegasus"
# model="google/pegasus-newsroom"
# dataset="newsroom"
# set="train"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="pegasus"
# model="Yale-LILY/brio-cnndm-uncased"
# dataset="cnn_dailymail:3.0.0"
# set="train"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="pegasus"
# model="Yale-LILY/brio-xsum-cased"
# dataset="xsum"
# set="train"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="pegasus"
# model="google/pegasus-gigaword"
# dataset="gigaword"
# set="train"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"



# model_type="t5"
# model="philschmid/flan-t5-base-samsum"
# dataset="samsum"
# set="train"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="pegasus"
# model="google/pegasus-cnn_dailymail"
# dataset="cnn_dailymail:3.0.0"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# input_max_length=1024
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="pegasus"
# model="google/pegasus-xsum"
# dataset="xsum"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="bart"
# model="a1noack/bart-large-gigaword"
# dataset="gigaword"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="bart"
# model="nikhedward/bart-large-cnn-finetuned-multi-news"
# dataset="multi_news"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="t5"
# model="philschmid/flan-t5-base-samsum"
# dataset="samsum"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="pegasus"
# model="google/pegasus-billsum"
# dataset="billsum"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# # <===================== Translation =====================>

#### OPUS-MT Don't support device_map="auto"
# model_type="opus-mt"
# model="Helsinki-NLP/opus-mt-zh-en"
# dataset="wmt18:zh-en"
# set="train,test,validation"
# output_max_length=512
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="wmt16:de-en"
# set="train,test,validation"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="wmt16:cs-en"
# set="train,test,validation"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="wmt16:tr-en"
# set="train,test,validation"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="wmt17:fi-en"
# set="train,test,validation"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# # # # <===================== Data2Text =====================>
# model_type="t5"
# model="t5-large"
# dataset="dart"
# set="train" # "train,validation,test"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="bart"
# model="facebook/bart-large"
# dataset="dart"
# set="train" # "train,validation,test"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="t5"
# model="Barkavi/t5base_totto"
# dataset="totto"
# set="train"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="kasnerz/logicnlg"
# set="train,validation,test"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="kasnerz/wikitabletext"
# set="train"
# output_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# # # # <===================== Long-form QA =====================>

# model_type="bart"
# model="din0s/bart-base-asqa-cb"
# dataset="din0s/asqa"
# set="train"
# output_max_length=512
# no_instruction=True
# input_max_length=256
# decoding_method="beam_search"
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length" "$decoding_method"

# model_type="t5"
# model="../../finetuned_models/google/flan-t5-base/ft_cosmos_qa/checkpoint-best"
# dataset="maximedb/natural_questions"
# set="train,validation,test"
# output_max_length=256
# no_instruction=False
# input_max_length=128
# decoding_method="beam_search"
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length" "$decoding_method"

# model_type="llama"
# model="eachadea/vicuna-13b-1.1"
# dataset="DongfuTingle/FeTaQA"
# set="train"
# output_max_length=512
# no_instruction=False
# input_max_length=512
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length"

# model_type="t5"
# model="nikhilsk/t5-base-finetuned-eli5"
# dataset="eli5"
# set="train"
# output_max_length=512
# no_instruction=True
# input_max_length=300
# decoding_method="beam_search"
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length" "$decoding_method"

# model_type="llama"
# model="eachadea/vicuna-13b-1.1"
# dataset="cosmos_qa"
# set="train"
# output_max_length=128
# no_instruction=False
# input_max_length=256
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# <===================== MathQA =====================>
model_type="llama"
model="WizardLM/WizardMath-13B-V1.0" # verified
dataset="gsm8k:main"
set="test"
output_max_length=1024
no_instruction=True
${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="llama"
# model="WizardLM/WizardMath-13B-V1.0" 
# dataset="math_qa"
# set="train"
# output_max_length=1024
# no_instruction=True
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"

# model_type="t5"
# model="mrm8488/flan-t5-large-finetuned-gsm8k" # verified
# dataset="gsm8k:main"
# set="validation,test,train"
# output_max_length=512
# no_instruction=False
# input_max_length=256
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="llama"
# model="eachadea/vicuna-13b-1.1" 
# dataset="math_qa"
# set="validation,test,train"
# output_max_length=300
# no_instruction=False
# input_max_length=144
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="llama"
# model="eachadea/vicuna-13b-1.1"
# dataset="qwedsacf/grade-school-math-instructions"
# set="validation,test,train"
# output_max_length=384
# no_instruction=False
# input_max_length=256
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="llama"
# model="eachadea/vicuna-13b-1.1" 
# dataset="competition_math"
# set="validation,test,train"
# output_max_length=512
# no_instruction=False
# input_max_length=384
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="t5"
# model="google/flan-t5-xxl" 
# dataset="math_qa"
# set="validation,test,train"
# output_max_length=300
# no_instruction=False
# input_max_length=144
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="qwedsacf/grade-school-math-instructions"
# set="validation,test,train"
# output_max_length=384
# no_instruction=False
# input_max_length=256
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="gpt3"
# model="gpt-3.5-turbo" 
# dataset="competition_math"
# set="validation,test,train"
# output_max_length=512
# no_instruction=False
# input_max_length=384
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# <===================== Other =====================>

# model_type="t5"
# model="google/flan-t5-xxl" 
# dataset="common_gen"
# set="validation,test,train"
# output_max_length=300
# no_instruction=False
# input_max_length=300
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="t5"
# model="google/flan-t5-xxl"
# dataset="lighteval/lsat_qa/all"
# set="validation"
# output_max_length=300
# no_instruction=False
# input_max_length=384
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="llama"
# model="eachadea/vicuna-13b-1.1" 
# dataset="vicgalle/alpaca-gpt4"
# set="validation,test,train"
# output_max_length=300
# no_instruction=False
# input_max_length=384
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"

# model_type="t5"
# model="google/flan-t5-xxl" 
# dataset="xnli/en"
# set="validation,test,train"
# output_max_length=300
# no_instruction=False
# input_max_length=384
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction" "$input_max_length"


# <===================== Code =====================>
# model_type="llama"
# model="WizardLM/WizardCoder-15B-V1.0" 
# dataset="deepmind/code_contests"
# set="validation"
# output_max_length=2048
# no_instruction=False
# ${CMD} _generate_candidates.sh "$dataset" "$set" "$model_type" "$model" "$output_max_length" "$no_instruction"
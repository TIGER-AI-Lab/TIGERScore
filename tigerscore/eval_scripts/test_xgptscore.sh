model_name="chatgpt"

## Summarization ##
input_file="../../data/evaluation/summarization/summeval/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "summarization" \
    --model_name $model_name

## Translation ##
input_file="../../data/evaluation/translation/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "translation" \
    --model_name $model_name

## Data2Text ##
input_file="../../data/evaluation/d2t/webnlg_2020/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "data2text" \
    --model_name $model_name

## Instructions ##
input_file="../../data/evaluation/instructions/just-eval-instruct/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "instructions" \
    --model_name $model_name

## Long Form QA ##
input_file="../../data/evaluation/lfqa/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "long-form QA" \
    --model_name $model_name

## Math QA ##
input_file="../../data/evaluation/mathqa/gsm8k/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "mathQA" \
    --model_name $model_name

## Story Generation ##
input_file="../../data/evaluation/storygen/test_data_prepared.json"
python ./test_xgptscore.py  \
    --input_file $input_file \
    --task "story_generation" \
    --model_name $model_name
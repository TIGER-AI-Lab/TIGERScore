model_name="gpt-4"
if [ ${model_name} == "gpt-4" ]; then
    export OPENAI_API_KEY=
    export OPENAI_API_BASE=""
    export OPENAI_API_TYPE="azure"
    export OPENAI_API_VERSION="2023-07-01-preview"
fi


# python check_responses.py \
#     --input_file "/home//WorkSpace/ExplainableGPTScore/data/wmt/zh-en/train_data.wmt_mqm.distil_new_wmt_mqm_200.json" \
#     --output_file "/home//WorkSpace/ExplainableGPTScore/data/wmt/zh-en/train_data.wmt_mqm.distil_new_wmt_mqm_200.check.json" \
#     --model_name ${model_name} \

# python check_responses.py \
#     --input_file "/home//WorkSpace/ExplainableGPTScore_bak/data/sum/train_data.align_score.filter_v2.json" \
#     --output_file "/home//WorkSpace/ExplainableGPTScore_bak/data/sum/train_data.align_score.filter_v2.check.json" \
#     --model_name ${model_name} \


python check_responses.py \
    --input_file "../../data/wmt/train_data.wmt_mqm.distill_new_wmt_mqm.json" \
    --output_file "../../data/wmt/train_data.wmt_mqm.distill_new_wmt_mqm.${model_name}.check.json" \
    --model_name ${model_name} \
model_name="gpt-4"
if [ ${model_name} == "gpt-4" ]; then
    export OPENAI_API_KEY=
    export OPENAI_API_BASE=""
    export OPENAI_API_TYPE="azure"
    export OPENAI_API_VERSION="2023-07-01-preview"
fi

python lfqa_gpt_rate.py \
    --input_file "../../data_bak/lfqa/test.json" \
    --output_file "../../data_bak/lfqa/test.${model_name}.rank.json" \
    --model_name ${model_name} \
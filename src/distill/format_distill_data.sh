DATA_DIR="../../data"

# # transllation
# INPUT_FILE="${DATA_DIR}/wmt/train_data.wmt_mqm.distill_new_wmt_mqm.json"
# OUTPUT_FILE="${DATA_DIR}/wmt/train_data.wmt_mqm.distill_new_wmt_mqm.format_txt.json"
# python format_distill_data.py --task "translation" --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \

# # summarization
# INPUT_FILE="${DATA_DIR}/WorkSpace/ExplainableGPTScore_bak/data/sum/train_data.align_score.filter_v2.json"
# OUTPUT_FILE="${DATA_DIR}/WorkSpace/ExplainableGPTScore_bak/data/sum/train_data.align_score.filter_v2.format_txt.json"
# python format_distill_data.py --task "summarization" \
#     --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#     --max_eval_input_length 400 --max_eval_hyp_length 300 --max_eval_output_length 400 \

# # data2text
# INPUT_FILE="${DATA_DIR}/WorkSpace/ExplainableGPTScore_bak/data/d2t/train_data.d2t.filter_v1.json"
# OUTPUT_FILE="${DATA_DIR}/WorkSpace/ExplainableGPTScore_bak/data/d2t/train_data.d2t.filter_v1.format_txt.json"
# python format_distill_data.py --task "data2text" --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#     --max_eval_input_length 400 --max_eval_hyp_length 400 --max_eval_output_length 400 \
# # long-form QA

# # SEScore3 zh-en debug
# INPUT_FILE="${DATA_DIR}/WorkSpace/ExplainableGPTScore/data/sescore3/sescore3_zh_en_llama_formatted_data.json"
# OUTPUT_FILE="${DATA_DIR}/WorkSpace/ExplainableGPTScore/data/sescore3/sescore3_zh_en_llama_formatted_data.format_txt.json"
# python format_distill_data.py --task "translation" --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#     # --max_eval_input_length 400 --max_eval_hyp_length 400 --max_eval_output_length 400 \

# # summarization v3
# INPUT_FILE="../../data/sum/train_data.align_score.filter_v3.json"
# OUTPUT_FILE="../../data/sum/train_data.align_score.filter_v3.format_txt.json"
# python format_distill_data.py --task "summarization" \
#     --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#     --max_eval_input_length 400 --max_eval_hyp_length 300 --max_eval_output_length 400 \


IFS=$'\n'
tasks=("translation" "long-form QA" "summarization" "data2text" "instruction-following")
for task in ${tasks[@]}; do
    INPUT_FILE="../../data/real_world/${task}.json"
    OUTPUT_FILE="../../data/real_world/${task}.format_txt.json"
    python format_distill_data.py --task ${task} \
        --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
        --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400
done

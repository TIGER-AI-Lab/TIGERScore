# INPUT_FILE="../../data/train_mix.check.clean.jsonl"
# OUTPUT_FILE="../../data/train_mix.check.clean.format_v2.json"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/train_mix.jsonl"
# OUTPUT_FILE="../../data/train_mix.format_v2.json"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# tasks=('data2text' 'instruction-following' 'long-form QA' 'mathQA' 'summarization' 'translation')
# for task in "${tasks[@]}"; do
#     INPUT_FILE="../../data/train_mix.${task}.jsonl"
#     OUTPUT_FILE="../../data/train_mix.${task}.format_v2.json"
#     python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#             --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400
# done

INPUT_FILE="../../data/train_mix.check.clean.mathQA.jsonl"
OUTPUT_FILE="../../data/train_mix.check.clean.mathQA.format_v2.json"
python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
        --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/additional/alpaca_cleaned/new_alpaca_cleaned.v2.8k.gen.jsonl"
# OUTPUT_FILE="../../data/additional/alpaca_cleaned/new_alpaca_cleaned.v2.8k.gen.format_v2.json"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/additional/alpaca_cleaned/new_alpaca_cleaned.v2.2k.jsonl"
# OUTPUT_FILE="../../data/additional/alpaca_cleaned/new_alpaca_cleaned.v2.2k.format_v2.json"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/new_std_400s_m_200s_l_1100s_i3-32k.check.clean.jsonl"
# OUTPUT_FILE="../../data/new_std_400s_m_200s_l_1100s_i3-32k.check.clean.format_v2.jsonl"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.story.ref.extracted.jsonl"
# OUTPUT_FILE="../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.story.ref.extracted.format_v2.jsonl"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.story.1k.gen.extracted.jsonl"
# OUTPUT_FILE="../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.story.1k.gen.extracted.format_v2.jsonl"
# # INPUT_FILE="TIGERScore/data/32k_final.json"
# # OUTPUT_FILE="TIGERScore/data/32k_final_distill.json"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400

# INPUT_FILE="../../data/additional/metamath/metamath.1k.ref.extracted.jsonl"
# OUTPUT_FILE="../../data/additional/metamath/metamath.1k.ref.extracted.format_v2.jsonl"
# # INPUT_FILE="TIGERScore/data/32k_final.json"
# # OUTPUT_FILE="TIGERScore/data/32k_final_distill.json"
# python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
#         --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400
INPUT_FILE="/home/dongfu/WorkSpace/TIGERScore/data/32k_final.json"
OUTPUT_FILE="/home/dongfu/WorkSpace/TIGERScore/data/32k_final_distill.json"
python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
        --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400
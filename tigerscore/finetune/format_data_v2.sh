INPUT_FILE="../../data/train_mix.check.clean.jsonl"
OUTPUT_FILE="../../data/train_mix.check.clean.format_v2.json"
python format_data_v2.py --input_file "${INPUT_FILE}" --output_file "${OUTPUT_FILE}" \
        --max_eval_input_length 600 --max_eval_hyp_length 400 --max_eval_output_length 400
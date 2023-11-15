# python check_data.py \
#     --input_file "../../data/train_synthetic.jsonl" \
#     --output_file "../../data/train_synthetic.check.jsonl" \
#     --model_name "gpt-4" \
#     --num_procs 5


python check_data.py \
    --input_file "../../data/train_mix.jsonl" \
    --output_file "../../data/train_mix.check_ChatGPT.jsonl" \
    --model_name "ChatGPT"
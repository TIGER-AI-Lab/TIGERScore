# python generate_inst_synthetic_data.py \
#     --input_file "../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.10k.jsonl" \
#     --output_file "../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.10k.gen.jsonl" \
#     --model_name "gpt-4" \
#     --num_samples 8000

python generate_inst_synthetic_data.py \
    --input_file "../../data/additional/metamath/metamath.8k.jsonl" \
    --output_file "../../data/additional/metamath/metamath.8k.gen.jsonl" \
    --model_name "gpt-4" \
    --num_samples 5

# python generate_inst_synthetic_data.py \
#     --input_file "../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.story.1k.jsonl" \
#     --output_file "../../data/additional/alpaca_cleaned/alpaca_cleaned.v2.story.1k.gen.jsonl" \
#     --model_name "gpt-4" \
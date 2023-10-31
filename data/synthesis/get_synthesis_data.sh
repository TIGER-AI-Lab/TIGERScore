# python get_synthesis_data.py --task "translation" \
#     --max_size_per_ds 200 --max_input_length 256 --max_output_length 256
# python get_synthesis_data.py --task "summarization" \
#     --max_size_per_ds 200 --max_input_length 1024 --max_output_length 256
# python get_synthesis_data.py --task "data2text" \
#     --max_size_per_ds 200 --max_input_length 384 --max_output_length 256
# python get_synthesis_data.py --task "mathQA" \
#     --max_size_per_ds 200 --max_input_length 128 --max_output_length 384
# python get_synthesis_data.py --task "instruction-following" \
#     --max_size_per_ds 200 --max_input_length 384 --max_output_length 384
# python get_synthesis_data.py --task "long-form QA" \
#     --max_size_per_ds 200 --max_input_length 256 --max_output_length 384


python get_synthesis_data.py --task "translation" \
    --max_size_per_ds 1000 --max_input_length 256 --max_output_length 256
python get_synthesis_data.py --task "summarization" \
    --max_size_per_ds 1000 --max_input_length 1024 --max_output_length 256
python get_synthesis_data.py --task "data2text" \
    --max_size_per_ds 1000 --max_input_length 384 --max_output_length 256
python get_synthesis_data.py --task "mathQA" \
    --max_size_per_ds 1000 --max_input_length 128 --max_output_length 384
python get_synthesis_data.py --task "instruction-following" \
    --max_size_per_ds 1000 --max_input_length 384 --max_output_length 384
python get_synthesis_data.py --task "long-form QA" \
    --max_size_per_ds 1000 --max_input_length 256 --max_output_length 384

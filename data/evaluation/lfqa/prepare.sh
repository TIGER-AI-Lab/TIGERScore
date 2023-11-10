dataset_root_path=".."
dataset_names=("eli5" "cosmos_qa" "din0s/asqa" "DongfuTingle/FeTaQA")
dataset_split="test"
model_list="ChatGPT,vicuna-33b-v1.3,vicuna-13b-v1.3,vicuna-7b-v1.3"

for dataset_name in ${dataset_names[@]}; do
    python prepare.py \
        --dataset_root_path ${dataset_root_path} \
        --dataset_name ${dataset_name} \
        --dataset_split ${dataset_split} \
        --model_list ${model_list}
done
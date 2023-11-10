from datasets import load_dataset
import json

dataset = load_dataset("llm-blender/mix-instruct")

new_data = []
test_data_inputs = dataset['test']['input']
for x in dataset['validation']:
    if x['input'] in test_data_inputs:
        continue
    if x['id'] == 'unified_chip2/85587':
        # a cornor case
        continue
    del x['cmp_results']
    new_data.append(x)
print(len(new_data))
with open("./train_data_prepared.json", 'w') as f:
    json.dump(new_data, f, indent=4, ensure_ascii=False)
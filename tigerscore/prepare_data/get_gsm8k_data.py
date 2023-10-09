from pathlib import Path
import json
import os
import sys

datas = []
tmp_data = {}
for split in ["train", "test"]:
    with open(f'/home//WorkSpace/ExplainableGPTScore_bak/gsm8k-ScRel/data/{split}_use.jsonl', 'r') as f:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            tmp_data[line["query"]] = {
                "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request. Let's think step by step.",
                # "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
                "input": line["query"],
                "output": line["response"],
                "candidates": []
            }

jsonl_dir = '/home//WorkSpace/ExplainableGPTScore_bak/gsm8k-ScRel/data/rft/'

for root, dirs, files in os.walk(jsonl_dir):
    for file in files:
        with open(jsonl_dir+file, 'r') as f:
            data = f.readlines()
            for line in data:
                line = json.loads(line)
                if line["query"] in tmp_data:
                    tmp_data[line["query"]]["candidates"].append({
                        "model": f"{file.split('.')[0]}",
                        "decoding_method": "greedy",
                        "text": line["response"],
                        "scores": {}
                    })
for k, v in tmp_data.items():
    if len(v["candidates"]) != 0:
        datas.append(v)
# datas = sorted(datas, key=lambda x: x["id"])

output_file = Path('/home//WorkSpace/ExplainableGPTScore_bak/gsm8k-ScRel/data/data_gen.json')
with open(output_file, 'w') as f:
    json.dump(datas, f, indent=4,ensure_ascii=False)
print(len(datas))
print(datas[0])
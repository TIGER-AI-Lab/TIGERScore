# %%
import json,os,sys
import numpy as np

roc_input_file_path = "/home//WorkSpace/ExplainableGPTScore_bak/OpenMEVA/benchmark/data/mans_roc.json"
with open(roc_input_file_path, 'r') as f:
    data = json.load(f)

new_data = []
for key, value in data.items():
    cands = []
    for _k, _v in value["gen"].items():
        cands.append({
            "model": _k,
            "decoding_method": "greedy",
            "text": _v["text"],
            "score": {"human":np.mean(_v["score"])},
        })
    new_data.append({
        "id": key,
        "instruction": "Generate a reasonable ending for the following story.",
        "input": value["prompt"],
        "output": value["gold_response"],
        "candidates": cands,
        "data_source": "roc",
    })
    
wp_input_file_path = "/home//WorkSpace/ExplainableGPTScore_bak/OpenMEVA/benchmark/data/mans_wp.json"
with open(wp_input_file_path, 'r') as f:
    data = json.load(f)
for key, value in data.items():
    cands = []
    for _k, _v in value["gen"].items():
        cands.append({
            "model": _k,
            "decoding_method": "greedy",
            "text": _v["text"],
            "score": {"human":np.mean(_v["score"])},
        })
    new_data.append({
        "id": key,
        "instruction": "Generate a reasonable ending for the following story.",
        "input": value["prompt"],
        "output": value["gold_response"],
        "candidates": cands,
        "data_source": "wp",
    })
    
output_file_path = "/home//WorkSpace/ExplainableGPTScore_bak/data/storygen/test.json"
with open(output_file_path, 'w') as f:
    json.dump(new_data, f, indent=4,ensure_ascii=False)
    



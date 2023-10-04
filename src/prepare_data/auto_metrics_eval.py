import sys
import os
from pathlib import Path
import json
sys.path.append("/home//WorkSpace/ExplainableGPTScore_bak/src")
from common.evaluation import overall_eval

output_file = "/home//WorkSpace/ExplainableGPTScore_bak/data/summeval/train_data_prepared.json"

with open(output_file, "r") as f:
    datas = json.load(f)
candidates = []
targets = []
candidates = [[candidate["text"] for candidate in data["candidates"] ]for data in datas]
targets = [data["output"] for data in datas]
print(candidates[0], targets[0])
metrics = ["bleu", "rouge1","rouge2","rougeL","bart_score","bart_score_cnn"]
# DS = datas
# DS = {x['id']: x for x in DS}
# pure_candidates = [[x['text'] for x in item['candidates']] for item in candidates]
# targets = [DS[x['id']]['output'] for x in candidates]
# evaluate
scores = overall_eval(candidates, targets, metrics, 1)

# scores = overall_eval( candidates,targets, metrics=metrics)

for metric in metrics:
    scores_metric = scores[metric]
    for i, sample in enumerate(datas):
        _sample_cands = sample['candidates']
        for j, _sample_cand in enumerate(_sample_cands):
            _sample_cand['scores'][metric] = scores_metric[i][j]
            
with open(output_file, "w") as f:
    json.dump(datas, f, indent=4)
import json
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
task = "data2text"
task_data_map = {
    "data2text": "/home//WorkSpace/ExplainableGPTScore_bak/data/d2t/train_data.align_score.filter_v1.json"
}
with open(task_data_map[task], 'r') as f:
    data = json.load(f)

import sys
sys.path.append("/home//WorkSpace/ExplainableGPTScore_bak/src")
from xgptscore.constants import EVAL_ASPECTS
xgptscore_types = ['xgptscore']
xgptscore_types = xgptscore_types + list(EVAL_ASPECTS[task].keys())
# xgptscore_types.remove('Coherence')
print(xgptscore_types)
scores = defaultdict(list)
datasets = defaultdict(int)
datasets_can = defaultdict(int)
for item in data:
    datasets[item['dataset']] += 1
    for cand in item['candidates']:
        datasets_can[item['dataset']] += 1
        cand_scores = defaultdict(int)
        if isinstance(cand['responses'][-1], dict):
            for error in cand['responses'][-1]['errors'].values():
                if error['score_reduction']: # is Not None and is not 0
                    cand_scores[error['error_aspect']] -= error['score_reduction']
                    cand_scores['xgptscore'] -= error['score_reduction']
        for key in xgptscore_types:
            scores[key].append(cand_scores[key])

        
fig, axes = plt.subplots(len(xgptscore_types) // 2 + 1, 2)
for i, key in enumerate(scores):
    ax = axes[i // 2, i % 2]
    # make bins' width wider 
    ax.hist(scores[key], bins=40)
    ax.set_title(key)

plt.tight_layout()
plt.show()
print(datasets)
print(datasets_can)
print(len([x for x in scores['xgptscore']]))
print(len([x for x in scores['xgptscore'] if x == 0]) / len(scores['xgptscore']))

#filter
from fuzzywuzzy import fuzz
from copy import deepcopy
new_data = []
tmp_data = deepcopy(data)
# merge coherence and fluency
ratios = defaultdict(list)
ratios_75 = {
    'totto': 47,
    'kasnerz/wikitabletext': 42,
    'webnlg': 21,
}
ratios_50 = {
    'totto': 68.0,
    'kasnerz/wikitabletext': 73,
    'webnlg': 35,
}
for item in tmp_data:
    new_cands = []
    for cand in item['candidates']:
        put_in = True
        cand_scores = defaultdict(int)
        if isinstance(cand['responses'][-1], dict):
            for error in cand['responses'][-1]['errors'].values():
                if not isinstance(error, dict):
                    put_in = False
                for key in error.keys():
                    if key not in ["error_aspect", "severity", "explanation", "error_location", "score_reduction"]:
                        put_in = False
                        # print(key)
                for value in error.values():
                    if value is None:
                        put_in = False

                if error["error_aspect"] in ["Coherence", "Fluency", "Coherence, Fluency"]:
                    error["error_aspect"] = "Fluency"

                if error['score_reduction'] is None or error['score_reduction'] < 0:
                    put_in = False
                try:
                    float(error['score_reduction'])
                except:
                    put_in = False
                
                if error["error_aspect"] not in EVAL_ASPECTS[task]:
                    put_in = False
                    # print(error["error_aspect"])
                if error["severity"] not in ["Major", "Minor"]:
                    put_in = False
                    # print(error["severity"])
                if error["explanation"] in ["", " ","N/A","None"]:
                    put_in = False
                    # print(error["explanation"])
                
                # if error["error_location"] is not None and error["error_location"] not in item["input"]:
                #     put_in = False
                #     # print(error["error_location"])
                # ratios[item["dataset"]].append(fuzz.token_set_ratio(item["input"], error["error_location"]))
                if fuzz.token_set_ratio(item["input"], error["error_location"]) < ratios_75[item["dataset"]]:
                    put_in = False
        else:
            put_in = False
        if put_in:
            new_cands.append(cand)
    if len(new_cands) > 0:
        item['candidates'] = new_cands
        new_data.append(item)
    
print(f"Old data: {len(data)}")
print(f"Old data Candidates: {sum([len(item['candidates']) for item in data])}")
print(f"New data: {len(new_data)}")
print(f"New data Candidates: {sum([len(item['candidates']) for item in new_data])}")

new_output_file = "/home//WorkSpace/ExplainableGPTScore_bak/data/d2t/train_data.align_score.filter_v1.json"
with open(new_output_file, 'w') as f:
    json.dump(new_data, f, indent=4)
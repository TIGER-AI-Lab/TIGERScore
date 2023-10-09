"""
Usage:
    python test_xgptscore_sum.py
"""
import json
import random
import logging
import sys
import numpy as np
import pickle
from pathlib import Path
from utils import MyCorrelation
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.xgptscore import xgptscore
from itertools import chain
from collections import Counter
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json_per_aspect
from xgptscore.constants import EVAL_ASPECTS
logging.basicConfig(level=logging.INFO)

# params
task='data2text'
bart_version="D2T"
dataset="SFHOT"
data_dir="../../BARTScore"
xgptscore_mode="d2t"
version_key=f"{xgptscore_mode}.ref.end_1_5"
our_score_name="xgptscore"
model_name="ChatGPT"
overwrite=False
max_size=200 # set to None to use all examples
num_sys=2
if isinstance(max_size, int) and max_size > 0:
    version_key = f"{version_key}_{max_size}"

# load data
input_file=Path(f"{data_dir}/{bart_version}/{dataset}/final_p_with_xgptscore.json")
if version_key:
    output_file = input_file.with_suffix(f".{version_key}.json")
else:
    output_file = input_file.with_suffix(f".default.json")

if not output_file.exists() or overwrite:
    # Load and shuffle data
    logging.info("Loading from {}".format(input_file))
    with open(input_file, "r") as f:
        items = json.load(f)
    if isinstance(max_size, int) and max_size > 0:
        items = items[:max_size]
    # random will cause wrong results
    
    # Data processing
    xgptitems = []
    for item in items:
        item['candidates'] = [
            {
                "model": "reference",
                "decoding_method": "greedy",
                "text": item['output'] if isinstance(item['output'], str) else item['output'][0],
                "scores": {},
            }
        ]
        xgptitems.append(XPGTItem(
            task=task,
            instruction=item['instruction'],
            input=item['input'],
            # ref_output=item['output'],
            ref_output="N/A",
            hypo_output=item['output'] if isinstance(item['output'], str) else item['output'][0],
        ))
    # Run xgptscore
    result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name,num_workers=5)
    idx = 0
    aspects = EVAL_ASPECTS[task].keys()
    score_dict = {"xgptscore_"+aspect: 0 for aspect in aspects}
    for item in items:
        for cand in item['candidates']:      
            cand['responses'] = result['round_completions'][idx]
            cand['messages_records'] = result['messages_records'][idx]
            xgptscore_ans = get_xgptscore_from_json_per_aspect(cand['responses'][-1])
            if xgptscore_ans is None:
                logging.info(f"XGPTScore failed for {cand['text']}")
                # cand['scores']['xgptscore'] = None
            else:
                cand['scores'].update(score_dict)
                cand['scores'].update(xgptscore_ans) 
            idx += 1
        
    # Save results
    with open(output_file, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(output_file))
else:
    logging.info("Loading existing results from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)


# by system
# Compute bias
xgptscores = []
for item in items:
    for cand in item['candidates']:
        if our_score_name in cand['scores']:
            xgptscores.append(cand['scores'][our_score_name])

print(f"Mean: {np.mean(xgptscores)}")
print(f"Distribution: {Counter(xgptscores)}")
print(f"Std: {np.std(xgptscores)}")
print(f"Max: {np.min(xgptscores)}")
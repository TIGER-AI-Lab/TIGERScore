"""
Usage:
    python generate_distill_d2t.py
"""
import json
import random
import logging
import sys
from pathlib import Path
from utils import MyCorrelation
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.xgptscore import xgptscore
from itertools import chain
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json_per_aspect
from xgptscore.constants import EVAL_ASPECTS
logging.basicConfig(level=logging.INFO)

# params
task='instruction-following'
xgptscore_mode="instruction_following"
version_key=f"{xgptscore_mode}_new.distill"
human_score_names=[]
model_name="ChatGPT"
overwrite=False
max_size=None # set to None to use all examples
if isinstance(max_size, int) and max_size > 0:
    version_key = f"{version_key}_{max_size}"

# load data
split="train"
data_dir="../../data"
input_file=Path(f"{data_dir}/ins/train_data_new.json")
if version_key:
    output_file = input_file.with_suffix(f".{version_key}.json")
else:
    output_file = input_file.with_suffix(f".default.json")
logging.info("Loading from {}".format(input_file))
with open(input_file, "r") as f:
    items = json.load(f)
if isinstance(max_size, int) and max_size > 0:
    items = items[:max_size]

xgptitems = []
for item in items:
    for cand in item['candidates']:
        xgptitems.append(XPGTItem(
            task=task,
            instruction=item['instruction'],
            input=item['input'],
            ref_output=item['output'],
            hypo_output=cand['text']
        ))

if not output_file.exists() or overwrite:
    logging.info("Running xgptscore")
    # run xgptscore
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
    with open(output_file, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(output_file))
else:
    logging.info("Loading from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)

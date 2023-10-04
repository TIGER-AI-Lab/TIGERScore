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
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json
from xgptscore.constants import EVAL_ASPECTS
logging.basicConfig(level=logging.WARNING)

# params
task='mathQA'
xgptscore_mode="mathqa"
version_key=f"{xgptscore_mode}.distill"
human_score_names=[]
model_name="ChatGPT"
overwrite=True
max_size=200 # set to None to use all examples
if isinstance(max_size, int) and max_size > 0:
    version_key = f"{model_name}_{version_key}_{max_size}"

# load data
split="train"
data_dir="../../data_bak"
input_file=Path(f"{data_dir}/mathqa/gsm8k_test_output_prepared.ChatGPT_score.json")
if version_key:
    output_file = input_file.with_suffix(f".{version_key}.json")
else:
    output_file = input_file.with_suffix(f".default.json")
logging.warning("Loading from {}".format(input_file))
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
    logging.warning("Running xgptscore")
    # run xgptscore
    result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name,num_workers=5)
    idx = 0
    for item in items:
        for cand in item['candidates']:      
            cand['responses'] = result['round_completions'][idx]
            cand['messages_records'] = result['messages_records'][idx]
            cand['scores']['xgptscore'] = get_xgptscore_from_json(cand['responses'][-1])
            idx += 1
    with open(output_file, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        logging.warning("Saved to {}".format(output_file))
else:
    logging.warning("Loading from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)

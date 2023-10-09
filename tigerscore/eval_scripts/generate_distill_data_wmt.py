"""
Usage:
    python test_xgptscore_wmt.py
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
logging.basicConfig(level=logging.INFO)

# params
task='translation'
xgptscore_mode="wmt_mqm"
version_key=f"{xgptscore_mode}.distill_new_wmt_mqm"
human_score_names=["mqm", "da"]
model_name="ChatGPT"
overwrite=True
max_size=None # set to None to use all examples
if isinstance(max_size, int) and max_size > 0:
    version_key = f"{version_key}_{max_size}"

# load data
split="train"
data_dir="../../data"
input_file=Path(f"{data_dir}/wmt/ru-en/{split}_data.json")
if version_key:
    output_file = input_file.with_suffix(f".{version_key}.json")
else:
    output_file = input_file.with_suffix(f".default.json")
logging.info("Loading from {}".format(input_file))
with open(input_file, "r") as f:
    items = json.load(f)
random.seed(42)
random.shuffle(items)
if isinstance(max_size, int) and max_size > 0:
    items = items[:max_size]

xgptitems = []
for item in items:
    for cand in item['candidates']:
        xgptitems.append(XPGTItem(
            task=task,
            instruction=item['instruction'],
            input=item['input'],
            ref_output=item['refs'],
            hypo_output=cand['text']
        ))

if not output_file.exists() or overwrite:
    logging.info("Running xgptscore")
    # run xgptscore
    result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name)
    idx = 0
    for item in items:
        for cand in item['candidates']:
            cand['responses'] = result['round_completions'][idx]
            cand['messages_records'] = result['messages_records'][idx]
            cand['scores']['xgptscore'] = get_xgptscore_from_json(cand['responses'][-1])
            idx += 1
    with open(output_file, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(output_file))
else:
    logging.info("Loading from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)

# Compute correlation
if human_score_names is not None:
    logging.info("Computing correlation with human score {}".format(human_score_names))
    human_scores = []
    for item in items:
        for cand in item['candidates']:
            for h_name in human_score_names:
                if h_name in cand['scores']:
                    human_scores.append(cand['scores'][h_name])
                    break

    metrics = ["xgptscore"]

    Pearson_corr = {}
    Spearman_corr = {}
    Kendall_corr = {}
    for metric in metrics:
        metric_scores = [[cand['scores'][metric] for cand in item['candidates']] for item in items]
        metric_scores = list(chain(*metric_scores)) # transpose and flatten
        metric_corr = MyCorrelation(1, human_scores, metric_scores)
        Pearson_corr[metric] = metric_corr.Pearson()
        Spearman_corr[metric] = metric_corr.Spearman()
        Kendall_corr[metric] = metric_corr.Kendall()

    # sort Corr
    Pearson_corr = {k: v for k, v in sorted(Pearson_corr.items(), key=lambda item: item[1][0], reverse=True)}
    Spearman_corr = {k: v for k, v in sorted(Spearman_corr.items(), key=lambda item: item[1][0], reverse=True)}
    Kendall_corr = {k: v for k, v in sorted(Kendall_corr.items(), key=lambda item: item[1][0], reverse=True)}
    Corr_record = {
        "Pearson": Pearson_corr,
        "Spearman": Spearman_corr,
        "Kendall": Kendall_corr,
    }
    # Save correlation results
    corr_results_file = Path("./eval_results/") / output_file.relative_to(data_dir).with_suffix(".corr.json")
    corr_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(corr_results_file, "w") as f:
        json.dump(Corr_record, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(corr_results_file))
else:
    logging.info("No human score provided, skip correlation computation")
    


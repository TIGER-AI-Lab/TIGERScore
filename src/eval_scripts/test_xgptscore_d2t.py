"""
Usage:
    python test_xgptscore_d2t.py
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
from collections import Counter
from xgptscore.xgptscore import xgptscore
from itertools import chain
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json_per_aspect
from xgptscore.constants import EVAL_ASPECTS
from bs_utils import reformat_d2t_for_bartscore
logging.basicConfig(level=logging.INFO)

# params
task='data2text'
bart_version="D2T"
dataset="BAGEL"
data_dir="../../BARTScore"
xgptscore_mode="d2t"
version_key=f"{xgptscore_mode}.final_instruction_1"
human_score_name="informativeness"
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
        for cand in item['candidates']:
            xgptitems.append(XPGTItem(
                task=task,
                instruction=item['instruction'],
                input=item['input'],
                ref_output=item['output'],
                hypo_output=cand['text']
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
# Compute correlation
num_cands = len(items[0]['candidates'])
human_scores = [[cand['scores'][human_score_name] for cand in item['candidates']] for item in items]
human_scores = list(chain(*zip(*human_scores))) # transpose and flatten
metrics = [our_score_name,
    'rouge1_p',
    'rouge2_p',
    'rougel_p',
    'bert_score_p',
    'mover_score',
    'bart_score_cnn_hypo_ref',
    # "bart_score_src_hypo",
    # 'bart_score_para_src_hypo',
    'xgptscore_Fluency',
    'xgptscore_Relevance',
    # 'xgptscore_Informativeness',
    'xgptscore_Accuracy',
]

Pearson_corr = {}
Spearman_corr = {}
Kendall_corr = {}
for metric in metrics:
    metric_scores = [[cand['scores'][metric] if metric in cand['scores'] else None for cand in item['candidates']] for item in items]
    metric_scores = list(chain(*zip(*metric_scores))) # transpose and flatten
    metric_corr = MyCorrelation(num_cands, human_scores, metric_scores)
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
corr_results_file = Path("./eval_results/") / f"{bart_version}/{dataset}" / (output_file.stem + ".corr.json")
corr_results_file.parent.mkdir(parents=True, exist_ok=True)
with open(corr_results_file, "w") as f:
    json.dump(Corr_record, f, indent=4, ensure_ascii=False)
    logging.info("Saved to {}".format(corr_results_file))
# save to another location
corr_results_file = output_file.parent / "eval_results" / (output_file.stem + ".corr.json")
corr_results_file.parent.mkdir(parents=True, exist_ok=True)
with open(corr_results_file, "w") as f:
    json.dump(Corr_record, f, indent=4, ensure_ascii=False)
    logging.info("Saved to {}".format(corr_results_file))

# by segment
d2t_eval_path = Path(f"{output_file}")
reformat_d2t_for_bartscore(d2t_eval_path,our_score_name)
from bs_analysis import D2TStat
d2t_stat = D2TStat(str(d2t_eval_path.with_name(d2t_eval_path.stem + "_bs_format.pkl")))
# Save correlation results
corr_results_file = Path("./eval_results/") / f"{bart_version}/{dataset}" / (output_file.stem + ".bs_corr.json")
corr_results_file.parent.mkdir(parents=True, exist_ok=True)
d2t_stat.evaluate_text(human_score_name,metrics,corr_results_file)
# summ_stat.get_fact_acc(metrics)
# save to another location
corr_results_file = output_file.parent / "eval_results" / (output_file.stem + ".bs_corr.json")
corr_results_file.parent.mkdir(parents=True, exist_ok=True)
d2t_stat.evaluate_text(human_score_name,metrics,table=corr_results_file)

# distribution of scores
our_score_list = [[cand['scores'][our_score_name] for cand in item['candidates']] for item in items]
print(f"Our score distribution: {Counter(sorted(list(chain(*our_score_list))))}")
"""
Usage:
    python test_xgptscore_wmt.py
"""
import json
import logging
import sys
import numpy as np
from pathlib import Path
from utils import MyCorrelation
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.xgptscore import xgptscore
from itertools import chain
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json
logging.basicConfig(level=logging.INFO)

# params
task='translation'
wmt_version="wmt22"
lang_pair="en-ru"
data_dir=f"../../data/"
xgptscore_mode="old_ea"
version_key=f"{xgptscore_mode}.random_2_sys_new_wmt_mqm"
human_score_name="mqm"
model_name="ChatGPT"
overwrite=True
max_size=25 # set to None to use all examples
num_sys=2
if isinstance(max_size, int) and max_size > 0:
    version_key = f"{version_key}_{max_size}"

# load data
input_file=Path(f"{data_dir}/{wmt_version}/{lang_pair}/eval_data.json")
if version_key:
    output_file = input_file.with_suffix(f".{version_key}.json")
else:
    output_file = input_file.with_suffix(f".default.json")

if not output_file.exists() or overwrite:
    # Load and shuffle data
    logging.info("Loading from {}".format(input_file))
    with open(input_file, "r") as f:
        items = json.load(f)
    np.random.seed(42)
    np.random.shuffle(items)
    if isinstance(max_size, int) and max_size > 0:
        items = items[:max_size]
    
    # randomly select 2 systems
    if isinstance(num_sys, int) and num_sys > 0:
        """Randomly select num_sys systems for each example"""
        # all_sys_names = set([x['model'] for x in items[0]['candidates']])
        # shuffled_sys_names = list(all_sys_names)
        # np.random.seed(42)
        # np.random.shuffle(shuffled_sys_names)
        # sub_sampled_sys = shuffled_sys_names[:num_sys]
        # for item in items:
        #     sys_cand_map = {x['model']: x for x in item['candidates']}
        #     item['candidates'] = [sys_cand_map[x] for x in sub_sampled_sys]
        """Randomly select num_sys candidates for each example"""
        for item in items:
            idxs = np.random.permutation(len(item['candidates']))[:num_sys]
            item['candidates'] = [item['candidates'][idx] for idx in idxs]
    
    # Data processing
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

    # Run xgptscore
    result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name)
    idx = 0
    for item in items:
        for cand in item['candidates']:
            cand['responses'] = result['round_completions'][idx]
            cand['messages_records'] = result['messages_records'][idx]
            cand['scores']['xgptscore'] = get_xgptscore_from_json(cand['responses'][-1])
            idx += 1
        
    # Save results
    with open(output_file, "w") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(output_file))
else:
    logging.info("Loading existing results from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)

# Compute correlation
num_cands = len(items[0]['candidates'])
human_scores = [[cand['scores'][human_score_name] for cand in item['candidates']] for item in items]
human_scores = list(chain(*zip(*human_scores))) # transpose and flatten
metrics = ["xgptscore", "BLEU-refA", "BERTScore-refA", "COMET-22-refA", "COMET-20-refA", "BLEURT-20-refA", "chrF-refA"]

Pearson_corr = {}
Spearman_corr = {}
Kendall_corr = {}
for metric in metrics:
    metric_scores = [[cand['scores'][metric] for cand in item['candidates']] for item in items]
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
corr_results_file = Path("./eval_results/") / f"{wmt_version}/{lang_pair}" / (output_file.stem + ".corr.json")
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
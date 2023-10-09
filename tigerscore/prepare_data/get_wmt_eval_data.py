"""
Get WMT train data
"""
import json
import argparse
import logging
import time
import numpy as np
from typing import List
from pathlib import Path
from datasets import load_dataset
from pathlib import Path
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer
from mt_metrics_eval.data import EvalSet


logging.basicConfig(level=logging.INFO)
# log current time
logging.info("Running time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


lang_map = {
    'zh': 'Chinese',
    'en': 'English',
    'de': 'German',
    'ru': 'Russian',
    'cs': 'Czech',
    'uk': 'Ukrainian',
    'hr': 'Croatian',
    'ja': 'Japanese',
    'liv': 'Livonian',
}

def get_eval_data(wmt_version, lang_pair, human_score_name):
    src_lang, tgt_lang = lang_pair.split("-")
    logging.info("Loading data from mt_metrics_eval EvalSet for \
        wmt_version={}, lang_pair={}, human_score_name={}".format(wmt_version, lang_pair, human_score_name))
    try:
        data = EvalSet(wmt_version, lang_pair, read_stored_metric_scores=True)
    except FileNotFoundError:
        logging.error("No stored metric scores found for wmt_version={}, lang_pair={}".format(wmt_version, lang_pair))
        return None
    human_scores_names = set(x for x in data.human_score_names if x.startswith(human_score_name))
    load_metrics = ['chrF', 'COMET', "BLEU", "BERTScore", "BLEURT"]
    metrics_names = set(x for x in data.metric_names if any(x.startswith(y) for y in load_metrics))
    valid_sys_names = data.sys_names - data.human_sys_names
    ref_names = data.ref_names
    print("Valid systems: {}".format(valid_sys_names))
    for sys_name in data.sys_names - data.human_sys_names:
        a = [len([x for x in data._scores['seg'][k][sys_name] if x is not None]) for k in human_scores_names.union(metrics_names) if data._scores['seg'][k]]
        for i, metric in enumerate(human_scores_names.union(metrics_names)):
            if a[i] == 0:
                print("Metric {} is missing for sys {}".format(metric, sys_name))
                try:
                    valid_sys_names.remove(sys_name)
                except KeyError:
                    logging.info("Sys {} is already removed".format(sys_name))

    formated_data = []
    for i in range(len(data.src)):
        if any([data._scores['seg'][human_score_name][sys_name][i] is None for sys_name in valid_sys_names]):
            # skip examples without human scores
            continue
        domain = None
        for d in data.domains:
            for start, end in data.domains[d]:
                if start <= i < end:
                    domain = d
                    break
        assert domain is not None
        formated_data.append({
            "id": f"{wmt_version}_{lang_pair}_eval_{i}",
            "instruction": f"Translate the following text from {lang_map[src_lang]} to {lang_map[tgt_lang]}.",
            "input": data.src[i],
            "refs": [data.sys_outputs[ref_name][i] for ref_name in ref_names],
            "data_source": f"{wmt_version}_{lang_pair}_{domain}",
            "task": "translation",
            "candidates": [
                {
                    "text": data.sys_outputs[sys_name][i],
                    "model": sys_name,
                    "decoding_method": "greedy",
                    "scores": {
                        k: data._scores['seg'][k][sys_name][i]
                        for k in human_scores_names.union(metrics_names) if data._scores['seg'][k]
                    },
                }
                for sys_name in valid_sys_names
            ]
        })
    logging.info(f"Eval {lang_pair} data statistics:")
    logging.info("# Examples: {}".format(len(formated_data)))
    logging.info("# Avg. Unique outputs: {}".format(sum([len(x['candidates']) for x in formated_data]) / len(formated_data)))
    logging.info("# Unique src: {}".format(len(set([x['input'] for x in formated_data]))))
    logging.info("Domain distribution:")
    domain_counter = Counter([x['data_source'].split("_")[-1] for x in formated_data])
    for domain in domain_counter:
        logging.info("  {}: {}".format(domain, domain_counter[domain]))
    logging.info("Year distribution:")
    year_counter = Counter([x['data_source'].split("_")[0] for x in formated_data])
    for year in year_counter:
        logging.info("  {}: {}".format(year, year_counter[year]))
    return formated_data
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--wmt_version', type=str, default='wmt22')
    parser.add_argument('--lang_pair', type=str, default='zh-en')
    parser.add_argument('--human_score_name', type=str, default='mqm', choices=['mqm', 'wmt-z', 'wmt-appraise-z'])
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--max_ex_size', type=int, default=3500)
    args = parser.parse_args()
    
    lang_pair = args.lang_pair
    
    # Load the eval data
    eval_data = get_eval_data(args.wmt_version, args.lang_pair, args.human_score_name)
    if eval_data is None:
        logging.error("No eval data found for wmt_version={}, lang_pair={}, human_score_name={}".format(args.wmt_version, args.lang_pair, args.human_score_name))
    else:
        logging.info("Loaded {} eval examples".format(len(eval_data)))

    # Save the data
    args.data_dir = Path(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    eval_file = args.data_dir / f"{args.wmt_version}/{args.lang_pair}/eval_data.json"
    eval_file.parent.mkdir(parents=True, exist_ok=True)

    if eval_data:
        logging.info("Eval data statistics:")
        logging.info("# Examples: {}".format(len(eval_data)))
        logging.info("# Avg. Unique outputs: {}".format(sum([len(x['candidates']) for x in eval_data]) / len(eval_data)))
        logging.info("# Unique src: {}".format(len(set([x['input'] for x in eval_data]))))
        with open(eval_file, "w") as f:
            json.dump(eval_data, f, indent=4, ensure_ascii=False)
            logging.info("Saved eval data to {}".format(eval_file))
        


    

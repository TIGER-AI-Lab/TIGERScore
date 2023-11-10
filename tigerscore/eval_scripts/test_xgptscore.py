"""
This script is used to test xgptscore for prompt engineering.
"""

from common import str2bool
from xgptscore.xgptscore import xgptscore
from itertools import chain
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json
import json
import logging
import sys
import numpy as np
import fire
from pathlib import Path
from utils import MyCorrelation
sys.path.append(str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)


def main(input_file: str, task: str, model_name: str, output_file: str, xgptscore_mode: str = "prompt", max_size: int = None, overwrite: str = "false"):
    overwrite = str2bool(overwrite)
    if output_file is None:
        output_file = Path(input_file).parent / \
            (Path(input_file).stem + "." + xgptscore_mode + ".json")
    if not output_file.exists() or overwrite:
        logging.info("Loading from {}".format(input_file))
        with open(input_file, "r") as f:
            items = json.load(f)
        np.random.seed(42)
        np.random.shuffle(items)
        if isinstance(max_size, int) and max_size > 0:
            items = items[:max_size]

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
        result = xgptscore(xgptitems, mode=xgptscore_mode,
                           model_name=model_name, num_workers=5)
        idx = 0
        for item in items:
            for cand in item['candidates']:
                cand['responses'] = result['round_completions'][idx]
                cand['messages_records'] = result['messages_records'][idx]
                cand['scores']['xgptscore'] = get_xgptscore_from_json(
                    cand['responses'][-1])
                idx += 1

        # Save results
        with open(output_file, "w") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            logging.info("Saved to {}".format(output_file))
    else:
        logging.info("Loading existing results from {}".format(output_file))
    with open(output_file, "r") as f:
        items = json.load(f)

    # evaluate system

    num_cands = len(items[0]['candidates'])
    human_scores = [[cand['scores']["rank"]
                     for cand in item['candidates']] for item in items]
    human_scores = list(chain(*zip(*human_scores)))  # transpose and flatten
    metrics = ["xgptscore", "bleu", "rouge1", "rouge2",
               "rougeL", "rougeLsum", "bart_score", "bart_score_cnn"]
    # metrics = ["xgptscore"]

    Pearson_corr = {}
    Spearman_corr = {}
    Kendall_corr = {}
    for metric in metrics:
        metric_scores = [[cand['scores'][metric]
                          for cand in item['candidates']] for item in items]
        metric_scores = list(chain(*zip(*metric_scores))
                             )  # transpose and flatten
        metric_corr = MyCorrelation(num_cands, human_scores, metric_scores)
        Pearson_corr[metric] = metric_corr.Pearson()
        Spearman_corr[metric] = metric_corr.Spearman()
        Kendall_corr[metric] = metric_corr.Kendall()

    # sort Corr
    Pearson_corr = {k: v for k, v in sorted(
        Pearson_corr.items(), key=lambda item: item[1][0], reverse=True)}
    Spearman_corr = {k: v for k, v in sorted(
        Spearman_corr.items(), key=lambda item: item[1][0], reverse=True)}
    Kendall_corr = {k: v for k, v in sorted(
        Kendall_corr.items(), key=lambda item: item[1][0], reverse=True)}
    Corr_record = {
        "Pearson": Pearson_corr,
        "Spearman": Spearman_corr,
        "Kendall": Kendall_corr,
    }
    # Save correlation results
    corr_results_file = Path("./eval_results/") / \
        (output_file.stem + ".corr.json")
    corr_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(corr_results_file, "w") as f:
        json.dump(Corr_record, f, indent=4, ensure_ascii=False)
    logging.info("Saved to {}".format(corr_results_file))
    # save to another location
    corr_results_file = output_file.parent / \
        "eval_results" / (output_file.stem + ".corr.json")
    corr_results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(corr_results_file, "w") as f:
        json.dump(Corr_record, f, indent=4, ensure_ascii=False)
    logging.info("Saved to {}".format(corr_results_file))
    # print("Correlation results:")
    # print(json.dumps(Corr_record, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    fire.Fire(main)

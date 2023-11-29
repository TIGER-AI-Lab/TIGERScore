import json
import random
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from tigerscore.xgptscore.xgptscore import xgptscore
from tigerscore.xgptscore.process_utils import XPGTItem
logging.basicConfig(level=logging.warning)
import fire
import re
from mt_metrics_eval.stats import Correlation
from typing import List

class MyCorrelation(Correlation):
    def __init__(self, num_sys: int, gold_scores: List[int], metric_scores: List[int]):
        # remove nan in metrics scores
        none_metric_scores_idxs = [idx for idx,
                                   x in enumerate(metric_scores) if x is None]
        print("Remove {} nan scores from {} scores".format(
            len(none_metric_scores_idxs),
            len(metric_scores)
        ))
        gold_scores = gold_scores.copy()
        # set gold scores to None if metric scores are None
        for idx in none_metric_scores_idxs[::-1]:
            gold_scores[idx] = None
        super().__init__(num_sys, gold_scores, metric_scores)


def find_first_float(s):
    match = re.search(r"[-+]?\d*\.\d+|\d+", s)
    return float(match.group()) if match else None

def main(
    task: str,
    data_path: str,
    output_file: str = None,
    xgptscore_mode: str = "zero_shot_baseline",
    model_name: str = "gpt-4",
    human_score_names: str = "human_score",
    overwrite: bool = False,
    max_size: int = None,
    seed: int = 42,
    shuffle_file: bool = False,
    source_max_length: int = None,
    ref_max_length: int = None,
    hypo_max_length: int = None,
    dataset_split: str = "test",
):
    logging.warning("Params: \n{}".format(json.dumps(locals(), indent=4)))
    # load data
    data_path = Path(data_path)
    input_file = data_path
    
    input_file=Path(input_file)
    if not output_file:
        output_file = Path(str(data_path.parent/data_path.stem) + "_0shot_{}.json".format(model_name))
        if not output_file.parent.parent.exists():
            output_file.parent.parent.mkdir(parents=True)
        if not output_file.parent.exists():
            output_file.parent.mkdir()
    else:
        output_file = Path(output_file)
    with open(input_file, "r") as f:
        items = json.load(f)
        logging.warning("Loaded {} items from {}".format(len(items), input_file))
    logging.warning("Preparing writing to {}...".format(output_file))
    
    random.seed(seed); logging.warning("Set seed to {}".format(seed))
    if shuffle_file:
        random.shuffle(items)
        logging.warning("Shuffled {} items".format(len(items)))
    if isinstance(max_size, int) and max_size > 0:
        items = items[:max_size]
        logging.warning("Truncated to {} items".format(len(items)))

    xgptitems = []
    for item in items:
        for cand in item['candidates']:
            xgptitems.append(XPGTItem(
                task=task,
                instruction=item['instruction'],
                input=item['input'],
                hypo_output=cand['text'],
                ref_output="a"
            ))
    output_file = Path(output_file)
    if not output_file.exists() or overwrite:
        logging.warning("Running xgptscore")
        # run xgptscore
        result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name,num_workers=5)
        idx = 0
        for item in items:
            for cand in item['candidates']:      
                cand['responses'] = result['round_completions'][idx]
                cand['messages_records'] = result['messages_records'][idx]
                xgptscore_ans = find_first_float(cand['responses'][-1])
                if xgptscore_ans is None:
                    logging.info(f"XGPTScore failed for {cand['text']}")
                    cand['scores']["{}_xgptscore".format(model_name)] = None
                else:
                    cand['scores']["{}_xgptscore".format(model_name)] = xgptscore_ans
                idx += 1
        with open(output_file, "w") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            logging.warning("Saved to {}".format(output_file))
    else:
        logging.warning("Loading from {}".format(output_file))
        with open(output_file, "r") as f:
            items = json.load(f)
    
    if isinstance(human_score_names, tuple):
        human_score_names = list(human_score_names)
    else:
        human_score_names = [human_score_names]


    for h_name in human_score_names:
        human_scores = []
        xgptscores = []
        for item in items:
            for cand in item['candidates']:
                for s_name, score in cand['scores'].items():
                    if s_name == h_name:
                        if "gpt-4_xgptscore" not in cand['scores']:
                            logging.info(f"XGPTScore failed for {cand['text']}")
                            cand['scores']["gpt-4_xgptscore"] = None
                        elif cand['scores']["gpt-4_xgptscore"] is not None:
                            if cand['scores']["gpt-4_xgptscore"] < 0 or cand['scores']["gpt-4_xgptscore"] > 10:
                                cand['scores']["gpt-4_xgptscore"] = None
                        xgptscores.append(cand['scores']["gpt-4_xgptscore".format(model_name)] )
                        human_scores.append(score)
                        
                        break
        assert len(human_scores) == len(xgptscores)
        print (len(human_scores))
        corr = MyCorrelation(1, human_scores, xgptscores)
        print("Human score: {}".format(h_name))
        print("Pearson correlation: {}".format(corr.Pearson()))
        print("Spearman correlation: {}".format(corr.Spearman()))
        print("Kendall correlation: {}".format(corr.Kendall()))


if __name__ == "__main__":
    fire.Fire(main)
"""
Usage:
    python test_xgptscore_wmt.py
"""
import json
import random
import logging
import sys
import fire
from pathlib import Path
from utils import MyCorrelation
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.xgptscore import xgptscore
from itertools import chain
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json
logging.basicConfig(level=logging.warning)

# # params
# task='translation'
# xgptscore_mode="wmt_mqm"
# version_key=f"{xgptscore_mode}.distill_new_wmt_mqm"
# human_score_names=["mqm", "da"]
# model_name="ChatGPT"
# overwrite=True
# max_size=None # set to None to use all examples
# if isinstance(max_size, int) and max_size > 0:
#     version_key = f"{version_key}_{max_size}"
# # load data
#     split="train"
#     data_dir="../../data"
#     input_file=Path(f"{data_dir}/wmt/ru-en/{split}_data.json")

def main(
    task: str,
    xgptscore_mode: str,
    model_name: str,
    input_file: str,
    version_key: str = None,
    overwrite: bool = False,
    max_size: int = None,
    seed: int = 42,
    shuffle: bool = False,
    source_max_length: int = None,
    ref_max_length: int = None,
    hypo_max_length: int = None,
):
    
    logging.warning("Loading from {}".format(input_file))
    with open(input_file, "r") as f:
        items = json.load(f)
    if shuffle:
        random.seed(seed)
        random.shuffle(items)
    suffix = f".{xgptscore_mode}.{model_name}"
    if version_key:
        suffix += f".{version_key}"
    if isinstance(max_size, int) and max_size > 0:
        items = items[:max_size]
        suffix += f".{max_size}"
    output_file = Path(input_file).with_suffix(f"{suffix}.json")

    xgptitems = []
    for item in items:
        for cand in item['candidates']:
            xgptitems.append(XPGTItem(
                task=task,
                instruction=item['instruction'],
                input=item['input'],
                ref_output=item['refs'] if 'refs' in item else item['output'],
                hypo_output=cand['text']
            ))

    if not output_file.exists() or overwrite:
        logging.warning("Running xgptscore")
        # run xgptscore
        xgptscore_params = {
            "max_lengths": {
                "input": source_max_length,
                "hypo_output": hypo_max_length,
                "ref_output": ref_max_length,
            },
        }
        result = xgptscore(xgptitems, mode=xgptscore_mode, model_name=model_name, **xgptscore_params)
        idx = 0
        for item in items:
            for cand in item['candidates']:
                cand['responses'] = result['round_completions'][idx]
                cand['messages_records'] = result['messages_records'][idx]
                if 'scores' not in cand:
                    cand['scores'] = {}
                try:
                    cand['scores']['xgptscore'] = get_xgptscore_from_json(cand['responses'][-1])
                except:
                    pass
                idx += 1
        with open(output_file, "w") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            logging.warning("Saved to {}".format(output_file))
    else:
        logging.warning("Found existing {}".format(output_file))
        logging.warning("Skipping xgptscore")

if __name__ == "__main__":
    logging.basicConfig(level=logging.warning)
    fire.Fire(main)
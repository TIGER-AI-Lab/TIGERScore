"""

"""
import json
import random
import logging
import sys
import fire
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.process_utils import XPGTItem, get_xgptscore_from_json_per_aspect
from xgptscore.xgptscore import xgptscore
from xgptscore.constants import EVAL_ASPECTS
logging.basicConfig(level=logging.warning)


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
        result = xgptscore(xgptitems, mode=xgptscore_mode,
                           model_name=model_name, num_workers=5)
        idx = 0
        aspects = EVAL_ASPECTS[task].keys()
        score_dict = {"xgptscore_" + aspect: 0 for aspect in aspects}
        for item in items:
            for cand in item['candidates']:
                cand['responses'] = result['round_completions'][idx]
                cand['messages_records'] = result['messages_records'][idx]
                xgptscore_ans = get_xgptscore_from_json_per_aspect(
                    cand['responses'][-1])
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
        logging.warning("Found existing {}".format(output_file))
        logging.warning("Skipping xgptscore")


if __name__ == "__main__":
    logging.basicConfig(level=logging.warning)
    fire.Fire(main)

"""
Usage:
    python test_xgptscore_wmt.py
"""
import json
import random
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.xgptscore import xgptscore
from itertools import chain
from xgptscore.process_utils import XPGTItem
logging.basicConfig(level=logging.warning)

def main(
    task: str,
    data_path: str,
    dataset: str,
    output_file: str = None,
    xgptscore_mode: str = "instruction",
    model_name: str = "ChatGPT",
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
    input_file = data_path / dataset / (dataset_split + "_data.json")
    
    input_file=Path(input_file)
    if not output_file:
        output_file = data_path / dataset / "candidates" / dataset_split / "top_p_sampling" / f"{model_name}.json"
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
        xgptitems.append(XPGTItem(
            task=task,
            instruction=item['instruction'],
            input=item['input'],
            ref_output=item['output'] if "output" in item else item['refs'],
            hypo_output=None,
        ))
        if "candidates" in item:
            del item["candidates"]

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
        for i, item in enumerate(items):
            item['responses'] = result['round_completions'][i]
            item['messages_records'] = result['messages_records'][i]
            item['candidates'] = [
                {"text":result['round_completions'][i][0],
                               "scores":{}
            }]
        # print(items)
        with open(output_file, "w") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            logging.warning("Saved to {}".format(output_file))
    else:
        logging.warning("Loading from {}".format(output_file))
        with open(output_file, "r") as f:
            items = json.load(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()
    main(**vars(args))

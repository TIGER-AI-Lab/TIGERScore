"""
Usage:
    Gererate candidates by GPT-3.5 or GPT-4.
"""

import json
import random
import logging
import sys
import fire
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from xgptscore.process_utils import XPGTItem
from xgptscore.xgptscore import xgptscore
logging.basicConfig(level=logging.warning)


def main(
    task: str,
    data_path: str,
    xgptscore_mode: str = "instruction",
    model_name: str = "gpt-4",
    overwrite: bool = False,
    max_size: int = None,
    seed: int = 42,
    shuffle_file: bool = False,
    source_max_length: int = None,
    ref_max_length: int = None,
    hypo_max_length: int = None,
    dataset_split: str = "test",
):
    """Gererate candidates by GPT-3.5 or GPT-4.

    Args:
        task (str): Task name.
        data_path (str): Path to the data.
        dataset (str): Dataset name.
        output_file (str, optional):  Defaults to None.
        xgptscore_mode (str, optional):  Defaults to "instruction".
        model_name (str, optional):  Defaults to "ChatGPT".
        overwrite (bool, optional):  Defaults to False.
        max_size (int, optional):  Defaults to None.
        seed (int, optional):  Defaults to 42.
        shuffle_file (bool, optional):  Defaults to False.
        source_max_length (int, optional):  Defaults to None.
        ref_max_length (int, optional):  Defaults to None.
        hypo_max_length (int, optional):  Defaults to None.
        dataset_split (str, optional):  Defaults to "test".
    """
    logging.warning("Params: \n{}".format(json.dumps(locals(), indent=4)))
    # load data
    data_path = Path(data_path)
    input_file = data_path

    input_file = Path(input_file)
    output_file = input_file
    with open(input_file, "r") as f:
        items = json.load(f)
        logging.warning("Loaded {} items from {}".format(
            len(items), input_file))
    logging.warning("Preparing writing to {}...".format(output_file))

    random.seed(seed)
    logging.warning("Set seed to {}".format(seed))
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
        result = xgptscore(xgptitems, mode=xgptscore_mode,
                           model_name=model_name,num_workers=5, **xgptscore_params)
        for i, item in enumerate(items):
            item['responses'] = result['round_completions'][i]
            item['messages_records'] = result['messages_records'][i]
            if item["output"] is not None:
                item["output"] = result['round_completions'][i][0]
        # print(items)
        with open(output_file, "w") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            logging.warning("Saved to {}".format(output_file))
    else:
        logging.warning("Loading from {}".format(output_file))
        with open(output_file, "r") as f:
            items = json.load(f)


if __name__ == "__main__":
    fire.Fire(main)

"""
Generate synthesis distillation data from a json file.
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
    input_file: str,
    output_file: str = None,
    xgptscore_mode: str = "kb_txt",
    model_name: str = "gpt-4",
    version_key: str = "default",
    overwrite: bool = False,
    max_size: int = None,
    seed: int = 42,
    shuffle_file: bool = False,
    source_max_length: int = None,
    ref_max_length: int = None,
    hypo_max_length: int = None,
):
    logging.warning("Params: \n{}".format(json.dumps(locals(), indent=4)))
    # params
    if isinstance(max_size, int) and max_size > 0:
        version_key = f"{version_key}_{max_size}"
    # load data
    input_file = Path(input_file)
    if not output_file:
        output_file = input_file.with_suffix(
            f".{xgptscore_mode}.{version_key}.json")
    else:
        output_file = Path(output_file)
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
    elif isinstance(max_size, float) and max_size > 0 and max_size < 1:
        items = random.sample(items, int(len(items) * max_size))
        logging.warning("Sampled to {} items".format(len(items)))

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
        result = xgptscore(xgptitems, mode=xgptscore_mode,
                           model_name=model_name, **xgptscore_params)
        for i, item in enumerate(items):
            item['responses'] = result['round_completions'][i]
            item['messages_records'] = result['messages_records'][i]
        with open(output_file, "w") as f:
            json.dump(items, f, indent=4, ensure_ascii=False)
            logging.warning("Saved to {}".format(output_file))
    else:
        logging.warning("Loading from {}".format(output_file))
        with open(output_file, "r") as f:
            items = json.load(f)


if __name__ == "__main__":
    fire.Fire(main)

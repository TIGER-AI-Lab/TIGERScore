import json
import sys
import fire
import random
import logging
from pathlib import Path
from typing import List, Union
from transformers import AutoTokenizer
from tqdm import tqdm
sys.path.append("..")
from common.datasets_config import DATASETS_CONFIG
logging.basicConfig(level=logging.INFO)

def main(
    task: str = "long-form QA",
    data_dir: str = "../../data/",
    shuffle: bool = True,
    seed: int = 42,
    max_size_per_ds: int = 1000,
    max_input_length: int = None,
    max_output_length: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    task_datasets_config = DATASETS_CONFIG[task]
    final_ds = []
    for dataset_name, dataset_config in task_datasets_config.items():
        ds_train_file = Path(data_dir) / dataset_name.replace(":", "/") / "train_data.json"
        with open(ds_train_file, "r") as f:
            ds_train = json.load(f)
        if shuffle:
            random.seed(seed)
            random.shuffle(ds_train)
        if isinstance(max_input_length, int) and max_input_length > 0:
            inputs = [item['instruction'] + "\n" + item["input"] for item in ds_train]
            inputs_ids = [tokenizer.encode(x, add_special_tokens=False) 
                for x in tqdm(inputs, desc=f"Tokenizing {dataset_name} inputs")]
            _ds_train = [x for x, _ids in zip(ds_train, inputs_ids) if len(_ids) <= max_input_length]
            logging.info("Removed {} examples with inst+input length > {}".format(
                len(ds_train) - len(_ds_train), max_input_length))
            ds_train = _ds_train
        if isinstance(max_output_length, int) and max_output_length > 0:
            outputs = [item['output'] for item in ds_train]
            outputs_ids = [tokenizer.encode(x, add_special_tokens=False) 
                for x in tqdm(outputs, desc=f"Tokenizing {dataset_name} outputs")]
            _ds_train = [x for x, _ids in zip(ds_train, outputs_ids) if len(_ids) <= max_output_length]
            logging.info("Removed {} examples with output length > {}".format(
                len(ds_train) - len(_ds_train), max_output_length))
        if isinstance(max_size_per_ds, int) and max_size_per_ds > 0:
            ds_train = ds_train[:max_size_per_ds]
        for item in ds_train:
            item["data_source"] = dataset_name
            item["task"] = task
        final_ds.extend(ds_train)
    logging.info("Final dataset size: {}".format(len(final_ds)))
    task_ds_train_file = Path(data_dir) / "synthesis_debug" / task / "train_data.json"
    task_ds_train_file.parent.mkdir(parents=True, exist_ok=True)
    with open(task_ds_train_file, "w") as f:
        json.dump(final_ds, f, indent=4, ensure_ascii=False)
        logging.info("Saved to {}".format(task_ds_train_file))
    
    
if __name__ == "__main__":
    fire.Fire(main)
    
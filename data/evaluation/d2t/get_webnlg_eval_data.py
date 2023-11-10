"""
Get WebNLG2017 data
"""

"""
Get WMT train data
"""

import json
import argparse
import time
import logging
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter
logging.basicConfig(level=logging.INFO)
# log current time
logging.info("\nRunning time: {}".format(
    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))


def format_eval_data(dataset, data_version):
    if data_version == 2017:
        return format_eval_data_2017(dataset)
    elif data_version == 2020:
        return format_eval_data_2020(dataset)
    else:
        logging.error("Unknown data version: {}".format(data_version))
        return None


def format_eval_data_2017(dataset):
    dataset = sorted(dataset, key=lambda x: x['X_unit_id'])
    # merge data with the same id and get mean for human score
    merged_dataset = []

    def merge_items(_now_items: list) -> dict:
        res = _now_items[0]
        scores = {
            "fluency": np.mean([item['fluency'] for item in _now_items]),
            "grammaticality": np.mean([item['grammaticality'] for item in _now_items]),
            "semantic_adequacy": np.mean([item['semantic_adequacy'] for item in _now_items]),
        }
        res.update(scores)
        return res

    now_items = []
    now_id = dataset[0]['X_unit_id']
    for item in dataset:
        x_unit_id = item['X_unit_id']
        if x_unit_id != now_id:
            # merge now_items
            now_items = merge_items(now_items)
            merged_dataset.append(now_items)
            now_items = []
            now_id = x_unit_id
        now_items.append(item)
    # merge last items
    now_items = merge_items(now_items)
    merged_dataset.append(now_items)

    dataset = merged_dataset

    formated_data = {}
    src_id = 0
    for item in dataset:
        src = item['mr'].replace("|", ",").replace("<br>", "\n").strip()
        src = "\n".join(
            ["( " + x.strip() + " )" for x in src.split("\n") if x.strip() != ""])
        if src not in formated_data:
            formated_data[src] = {
                "id": f"webnlg2017_human_eval_{src_id}",
                "instruction": "Generate a description for the following triples.",
                "input": src,
                "output": [],
                "data_source": "webnlg2017",
                "task": "data2text",
                "candidates": [],
            }
            src_id += 1
        # add candidate and avoid duplicate candidates
        if any([cand['text'] == item['text'] for cand in formated_data[src]['candidates']]):
            continue
        if item['team'] == 'webnlg':
            formated_data[src]['output'].append(item['text'])
            continue
        formated_data[src]['candidates'].append({
            "text": item['text'],
            "model": item['team'],
            "decoding_method": "greedy",
            "domain": item['category'],
            "type": item['type'],
            "scores": {
                "fluency": item['fluency'],
                "grammaticality": item['grammaticality'],
                "semantic_adequacy": item['semantic_adequacy'],
            }
        })
    train_data = list(formated_data.values())
    logging.info("Train data statistics:")
    logging.info("# Examples: {}".format(len(train_data)))
    logging.info("# Avg. Unique outputs: {}".format(
        sum([len(x['candidates']) for x in train_data]) / len(train_data)))
    logging.info("# Unique src: {}".format(
        len(set([x['input'] for x in train_data]))))
    logging.info("Domain distribution:")
    domain_counter = Counter(
        [x['data_source'].split("_")[-1] for x in train_data])
    for domain in domain_counter:
        logging.info("  {}: {}".format(domain, domain_counter[domain]))
    logging.info("Type distribution:")
    type_counter = Counter([c['type']
                           for x in train_data for c in x['candidates']])
    for type in type_counter:
        logging.info("  {}: {}".format(type, type_counter[type]))
    return train_data


def format_eval_data_2020(dataset):

    def merge_items(now_items: list) -> dict:
        res = now_items[0]
        scores = {
            "fluency": np.mean([item['fluency'] for item in now_items]),
            "grammaticality": np.mean([item['grammaticality'] for item in now_items]),
            "semantic_adequacy": np.mean([item['semantic_adequacy'] for item in now_items]),
        }
        res.update(scores)
        return res

    dataset = sorted(dataset, key=lambda x: x['X_unit_id'])
    # merge data with the same id and get mean for human score
    merged_dataset = []

    now_id = dataset[0]['X_unit_id']
    now_items = []
    for item in dataset:
        x_unit_id = item['X_unit_id']
        if x_unit_id != now_id:
            # merge now_items
            now_items = merge_items(now_items)
            merged_dataset.append(now_items)
            now_items = []
            now_id = x_unit_id

    dataset = merged_dataset
    formated_data = {}
    src_id = 0
    for item in dataset:
        src = item['mr'].replace("|", ",").replace("<br>", "\n").strip()
        if src not in formated_data:
            formated_data[src] = {
                "id": f"webnlg2017_human_eval_{src_id}",
                "instruction": "Generate a description for the following triples.",
                "input": src,
                "output": [],
                "data_source": "webnlg2017",
                "task": "data2text",
                "candidates": [],
            }
            src_id += 1
        # add candidate and avoid duplicate candidates
        if any([cand['text'] == item['text'] for cand in formated_data[src]['candidates']]):
            continue
        if item['team'] == 'webnlg':
            formated_data[src]['output'].append(item['text'])
            continue
        formated_data[src]['candidates'].append({
            "text": item['text'],
            "model": item['team'],
            "decoding_method": "greedy",
            "domain": item['category'],
            "type": item['type'],
            "scores": {
                "fluency": item['fluency'],
                "grammaticality": item['grammaticality'],
                "semantic_adequacy": item['semantic_adequacy'],
            }
        })
    train_data = list(formated_data.values())
    logging.info("Train data statistics:")
    logging.info("# Examples: {}".format(len(train_data)))
    logging.info("# Avg. Unique outputs: {}".format(
        sum([len(x['candidates']) for x in train_data]) / len(train_data)))
    logging.info("# Unique src: {}".format(
        len(set([x['input'] for x in train_data]))))
    logging.info("Domain distribution:")
    domain_counter = Counter(
        [x['data_source'].split("_")[-1] for x in train_data])
    for domain in domain_counter:
        logging.info("  {}: {}".format(domain, domain_counter[domain]))
    logging.info("Type distribution:")
    type_counter = Counter([x['type'] for x in train_data])
    for type in type_counter:
        logging.info("  {}: {}".format(type, type_counter[type]))
    return train_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--max_ex_size', type=int, default=1000)
    parser.add_argument('--overwrite', type=str2bool, default=True)
    parser.add_argument('--data_version', type=int,
                        default=2017, choices=[2017, 2020])
    args = parser.parse_args()

    dataset_name = f"teven/webnlg_{args.data_version}_human_eval"

    # Load the training data
    logging.info("Loading data from {}".format(dataset_name))
    dataset = load_dataset(dataset_name, split='train')

    if len(dataset) == 0:
        logging.error("error")
        exit(0)
    else:
        logging.info("Loaded {} examples for {}".format(
            len(dataset), dataset_name))
    # Save the data
    args.data_dir = Path(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.data_dir / f"webnlg_{args.data_version}/train_data.json"
    train_file.parent.mkdir(parents=True, exist_ok=True)
    if not train_file.exists() or args.overwrite:
        train_data = format_eval_data(dataset, args.data_version)
        with open(train_file, "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
            logging.info("Saved training data to {}".format(train_file))

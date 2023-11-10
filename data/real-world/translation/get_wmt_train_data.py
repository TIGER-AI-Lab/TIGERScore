"""
Get WMT train data
"""
import json
import argparse
import time
import logging
import numpy as np
from typing import List
from pathlib import Path
from datasets import load_dataset
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO)
# log current time
logging.info("\nRunning time: {}".format(
    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

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
    'fi': 'Finnish',
    'fr': 'French',
}


def format_train_data(dataset, lang_pair, human_score_name):
    src_lang, tgt_lang = lang_pair.split("-")
    formated_data = {}
    src_id = 0
    for item in dataset:
        year = item['year'] % 100
        src = item['src']
        if src not in formated_data:
            formated_data[src] = {
                "id": f"wmt{year}_{lang_pair}_train_{src_id}",
                "instruction": f"Translate the following text from {lang_map[src_lang]} to {lang_map[tgt_lang]}.",
                "input": item['src'],
                "refs": [item['ref']],
                "data_source": f"wmt{year}_{lang_pair}_{item['domain']}",
                "task": "translation",
                "candidates": []
            }
            src_id += 1
        # add candidate and avoid duplicate candidates
        if any([cand['text'] == item['mt'] for cand in formated_data[src]['candidates']]):
            continue
        formated_data[src]['candidates'].append({
            "text": item['mt'],
            "model": item['system'] if "system" in item else "unknown",
            "decoding_method": "greedy",
            "domain": item['domain'],
            "scores": {
                human_score_name: item['score']
            }
        })
        if item['ref'] not in formated_data[src]['refs']:
            formated_data[src]['refs'].append(item['ref'])
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
    logging.info("Year distribution:")
    year_counter = Counter([x['data_source'].split("_")[0]
                           for x in train_data])
    for year in year_counter:
        logging.info("  {}: {}".format(year, year_counter[year]))
    return train_data


def down_sample_train_data(train_data: List[dict], max_ex_size: int):
    """
    Down sample the formatted training data to have at most max_uni_outputs unique outputs
    Args:
        train_data: formatted training data
        max_ex_size: max total number of candidate outputs
    """
    # randomly downsample
    total_num_cands = sum([len(x['candidates']) for x in train_data])
    np.random.seed(42)
    cands_idxs = np.arange(total_num_cands)
    np.random.shuffle(cands_idxs)
    sampled_cands_idxs = cands_idxs[:max_ex_size]
    sampled_cands_idxs.sort()
    sampled_train_data = []
    cur_cand_idx = 0
    for i, ex in enumerate(train_data):
        sampled_ex = {
            "id": ex['id'],
            "instruction": ex['instruction'],
            "input": ex['input'],
            "refs": ex['refs'],
            "data_source": ex['data_source'],
            "task": ex['task'],
            "candidates": []
        }
        for j, cand in enumerate(ex['candidates']):
            if cur_cand_idx in sampled_cands_idxs:
                sampled_ex['candidates'].append(cand)
            cur_cand_idx += 1
        if len(sampled_ex['candidates']) > 0:
            sampled_train_data.append(sampled_ex)
    # report statistics
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    cand_token_ids = [[tokenizer.encode(x['text'], add_special_tokens=False)
                       for x in ex['candidates']] for ex in sampled_train_data]
    cand_token_lens = [[len(x) for x in cand_token_id]
                       for cand_token_id in cand_token_ids]
    cand_token_lens = list(chain(*cand_token_lens))

    logging.info("Downsampled train data statistics:")
    logging.info("# Examples: {}".format(len(sampled_train_data)))
    logging.info("# Avg. Unique outputs: {}".format(sum(
        [len(x['candidates']) for x in sampled_train_data]) / len(sampled_train_data)))
    logging.info("# Avg. Candidates Min/Mean/Max Length: {},{},{}".format(
        np.min(cand_token_lens), np.mean(cand_token_lens), np.max(cand_token_lens)))
    logging.info("Domain distribution:")
    domain_counter = Counter([x['data_source'].split("_")[-1]
                             for x in sampled_train_data])
    for domain in domain_counter:
        logging.info("  {}: {}".format(domain, domain_counter[domain]))
    logging.info("Year distribution:")
    year_counter = Counter([x['data_source'].split("_")[0]
                           for x in sampled_train_data])
    for year in year_counter:
        logging.info("  {}: {}".format(year, year_counter[year]))

    return sampled_train_data


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
    parser.add_argument('--lang_pair', type=str, default='zh-en')
    parser.add_argument('--human_score_name', type=str,
                        default='mqm', choices=['mqm', 'da'])
    parser.add_argument('--data_dir', type=str, default='./')
    parser.add_argument('--max_ex_size', type=int, default=1000)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    args = parser.parse_args()

    dataset_name = f"RicardoRei/wmt-{args.human_score_name}-human-evaluation"
    lang_pair = args.lang_pair

    # Load the training data
    logging.info("Loading data from {}".format(dataset_name))
    dataset = load_dataset(dataset_name, split='train')
    dataset = dataset.filter(
        lambda x: x['lp'] == lang_pair)  # filter by lang_pair
    # only use data before WMT22
    dataset = dataset.filter(lambda x: x['year'] < 2022)

    if len(dataset) == 0:
        logging.error("No data found for lang_pair={}".format(lang_pair))
        exit(0)
    else:
        logging.info("Loaded {} examples for lang_pair={}".format(
            len(dataset), lang_pair))
    # Save the data
    args.data_dir = Path(args.data_dir)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    train_file = args.data_dir / f"wmt/{args.lang_pair}/train_data.json"
    train_file.parent.mkdir(parents=True, exist_ok=True)
    if not train_file.exists() or args.overwrite:
        train_data = format_train_data(
            dataset, lang_pair, args.human_score_name)
        train_data = down_sample_train_data(train_data, args.max_ex_size)
        with open(train_file, "w") as f:
            json.dump(train_data, f, indent=4, ensure_ascii=False)
            logging.info("Saved training data to {}".format(train_file))

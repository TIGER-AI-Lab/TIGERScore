"""
Get SUM train data
"""
import json
import argparse
import time
import logging
import numpy as np
from typing import List
from pathlib import Path
from datasets import load_dataset
from pathlib import Path
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
# log current time
logging.info("\nRunning time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

"""
Datasets:
50k data total
    - CNN/DM cnn_dailymail 30000 Yale-LILY/brio-cnndm-uncased
    - XSUM 2000 Yale-LILY/brio-xsum-cased
    - Newsroom 600 BARTScore
    - Samsum 2000
    - Gigaword 2000
    - REALSumm manu/REALSumm 2400 BARTScore
"""
# cnn/dm
from pathlib import Path
import json
import os
import sys

datas = []
tmp_data = {}
with open('./summeval/M0/paired/outputs.aligned.paired.jsonl', 'r') as f:
    data = f.readlines()
    for line in data:
        line = json.loads(line)
        tmp_data[line["text"]] = {
            "id": line["id"],
            "instruction": "Write a summary of the text below.",
            "input": line["text"],
            "output": line["reference"],
            "candidates": []
        }
        # datas.append({
        #     "id": line["id"],
        #     "instruction": "Write a summary of the text below.",
        #     "input": line["text"],
        #     "output": line["reference"],
        #     "candidates": []
        # })
file_caterogy =[
    "outputs.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs_rouge.aligned.paired.jsonl",
    "outputs_cnndm.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs_ptrgen.aligned.paired.jsonl",
    "outputs_extabs+rl+rerank.aligned.paired.jsonl",
    "outputs_transformer.aligned.paired.jsonl",
    "outputs_novelty+lm.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs_coverage.aligned.paired.jsonl",
    "outputs_ext.aligned.paired.jsonl",
    "outputs_large.aligned.paired.jsonl",
    "outputs_abs_bert.aligned.paired.jsonl",
    "outputs_cnndm_bertsumextabs.aligned.paired.jsonl",
    "outputs_rl+supervised.aligned.paired.jsonl",
    "outputs.aligned.paired.jsonl",
    "outputs_cnndm.aligned.paired.jsonl",
    "outputs_dynamicmix_cnn_dailymail.aligned.paired.jsonl",
]
for i, file in enumerate(file_caterogy):
    file_i = './summeval/M{}/paired/{}'.format(i, file)
    
    with open(file_i, 'r') as f:
        data = f.readlines()
        for line in data:
            line = json.loads(line)
            try:
                if line["text"] in tmp_data:
                    tmp_data[line["text"]]["candidates"].append({
                        "model": f"M{i}",
                        "decoding_method": "greedy",
                        "text": line["decoded"].replace("-lrb-", "(").replace("-rrb-", ")"), # for cnndm
                        "scores": {}
                    })
            except KeyError:
                continue
for k, v in tmp_data.items():
    datas.append(v)
datas = sorted(datas, key=lambda x: x["id"])

output_file = Path('./summeval/summ_gen.json')
with open(output_file, 'w') as f:
    json.dump(datas, f, indent=4,ensure_ascii=False)

        

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
    parser.add_argument('--data_dir', type=str, default='../../data/sum')
    parser.add_argument('--max_ex_size', type=int, default=1000)
    parser.add_argument('--overwrite', type=str2bool, default=False)
    args = parser.parse_args()
    
    dataset_list = ["cnn_dailymail:3.0.0","xsum","gigaword","samsum"]
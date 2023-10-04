import json
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO)
"""
Datasets:
    - SummEval(CNNDM)
    - XSUM
    - Newsroom
    - Samsum
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data')
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    agg_data = []
    datasets = ["summeval","xsum","newsroom","samsum"]
    np.random.seed(42)
    logging.info("Aggregating data from {}".format(args.data_dir))
    for dataset in datasets:
        with open(args.data_dir /"{}".format(dataset) /"train_data_prepared.json", "r") as f:
            data = json.load(f)
            data = [x for x in data if len(x['candidates']) > 0]
            # 放宽长度，提升Unique outputs，均匀的error数量，均匀的candidate质量
            data = [x for x in data if len(tokenizer.encode(x['input'], add_special_tokens=False)) < 800 and  len(tokenizer.encode(x['output'], add_special_tokens=False)) < 400]
            np.random.shuffle(data)
            if dataset != "summeval":
                for i, item in enumerate(data):
                    new_cands = sorted(item['candidates'], key=lambda x: x['scores']['bart_score_cnn'], reverse=True)
                    if i < len(data)//6 :
                        item['candidates'] = [new_cands[0]]
                    elif i < len(data)//3:
                        item['candidates'] = [new_cands[2]]
                    else:
                        item['candidates'] = [new_cands[-1]]
            else:
                for i, item in enumerate(data):
                    new_cands = sorted(item['candidates'], key=lambda x: x['scores']["bart_score_cnn"], reverse=True)
                    item['candidates'] = [new_cands[0],new_cands[len(new_cands)//2],new_cands[-1]]
            logging.info("Loaded {} data from {}".format(len(data), args.data_dir /"{}".format(dataset) /"train_data_prepared.json"))
            logging.info("  Unique outputs: {}".format(np.sum([len(x['candidates']) for x in data])))
            for x in data:
                x['dataset'] = dataset
            agg_data += data
    logging.info("# Total aggregated # {} data for summarization".format(len(agg_data)))
    logging.info("# Unique outputs: {}".format(np.sum([len(x['candidates']) for x in agg_data])))
    # length statistics
    input_lens = [len(tokenizer.encode(x['input'], add_special_tokens=False)) for x in agg_data]
    cand_lens = [[len(tokenizer.encode(cand['text'], add_special_tokens=False)) for cand in x['candidates']] for x in agg_data]
    cand_lens = list(chain(*cand_lens))
    ref_lens = [len(tokenizer.encode(x['output'], add_special_tokens=False))  for x in agg_data]
    logging.info("Input length statistics:")
    logging.info("  Avg/Max/90%/50%: {}/{}/{}/{}".format(np.mean(input_lens), np.max(input_lens), np.percentile(input_lens, 90), np.percentile(input_lens, 50)))
    logging.info("Candidate length statistics:")
    logging.info("  Avg/Max/90%: {}/{}/{}".format(np.mean(cand_lens), np.max(cand_lens), np.percentile(cand_lens, 90)))
    logging.info("Reference length statistics:")
    logging.info("  Avg/Max/90%: {}/{}/{}".format(np.mean(ref_lens), np.max(ref_lens), np.percentile(ref_lens, 90)))
    output_file = args.data_dir /"sum" / "train_data_new.json"
    with open(output_file, "w") as f:
        json.dump(agg_data, f, indent=4, ensure_ascii=False)
        logging.info("Saved aggregated data to {}".format(output_file))







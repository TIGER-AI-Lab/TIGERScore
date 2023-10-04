import json
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import chain
from transformers import AutoTokenizer
logging.basicConfig(level=logging.INFO)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/wmt')
    args = parser.parse_args()
    args.data_dir = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    agg_data = []
    logging.info("Aggregating data from {}".format(args.data_dir))
    for lang_pair in args.data_dir.iterdir():
        if not lang_pair.is_dir():
            continue
        src_lang, tgt_lang = lang_pair.name.split("-")
        with open(lang_pair / "train_data.json") as f:
            data = json.load(f)
        agg_data.extend(data)
    logging.info("# Total aggregated # {} data for translation".format(len(agg_data)))
    logging.info("# Unique outputs: {}".format(np.sum([len(x['candidates']) for x in agg_data])))
    # report total statistics
    # year statistics
    logging.info("Year statistics:")
    year_counter = Counter([x['data_source'].split("_")[0] for x in agg_data])
    for year in year_counter:
        logging.info("  {}: {}".format(year, year_counter[year]))
    # lang_pair statistics
    logging.info("Lang pair statistics:")
    lang_pair_counter = Counter([x['data_source'].split("_")[1] for x in agg_data])
    for lang_pair in lang_pair_counter:
        logging.info("  {}: {}".format(lang_pair, lang_pair_counter[lang_pair]))
    # domain statistics
    logging.info("Domain statistics:")
    domain_counter = Counter([x['data_source'].split("_")[-1] for x in agg_data])
    for domain in domain_counter:
        logging.info("  {}: {}".format(domain, domain_counter[domain]))
    # Model statistics
    logging.info("System models statistics:")
    model_counter = Counter(list(chain(*[[cand['model'] for cand in x['candidates']] for x in agg_data])))
    for model in model_counter:
        logging.info("  {}: {}".format(model, model_counter[model]))
    # length statistics
    input_lens = [len(tokenizer.encode(x['input'], add_special_tokens=False)) for x in agg_data]
    cand_lens = [[len(tokenizer.encode(cand['text'], add_special_tokens=False)) for cand in x['candidates']] for x in agg_data]
    cand_lens = list(chain(*cand_lens))
    ref_lens = [[len(tokenizer.encode(ref, add_special_tokens=False)) for ref in x['refs']] for x in agg_data]
    ref_lens = list(chain(*ref_lens))
    logging.info("Input length statistics:")
    logging.info("  Min/Avg/Max/90%: {}/{}/{}/{}".format(np.min(input_lens), np.mean(input_lens), np.max(input_lens), np.percentile(input_lens, 90)))
    logging.info("Candidate length statistics:")
    logging.info("  Min/Avg/Max/90%: {}/{}/{}/{}".format(np.min(cand_lens), np.mean(cand_lens), np.max(cand_lens), np.percentile(cand_lens, 90)))
    logging.info("Reference length statistics:")
    logging.info("  Min/Avg/Max/90%: {}/{}/{}/{}".format(np.min(ref_lens), np.mean(ref_lens), np.max(ref_lens), np.percentile(ref_lens, 90)))
    with open(args.data_dir / "train_data.json", "w") as f:
        json.dump(agg_data, f, indent=4, ensure_ascii=False)
        logging.info("Saved aggregated data to {}".format(args.data_dir / "train_data.json"))

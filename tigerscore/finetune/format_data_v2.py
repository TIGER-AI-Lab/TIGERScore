"""
Usage: transforms the xgptscore data format into the alpaca data format for finetuning.

"""
import sys
import os
sys.path.append("../")
templates_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(templates_path)
from tqdm import tqdm
from transformers import AutoTokenizer
from common.datasets_config import DATASETS_CONFIG
from pathlib import Path
from string import Template
import json
import logging
import fire
import regex as re
import numpy as np
from collections import Counter
from itertools import chain


# FINETUNE_INST = "You are evaluating errors in a model-generated output for a(an) ${task} task."
# FINETUNE_INPUT = """\
# Task instruction: ${generation_instruction}
# Source: ${input_context}
# Model-generated Output: ${hypothesis_output}

# Based on the given task instruction and source, identify the major and minor errors in this model-generated output.
# Note that Major errors refer to actual errors that affects the task severely, and Minor errors refer to small imperfections, and purely subjective opinions about the output.
# For each error you give in the response, please also elaborate the following information:
# - error location (the words that are wrong in the output)
# - error aspect it belongs to.
# - explanation why it's an error, and the correction suggestions.
# - severity of the error ("Major" or "Minor").
# - reduction of score (between 0.5 and 5)

# Your evaluation output in the json format:
# """
INST = "You are evaluating errors in a model-generated output for a given instruction."
TEMPLATE = """\
Instruction: 
${generation_instruction}
${input_context}

Model-generated Output: 
${hypothesis_output}

For each error you give in the response, please also elaborate the following information:
- error location (the words that are wrong in the output)
- error aspect it belongs to.
- explanation why it's an error, and the correction suggestions.
- severity of the error ("Major" or "Minor"). 
- reduction of score (between 0.5 and 5 given the severity of the error)

Your evaluation output:\
"""

def main(
    seed: int = 42,
    input_file: str = None,
    output_file: str = None,
    overwrite: bool = False,
    max_eval_input_length: int = None,
    max_eval_hyp_length: int = None,
    max_eval_output_length: int = None,
):
    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    with open(input_file, 'r') as f:
        if input_file.endswith(".json"):
            data = json.load(f)
        elif input_file.endswith(".jsonl"):
            data = [json.loads(line) for line in f]
    formatted_data = []
    for item in data:
        inst = INST
        input_ = Template(TEMPLATE).substitute(
            generation_instruction=item['instruction'],
            input_context=item['input_context'],
            hypothesis_output=item['hypo_output']
        )
        output_ = item['errors']
        formatted_data.append({
            "instruction": inst,
            "input": input_,
            "output": output_,
        })

    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved to {output_file}")

    # count the dataset statistics
    dataset_statistics = {}
    dataset_statistics["#total"] = len(formatted_data)
    dataset_statistics["#unique input"] = len(
        set([item["input"] for item in formatted_data]))
    input_lens = [len(tokenizer.encode(item["input"]))
                  for item in tqdm(formatted_data, desc="Counting input length")]
    output_lens = [len(tokenizer.encode(item["output"]))
                   for item in tqdm(formatted_data, desc="Counting output length")]
    total_lens = [x + y for x, y in zip(input_lens, output_lens)]
    dataset_statistics["input_length"] = {}
    dataset_statistics["input_length"]["mean"] = np.mean(input_lens).item()
    dataset_statistics["input_length"]["percentile"] = np.percentile(
        input_lens, [0, 25, 50, 90, 100]).tolist()
    dataset_statistics["input_length"]["max"] = max(input_lens)
    dataset_statistics["input_length"]["min"] = min(input_lens)
    dataset_statistics["output_length"] = {}
    dataset_statistics["output_length"]["mean"] = np.mean(output_lens).item()
    dataset_statistics["output_length"]["percentile"] = np.percentile(
        output_lens, [0, 25, 50, 90, 100]).tolist()
    dataset_statistics["output_length"]["max"] = max(output_lens)
    dataset_statistics["output_length"]["min"] = min(output_lens)
    dataset_statistics["total_length"] = {}
    dataset_statistics["total_length"]["mean"] = np.mean(total_lens).item()
    dataset_statistics["total_length"]["percentile"] = np.percentile(
        total_lens, [0, 25, 50, 90, 100]).tolist()
    dataset_statistics["total_length"]["max"] = max(total_lens)
    dataset_statistics["total_length"]["min"] = min(total_lens)
    error_aspects = [re.findall(
        r'(?<=Error aspect \d+: )[ \w]+', item['output']) for item in formatted_data]
    error_aspects = list(chain(*error_aspects))
    dataset_statistics["error_aspects_distribution"] = Counter(error_aspects)

    num_errors = [len(re.findall(r'(?<=Error location \d+: ).*(?=\n|$)',
                      item['output'])) for item in formatted_data]
    dataset_statistics["num_errors_distribution"] = Counter(num_errors)
    # severity distributions
    severities = [re.findall(
        r'(?<=Severity \d+: ).*(?=\n|$)', item['output']) for item in formatted_data]
    severities = list(chain(*severities))
    dataset_statistics["severity_distribution"] = Counter(severities)
    # score reduction distributions
    score_reductions = [re.findall(
        r'(?<=Score reduction \d+: ).*(?=\n|$)', item['output']) for item in formatted_data]
    score_reductions = list(chain(*score_reductions))
    score_reductions = [abs(float(x.replace(" ", "")))
                        for x in score_reductions]
    dataset_statistics["score_reduction_distribution"] = Counter(
        score_reductions)

    print(dataset_statistics)
    output_file = Path(output_file).with_suffix(".statistics.json")
    with open(output_file, "w") as f:
        json.dump(dataset_statistics, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved statistics to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)

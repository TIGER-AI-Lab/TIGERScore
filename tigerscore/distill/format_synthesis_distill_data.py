"""
Usage: transforms the xgptscore data format into the alpaca data format for finetuning.

"""
import json
import logging
import sys
import os
import random
import fire
import regex as re
import numpy as np
from collections import Counter
from itertools import chain

sys.path.append("../")
templates_path = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(templates_path)
from string import Template
from common.evaluation import overall_eval
from xgptscore.constants import EVAL_ASPECTS
from pathlib import Path
from common.datasets_config import DATASETS_CONFIG
from transformers import AutoTokenizer
from tqdm import tqdm


FINETUNE_INST = "You are evaluating errors in a model-generated output for a(an) ${task} task."
FINETUNE_INPUT = """\
Task instruction: ${generation_instruction}
Source: ${input_context}
Model-generated Output: ${hypothesis_output}

Based on the given task instruction and source, identify errors in this model-generated output.
For each error you give in the response, please also elaborate the following information:
- error location (the words that are wrong in the output)
- error aspect it belongs to.
- explanation why it's an error, and the correction suggestions.
- severity of the error ("Major" or "Minor"). 
- reduction of score (an interger between 1 and 5 given the severity of the error)

Your evaluation output:
"""

def main(
    task: str,
    seed: int = 42,
    input_file: str = None,
    output_file: str = None,
    overwrite: bool = False,
    max_eval_input_length: int = None,
    max_eval_hyp_length: int = None,
    max_eval_output_length: int = None,
):
    if task == "all":
        tasks = list(DATASETS_CONFIG.keys())
    else:
        assert task in DATASETS_CONFIG.keys()
        tasks = [task]

    tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

    with open(input_file, 'r') as f:
        data = json.load(f)
    formatted_data = []
    for item in data:
            syn_output = item['responses'][-1]
            syn_output = syn_output.replace(": \n", ": ")
            # decode the synthesis outputs
            try:
                start_pos = syn_output.index("Generated incorrect output: ") + len("Generated incorrect output: ")
                end_pos = syn_output.index("\nError location 1")
                hyp = syn_output[start_pos:end_pos].strip('\n ')
                assert len(hyp) > 0
            except Exception as e:
                logging.warning("Failed to parse the synthesis output: {}".format(syn_output))
                continue
            inst = Template(FINETUNE_INST).substitute(task=task)
            input_context_ids = tokenizer.encode(item['input'], add_special_tokens=False)
            hyp_ids = tokenizer.encode(hyp, add_special_tokens=False)
            if max_eval_input_length is not None and len(input_context_ids) > max_eval_input_length:
                input_context = tokenizer.decode(input_context_ids[:max_eval_input_length]) + "..."
            else:
                input_context = item['input']
            if max_eval_hyp_length is not None and len(hyp_ids) > max_eval_hyp_length:
                hypothesis_output = tokenizer.decode(hyp_ids[:max_eval_hyp_length]) + "..."
            else:
                hypothesis_output = hyp
            input_ = Template(FINETUNE_INPUT).substitute(
                generation_instruction=item['instruction'],
                input_context=input_context,
                hypothesis_output=hypothesis_output,
            )
            try:
                error_locations = re.findall(r'(?<=Error location \d+: ).*(?=\n|$)', syn_output)
                error_aspects = re.findall(r'(?<=Error aspect \d+: ).*(?=\n|$)', syn_output)
                explanations = re.findall(r'(?<=Explanation \d+: ).*(?=\n|$)', syn_output)
                severities = re.findall(r'(?<=Severity \d+: ).*(?=\n|$)', syn_output)
                score_reductions = re.findall(r'(?<=Score reduction \d+: ).*(?=\n|$)', syn_output)
                score_reductions = [abs(int(x.replace(" ", ""))) for x in score_reductions]
            except Exception as e:
                logging.warning("Failed to parse the synthesis output: {}".format(syn_output))
                continue
            
            if not len(error_locations) == len(error_aspects) == len(explanations) == len(severities) == len(score_reductions):
                logging.warning("The number of errors properties does not match!: {}".format(syn_output))
                continue
            
            txt_output = "The model-generated output contains {} errors, with a total score reduction of {}.".format(
                len(error_locations),
                sum([int(score) for score in score_reductions]),
            )
            task_eval_aspects = list(EVAL_ASPECTS[task].keys())
            for i in range(len(error_locations)):
                txt_output += "\nError location {}: {}\n".format(i + 1, error_locations[i])
                txt_output += "Error aspect {}: {}\n".format(i + 1, error_aspects[i])
                txt_output += "Explanation {}: {}\n".format(i + 1, explanations[i])
                txt_output += "Severity {}: {}\n".format(i + 1, severities[i])
                txt_output += "Score reduction {}: {}".format(i + 1, score_reductions[i])
            output_ = txt_output.strip(' \n')
            formatted_data.append({
                "instruction": inst,
                "input": input_,
                "output": output_,
                "task": task,
            })
    
    # # append 20% non-error examples
    # for item in data:
    #     if random.random() < 0.2:
    #         inst = Template(FINETUNE_INST).substitute(task=task)
    #         input_context_ids = tokenizer.encode(item['input'], add_special_tokens=False)
    #         if max_eval_input_length is not None and len(input_context_ids) > max_eval_input_length:
    #             input_context = tokenizer.decode(input_context_ids[:max_eval_input_length]) + "..."
    #         else:
    #             input_context = item['input']
    #         input_ = Template(FINETUNE_INPUT).substitute(
    #             generation_instruction=item['instruction'],
    #             input_context=input_context,
    #             hypothesis_output=item['output'],
    #         )
    #         output_ = "The model-generated output contains 0 errors, with a total score reduction of 0."
    #         formatted_data.append({
    #             "instruction": inst,
    #             "input": input_,
    #             "output": output_,
    #             "task": task,
    #         })
    
    with open(output_file, 'w') as f:
        json.dump(formatted_data, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved to {output_file}")
        

    # count the dataset statistics
    dataset_statistics = {}
    dataset_statistics["#total"] = len(formatted_data)
    dataset_statistics["#unique input"] = len(set([item["input"] for item in formatted_data]))
    input_lens = [len(tokenizer.encode(item["input"])) for item in tqdm(formatted_data, desc="Counting input length")]
    output_lens = [len(tokenizer.encode(item["output"])) for item in tqdm(formatted_data, desc="Counting output length")]
    total_lens = [x + y for x, y in zip(input_lens, output_lens)]
    dataset_statistics["input_length"] = {}
    dataset_statistics["input_length"]["mean"] = np.mean(input_lens).item()
    dataset_statistics["input_length"]["percentile"] = np.percentile(input_lens, [0, 25, 50, 90, 100]).tolist()
    dataset_statistics["input_length"]["max"] = max(input_lens)
    dataset_statistics["input_length"]["min"] = min(input_lens)
    dataset_statistics["output_length"] = {}
    dataset_statistics["output_length"]["mean"] = np.mean(output_lens).item()
    dataset_statistics["output_length"]["percentile"] = np.percentile(output_lens, [0, 25, 50, 90, 100]).tolist()
    dataset_statistics["output_length"]["max"] = max(output_lens)
    dataset_statistics["output_length"]["min"] = min(output_lens)
    dataset_statistics["total_length"] = {}
    dataset_statistics["total_length"]["mean"] = np.mean(total_lens).item()
    dataset_statistics["total_length"]["percentile"] = np.percentile(total_lens, [0, 25, 50, 90, 100]).tolist()
    dataset_statistics["total_length"]["max"] = max(total_lens)
    dataset_statistics["total_length"]["min"] = min(total_lens)
    error_aspects = [re.findall(r'(?<=Error aspect \d+: ).*(?=\n|$)', item['output']) for item in formatted_data]
    error_aspects = list(chain(*error_aspects))
    dataset_statistics["error_aspects_distribution"] = Counter(error_aspects)
    # number of errors distributions
    num_errors = [len(re.findall(r'(?<=Error location \d+: ).*(?=\n|$)', item['output'])) for item in formatted_data]
    dataset_statistics["num_errors_distribution"] = Counter(num_errors)
    # severity distributions
    severities = [re.findall(r'(?<=Severity \d+: ).*(?=\n|$)', item['output']) for item in formatted_data]
    severities = list(chain(*severities))
    dataset_statistics["severity_distribution"] = Counter(severities)
    # score reduction distributions
    score_reductions = [re.findall(r'(?<=Score reduction \d+: ).*(?=\n|$)', item['output']) for item in formatted_data]
    score_reductions = list(chain(*score_reductions))
    score_reductions = [abs(int(x.replace(" ", ""))) for x in score_reductions]
    dataset_statistics["score_reduction_distribution"] = Counter(score_reductions)
    
    print(dataset_statistics)
    output_file = Path(output_file).with_suffix(".statistics.json")
    with open(output_file, "w") as f:
        json.dump(dataset_statistics, f, indent=4, ensure_ascii=False)
    logging.info(f"Saved statistics to {output_file}")
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
                        
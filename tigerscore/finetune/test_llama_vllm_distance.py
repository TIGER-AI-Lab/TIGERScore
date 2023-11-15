
import argparse
import json
import torch
import logging
import sys
import regex as re
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from vllm import LLM, SamplingParams
from typing import List
from string import Template
from mt_metrics_eval.stats import Correlation


MAX_INT = sys.maxsize

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
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
- reduction of score (an interger between 0.5 and 5 given the severity of the error)

Your evaluation output:
"""


def get_sum_penalties(eval_output: dict):
    """
    Args:
        eval_output: dict, the json output of the eval function

    Returns:
    """
    try:
        penalty_score = 0
        for aspect in eval_output:
            for penalty_point in eval_output[aspect]["penalty_points"]:
                penalty_score += penalty_point["score_reduction"]
        return - penalty_score
    except Exception:
        return None


def get_torch_dtype(dtype_str):
    """
        Get the torch dtype from a string
    """
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "int8":
        return torch.int8
    else:
        raise ValueError("Invalid dtype {}".format(dtype_str))


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n - 1):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch_data.append(data_list[start:end])

    last_start = (n - 1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


class MyCorrelation(Correlation):
    def __init__(self, num_sys: int, gold_scores: List[int], metric_scores: List[int]):
        # remove nan in metrics scores
        none_metric_scores_idxs = [idx for idx,
                                   x in enumerate(metric_scores) if x is None]
        logging.info("Remove {} nan scores from {} scores".format(
            len(none_metric_scores_idxs),
            len(metric_scores)
        ))
        gold_scores = gold_scores.copy()
        # set gold scores to None if metric scores are None
        for idx in none_metric_scores_idxs[::-1]:
            gold_scores[idx] = None
        super().__init__(num_sys, gold_scores, metric_scores)


def main(args):

    if args.output_path is not None:
        output_file = Path(args.output_path)
    else:
        output_file = Path(args.data_path).with_suffix(
            '.xgptscore.output.json')
    if not output_file.exists() or args.overwrite:
        logging.info("Loading model...")
        sampling_params = SamplingParams(
            temperature=0, top_p=1, max_tokens=1024)
        llm = LLM(model=args.model_name_or_path, tensor_parallel_size=1)
        logging.info("Model loaded from {}".format(args.model_name_or_path))

        eval_outputs = []

        logging.info("Load input data from {}".format(args.data_path))
        with open(args.data_path, "r") as f:
            input_data = json.load(f)
        formatted_data = []
        for item in input_data:
            inst = Template(FINETUNE_INST).substitute(task=args.task)
            input_ = Template(FINETUNE_INPUT).substitute(
                task=args.task,
                generation_instruction=item['instruction'],
                input_context=item['input'],
                hypothesis_output=item['output'],
            )
            if "candidates" in item:
                del item["candidates"]
            formatted_data.append({
                "instruction": inst,
                "input": input_,
            })
            prompt_sources = [example['instruction'] + '\n' +
                              example['input'] for example in formatted_data]
            prompt_sources = [x.strip(' \n') + "\n" for x in prompt_sources]

        batch_prompts = batch_data(prompt_sources, batch_size=args.batch_size)

        for idx, batch_prompt in enumerate(batch_prompts):
            if isinstance(batch_prompt, list):
                pass
            else:
                batch_prompt = [batch_prompt]

            completions = llm.generate(batch_prompt, sampling_params)
            for output in completions:
                generated_text = output.outputs[0].text
                eval_outputs.append(generated_text)

        for idx, (item, eval_output) in enumerate(zip(input_data, eval_outputs)):
            item['eval_output'] = eval_outputs[idx]
            score_reductions = re.findall(
                r"(?<=\nScore reduction \d+: )(\d+\.\d+|\d+)", eval_outputs[idx])
            item['xgptscore'] = -sum(map(float, score_reductions))

        with open(output_file, 'w') as f:
            json.dump(input_data, f, indent=4, ensure_ascii=False)
        logging.info("Saved eval results to {}".format(output_file))
    else:
        with open(output_file, 'r') as f:
            input_data = json.load(f)
        for ex in input_data:
            score_reductions = re.findall(
                r"(?<=\nScore reduction \d+: )(\d+\.\d+|\d+)", ex['eval_output'])
            ex['xgptscore'] = -sum(map(float, score_reductions))
        with open(output_file, 'w') as f:
            json.dump(input_data, f, indent=4, ensure_ascii=False)
        logging.info("Loaded eval results from {}".format(output_file))
    # Compute correlation
    for item in input_data:
        xgptscores.append(item['xgptscore'])
    logging.info("Absolute score sum: {}".format(abs(sum(xgptscores))))
    logging.info("Average score: {}".format(sum(xgptscores) / len(xgptscores)))
    logging.info("Median score: {}".format(np.median(xgptscores)))
    logging.info("Standard deviation: {}".format(np.std(list(map(abs, xgptscores)))))



if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task", type=str, default="summarization")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--human_score_names", type=str, default="score")
    args = parser.parse_args()
    main(args)

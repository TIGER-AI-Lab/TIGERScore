import json
import transformers
import torch
import logging
import sys
import regex as re
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence, Dict, List
from torch.utils.data import Dataset, DataLoader
from string import Template
from mt_metrics_eval.stats import Correlation
from peft import PeftModel


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
FINETUNE_INST = "You are evaluating the errors in a model-generated output for a(an) ${task} task."
FINETUNE_INPUT = """\
Task instruction: ${generation_instruction}
Source: ${input_context}
Model-generated Output: ${hypothesis_output}

Based on the given task instruction and source, identify the major and minor errors in this model-generated output.
Note that Major errors refer to actual errors that affects the task severely, and Minor errors refer to small imperfections, and purely subjective opinions about the output.
For each error you give in the response, please also elaborate the following information:
- error location (the words that are wrong in the output)
- error aspect it belongs to.
- explanation why it's an error, and the correction suggestions.
- severity of the error ("Major" or "Minor").
- reduction of score (between 0.5 and 5)

Your evaluation output in the json format:
"""

# FINETUNE_INST = ""
# FINETUNE_INPUT = """\
# Task instruction: ${generation_instruction}
# Source: ${input_context}
# Output: ${hypothesis_output}
# Based on the given task instruction and source, identify the major and minor errors in this output for a(an) ${task} task. Note that Major errors refer to actual errors that affects the task severely, and Minor errors refer to smaller imperfections, and purely subjective opinions about the output.
# For each error you give in the response, please also elaborate the following information:
# - error location (the words that are wrong in the output)
# - error aspect it belongs to.
# - explanation why it's an error, and the correction suggestions.
# - severity of the error ("Major" or "Minor").
# - reduction of score (Major for 5, Minor for 1)
# Your evaluation output in the json format:
# """


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-125m", metadata={"help": "Path to the model checkpoint or pretrained model."})
    load_lora: str = field(default=None, metadata={
                           "help": "Path to the lora model checkpoint."})


@dataclass
class DataArguments:
    task: str = field(default="translation", metadata={
                      "help": "Task to evaluate on."})
    max_eval_input_length: int = field(
        default=256,
        metadata={
            "help": "Maximum input context (e.g. source text for translation) sequence length."},
    )
    max_eval_hyp_length: int = field(
        default=128,
        metadata={
            "help": "Maximum hypothesis output (e.g. candidate text for translation) sequence length."},
    )
    max_eval_output_length: int = field(
        default=256,
        metadata={
            "help": "Maximum output (e.g. evaluation results) sequence length."},
    )
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})
    output_path: str = field(default=None, metadata={
                             "help": "Path to the output file."})
    human_score_names: str = field(default="mqm,da", metadata={
                                   "help": "Name of the human scores."})
    overwrite: bool = field(default=False, metadata={
                            "help": "Overwrite the output file if it exists."})


@dataclass
class EvalArguments:
    batch_size: int = field(default=8, metadata={
                            "help": "Batch size for evaluation."})
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=512, metadata={"help": "Maximum sequence length."})
    torch_dtype: str = field(default="float32", metadata={
                             "help": "Data type to use for evaluation."})
    decoding_method: Literal["beam_search", "top_p_sampling"] = field(
        default="top_p_sampling", metadata={"help": "Decoding method."}
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-
                                                num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-
                                                  num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


class EvalDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_args: DataArguments, tokenizer: transformers.PreTrainedTokenizer):
        super(EvalDataset, self).__init__()
        logging.info("Loading data...")
        with open(data_args.data_path, "r") as f:
            self.examples = json.load(f)
        logging.info("Formatting inputs...")
        formatted_data = []
        for item in self.examples:
            for cand in item['candidates']:
                inst = Template(FINETUNE_INST).substitute(task=data_args.task)
                input_ = Template(FINETUNE_INPUT).substitute(
                    task=data_args.task,
                    generation_instruction=item['instruction'],
                    input_context=item['input'],
                    hypothesis_output=cand['text'],
                )
                formatted_data.append({
                    "instruction": inst,
                    "input": input_,
                })
        # for item in self.examples:
        #     inst = Template(FINETUNE_INST).substitute(task=data_args.task)
        #     input_ = Template(FINETUNE_INPUT).substitute(
        #         task=data_args.task,
        #         generation_instruction=item['instruction'],
        #         input_context=item['input'],
        #         hypothesis_output=item['output'][len("Generated incorrect output: "):item['output'].find("\n")],
        #     )
        #     formatted_data.append({
        #         "instruction": inst,
        #         "input": input_,
        #     })
        self.formated_data = formatted_data
        sources = [example['instruction'] + '\n' + example['input']
                   for example in formatted_data]
        sources = [x.strip(' \n') + "\n" for x in sources]
        logging.info("Tokenizing inputs... This may take some time...")
        self.input_ids = _tokenize_fn(sources, tokenizer)["input_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=None)


@dataclass
class DataCollatorForEvalDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance["input_ids"] for instance in instances]
        reverse_input_ids = [torch.flip(input_id, dims=(0,))
                             for input_id in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            reverse_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        input_ids = torch.flip(input_ids, dims=(1,))
        return dict(
            input_ids=input_ids,
            labels=None,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_eval_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    eval_dataset = EvalDataset(data_args=data_args, tokenizer=tokenizer)
    data_collator = DataCollatorForEvalDataset(tokenizer=tokenizer)
    return dict(train_dataset=None, eval_dataset=eval_dataset, data_collator=data_collator)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


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


def main(data_args, model_args, eval_args):

    if data_args.output_path is not None:
        output_file = Path(data_args.output_path)
    else:
        output_file = Path(data_args.data_path).with_suffix(
            '.xgptscore.output.json')
    if not output_file.exists() or data_args.overwrite:
        logging.info("Loading model...")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=eval_args.cache_dir,
            torch_dtype=get_torch_dtype(eval_args.torch_dtype),
            device_map="auto"
        )
        logging.info("Model loaded from {}".format(
            model_args.model_name_or_path))
        model.eval()
        if model_args.load_lora is not None:
            logging.info("Loading lora model...")
            model = PeftModel.from_pretrained(model, model_args.load_lora)
            model = model.merge_and_unload()
            logging.info("Loaded lora model from {}".format(
                model_args.load_lora))

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=eval_args.cache_dir,
            model_max_length=eval_args.model_max_length,
            padding_side="left",
            use_fast=False,
        )
        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model,
        )

        data_module = make_eval_data_module(
            tokenizer=tokenizer, data_args=data_args)
        dataloader = DataLoader(
            data_module["eval_dataset"],
            batch_size=eval_args.batch_size,
            collate_fn=data_module["data_collator"])

        generation_params = {
            "max_new_tokens": data_args.max_eval_output_length,
            "early_stopping": True,
        }
        if eval_args.decoding_method == "beam_search":
            generation_params["num_beams"] = 4
            logging.info("Using beam search with num_beams = {}".format(
                generation_params["num_beams"]))
        elif eval_args.decoding_method == "top_p_sampling":
            generation_params["do_sample"] = True
            generation_params["top_p"] = 1.0
            generation_params["temperature"] = 0.7
            logging.info("Using top-p sampling with top_p = {}, temperature = {}".format(
                generation_params["top_p"], generation_params["temperature"]))
        else:
            raise ValueError("Invalid decoding method {}".format(
                eval_args.decocoding_method))

        logging.info("Batch size: {}".format(eval_args.batch_size))
        eval_outputs = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", total=len(dataloader)):
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=data_args.max_eval_output_length,
                    do_sample=True,
                    top_p=0.95,
                    temperature=0.8,
                )
                input_len = input_ids.shape[1]
                outputs = outputs[:, input_len:]
                decoded_outputs = tokenizer.batch_decode(
                    outputs, skip_special_tokens=True)
                eval_outputs.extend(decoded_outputs)

        eval_examples = data_module["eval_dataset"].examples
        cand_idx = 0
        for ex in eval_examples:
            for cand in ex['candidates']:
                # cand['eval_output'] = json_postprocess(eval_outputs[cand_idx])
                # cand['xgptscore'] = get_xgptscore_from_json(cand['eval_output'])
                # if cand['xgptscore'] is None:
                #     cand['xgptscore'] = get_sum_penalties(cand['eval_output'])
                cand['eval_output'] = eval_outputs[cand_idx]
                score_reductions = re.findall(
                    r"(?<=\nScore reduction \d+: )(\d+\.\d+|\d+)", eval_outputs[cand_idx])
                cand['xgptscore'] = -sum(map(float, score_reductions))
                cand_idx += 1
        # for ex in eval_examples:
        #     # cand['eval_output'] = json_postprocess(eval_outputs[cand_idx])
        #     # cand['xgptscore'] = get_xgptscore_from_json(cand['eval_output'])
        #     # if cand['xgptscore'] is None:
        #     #     cand['xgptscore'] = get_sum_penalties(cand['eval_output'])
        #     ex['eval_output'] = eval_outputs[cand_idx]
        #     score_reductions = re.findall(r"(?<=\nScore reduction \d+: )(\d+\.\d+|\d+)", eval_outputs[cand_idx])
        #     ex['xgptscore'] = -sum(map(float, score_reductions))
        #     cand_idx += 1

        with open(output_file, 'w') as f:
            json.dump(eval_examples, f, indent=4, ensure_ascii=False)
        logging.info("Saved eval results to {}".format(output_file))
    else:
        with open(output_file, 'r') as f:
            eval_examples = json.load(f)
        for ex in eval_examples:
            for cand in ex['candidates']:
                score_reductions = re.findall(
                    r"(?<=\nScore reduction \d+: )(\d+\.\d+|\d+)", cand['eval_output'])
                cand['xgptscore'] = -sum(map(float, score_reductions))
        with open(output_file, 'w') as f:
            json.dump(eval_examples, f, indent=4, ensure_ascii=False)
        logging.info("Loaded eval results from {}".format(output_file))
    # Compute correlation
    human_score_names = data_args.human_score_names.split(',')

    for h_name in human_score_names:
        human_scores = []
        xgptscores = []
        for item in eval_examples:
            for cand in item['candidates']:
                for s_name, score in cand['scores'].items():
                    if s_name == h_name:
                        xgptscores.append(cand['xgptscore'])
                        human_scores.append(score)
                        break
                # assert flag, "No human score found in {}".format(cand['scores'])
        corr = MyCorrelation(1, human_scores, xgptscores)
        logging.info("Human score: {}".format(h_name))
        logging.info("Pearson correlation: {}".format(corr.Pearson()))
        logging.info("Spearman correlation: {}".format(corr.Spearman()))
        logging.info("Kendall correlation: {}".format(corr.Kendall()))

    # for item in eval_examples:
    #     for cand in item['candidates']:
    #         flag = False
    #         for s_name, score in cand['scores'].items():
    #             if s_name in human_score_names:
    #                 human_scores.append(score)
    #                 flag = True
    #                 break
    #         assert flag, "No human score found in {}".format(cand['scores'])
    #         xgptscores.append(cand['xgptscore'])
    # corr = MyCorrelation(1, human_scores, xgptscores)
    # logging.info("Pearson correlation: {}".format(corr.Pearson()))
    # logging.info("Spearman correlation: {}".format(corr.Spearman()))
    # logging.info("Kendall correlation: {}".format(corr.Kendall()))


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, EvalArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(level=logging.INFO)
    main(data_args, model_args, eval_args)

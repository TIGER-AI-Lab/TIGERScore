import torch
import regex as re
from transformers import AutoTokenizer, AutoModelForCausalLM
from string import Template
from typing import List
from tqdm import tqdm


TIGERScore_model_map = {
    "7b": "TIGER-Lab/TIGERScore-7B-V1.0",
    "13b": "TIGER-Lab/TIGERScore-13B-V1.0",
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
- reduction of score (between 0.5 and 5 given the severity of the error)

Your evaluation output:
"""


class TIGERScorer(object):
    def __init__(self, model_size, quantized=False):
        """Initialize the TIGERScore model.

        Args:
            model_size:
                either "7b" or "13b"
            quantized:
                If true, load the 4-bit quantized version of the model.
                quantized version occupies 2-3 times less memory but will running slower.

        """
        assert model_size in TIGERScore_model_map
        model_name = TIGERScore_model_map[model_size]
        if quantized:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                load_in_4bit=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side="left",
        )

    def decode_tigerscore_output(self, output):
        """Decode the output of TIGERScore model into structured error explanations.

        Args:
            output (str):
                the output of TIGERScore model.
        Returns:
            errors (List[Dict]):
                structured error explanations for each error in the output.
                Each error explanation is a dictionary with the following fields:
                    - error_location (str): the words that are wrong in the output
                    - error_aspect (str): the aspect of the error
                    - error_explanation (str): explanation why it's an error, and the correction suggestions
                    - error_severity (str): severity of the error ("Major" or "Minor")
                    - score_reduction (float): reduction of score (between 0.5 and 5 given the severity of the error)
                There can be multiple errors in each input.
        """
        result = {}
        result['num_errors'] = re.search(
            r"(?<=The model-generated output contains )\d+(?= errors)", output).group(0)
        result['score'] = re.search(
            r"(?<=, with a total score reduction of )\d+", output).group(0)
        result['num_errors'] = int(result['num_errors'])
        result['score'] = -float(result['score'])
        result['errors'] = {}
        error_locations = re.findall(
            r"(?<=Error location \d+: ).*?(?=\n)", output)
        error_aspects = re.findall(r"(?<=Error aspect \d+: ).*?(?=\n)", output)
        error_explanations = re.findall(
            r"(?<=Explanation \d+: ).*?(?=\n)", output)
        error_severities = re.findall(r"(?<=Severity \d+: ).*?(?=\n)", output)
        score_reductions = re.findall(
            r"(?<=\nScore reduction \d+: )(\d+\.\d+|\d+)", output)
        assert len(error_locations) == len(error_aspects) == len(error_explanations) == len(error_severities) == len(score_reductions), \
            "The number of errors does not match."
        for i in range(len(error_locations)):
            error = {}
            error['location'] = error_locations[i]
            error['aspect'] = error_aspects[i]
            error['explanation'] = error_explanations[i]
            error['severity'] = error_severities[i]
            error['score_reduction'] = score_reductions[i]
            result['errors'][f"error_{i}"] = error
        return result

    def _score_batch(
        self,
        tasks: List[str],
        insts: List[str],
        input_contexts: List[str],
        hypo_outputs: List[str],
        **generate_kwargs
    ):
        """Internal function to score a batch of inputs.
        Args:
            (See score() function)
        Returns:
            (See score() function)
        """
        assert len(tasks) == len(insts) == len(
            input_contexts) == len(hypo_outputs)
        inst_template = Template(FINETUNE_INST)
        input_template = Template(FINETUNE_INPUT)

        insts = [inst_template.substitute(task=task) for task in tasks]
        inputs = [
            input_template.substitute(
                generation_instruction=inst,
                input_context=input_context,
                hypothesis_output=hypo_output
            )
            for inst, input_context, hypo_output in zip(insts, input_contexts, hypo_outputs)
        ]
        prompts = [(inst + "\n" + input_part).strip("\n ") +
                   "\n" for inst, input_part in zip(insts, inputs)]

        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                   truncation=True, max_length=self.tokenizer.model_max_length)
        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)
        gen_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 512,
            "do_sample": True,
            "top_k": 1,
            "num_return_sequences": 1,
        }
        gen_params.update(generate_kwargs)
        outputs = self.model.generate(**gen_params)

        # input_len = input_ids.shape[1]
        # completion_ids = [output[input_len:] for output in outputs]
        completion_ids = outputs
        completions = [self.tokenizer.decode(
            completion, skip_special_tokens=True) for completion in completion_ids]
        tigerscore_results = []
        for completion in completions:
            try:
                result = self.decode_tigerscore_output(completion)
                result['score'] = result['score']
                result['num_errors'] = result['num_errors']
                result['errors'] = result['errors']
            except Exception:
                result = {}
                result['score'] = None
                result['num_errors'] = None
                result['errors'] = None
            result['raw_output'] = completion
            tigerscore_results.append(result)
        return tigerscore_results

    def score(
        self,
        tasks: List[str],
        insts: List[str],
        input_contexts: List[str],
        hypo_outputs: List[str],
        batch_size: int = 8,
        **generate_kwargs
    ):
        """Score and identify errors in the model-generated outputs

        Example Usage:
        ```python
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            from datasets import load_dataset
            from tigerscore import TIGERScorer
            scorer = TIGERScorer(model_size="7b", quantized=True)
            dataset = load_dataset("TIGER-Lab/MetricInstruct")
            num_few_examples = 10
            tasks = dataset["train_mix"]['task'][0:num_few_examples]
            insts = dataset["train_mix"]['instruction'][0:num_few_examples]
            input_contexts = dataset["train_mix"]['input_context'][0:num_few_examples]
            hypo_output = dataset["train_mix"]['hypo_output'][0:num_few_examples]
            results = scorer.score(tasks, insts, input_contexts, hypo_output)
            scores = [result["score"] for result in results]
            print(results)
        ```

        Args:
            tasks:
                a list of tasks, each task is a string. 
                6 pre-defined tasks are:
                    - "translation"
                    - "summarization"
                    - "data2text"
                    - "mathQA"
                    - "long-form QA"
                    - "instruction-following"
                You can also define your own task.
            insts:
                a list of instruction strings; One instruction example is:
                "Translate the following text from German to English."
                A instruction is a short description of the task. 
                It contains specific requirements for the model-generated output.
            input_contexts:
                a list of input contexts; One input context example is the source German text.
            hypo_outputs:
                a list of hypothesis outputs; One hypothesis output example is the model-generated English translation.
            batch_size:
                batch size for scoring. 
            generate_kwargs:
                keyword arguments for the model.generate() method. 
                See https://huggingface.co/transformers/main_classes/model.html
        Returns:
            results (List[Dict]):
                Contains the following fields:
                - score (float): the TIGERScore score for the input.
                - num_errors (int): the number of errors in the input.
                - errors (List[Dict]): structured error explanations for each error in the input.
                    - location (str): the words that are wrong in the output
                    - aspect (str): the aspect of the error
                    - explanation (str): explanation why it's an error, and the correction suggestions
                    - severity (str): severity of the error ("Major" or "Minor")
                    - reduction (float): reduction of score (between 0.5 and 5 given the severity of the error)
                - raw_output (str): the raw output of the TIGERScore model.
        """
        assert len(tasks) == len(insts) == len(
            input_contexts) == len(hypo_outputs)
        results = []
        for i in tqdm(
            range(0, len(tasks), batch_size),
            desc="TIGERScore Batch Scoring",
            total=len(tasks) // batch_size + 1
        ):
            batch_tasks = tasks[i:i + batch_size]
            batch_insts = insts[i:i + batch_size]
            batch_input_contexts = input_contexts[i:i + batch_size]
            batch_hypo_outputs = hypo_outputs[i:i + batch_size]
            batch_results = self._score_batch(
                batch_tasks, batch_insts, batch_input_contexts, batch_hypo_outputs, **generate_kwargs
            )
            results.extend(batch_results)
        return results

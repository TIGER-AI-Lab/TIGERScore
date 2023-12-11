import os
import regex as re
import importlib
import llama_cpp
from string import Template
from typing import List
from tqdm import tqdm

TEMPLATE = """You are evaluating errors in a model-generated output for a given instruction.
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

Your evaluation output:
"""

class TIGERScorer(object):
    def __init__(self, model_name, quantized=False, use_vllm=False, use_llamacpp=False):
        """Initialize the TIGERScore model.

        Args:
            model_name:
                basic model names:
                    - "TIGER-Lab/TIGERScore-7B",
                    - "TIGER-Lab/TIGERScore-13B",
                for llamacpp models:
                    - "TIGER-Lab/TIGERScore-7B-GGUF",
                    - "TIGER-Lab/TIGERScore-13B-GGUF",
            quantized (Run on GPU):
                If true, load the 4-bit quantized version of the model.
                quantized version occupies 2-3 times less memory but will running slower.
            use_vllm (Run on GPU):
                If true, use the VLLM version of the model. The inference speed can be 0.2s per input.
                if false, use the Hugging face inference API. The inference speed can be slower.
                vllm currently does not work with quantized models, so quantized will be ignored if use_vllm is true.
            use_llamacpp (Run on CPU):
                True indicates that the model_name is a path to a llamacpp model to run on the CPU.
                Will ignore use_vllm if True.
        """
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.quantized = quantized
        self.use_llamacpp = use_llamacpp
        self.tokenizer = None
        if use_llamacpp:
            if use_vllm:
                print("Warning: use_vllm is ignored when use_llamacpp is True.")
            # assert model_name.endswith(".gguf"), "llamacpp model name should end with .gguf, please check if this model is a valid llamacpp model."
            if not os.path.exists(model_name):
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=model_name, filename="ggml-model-q4_0.gguf")
                self.model = llama_cpp.Llama(model_path, n_ctx=1024)
            else:
                self.model = llama_cpp.Llama(model_name, n_ctx=1024)
        elif use_vllm:
            import torch
            import vllm
            num_gpus = torch.cuda.device_count()
            self.model = vllm.LLM(model_name, dtype=torch.bfloat16, tensor_parallel_size=num_gpus)
        else:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
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
        self.template = Template(TEMPLATE)

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
            r"(?<=Error location \d+:[ \n]*).*?(?=\n)", output)
        error_aspects = re.findall(r"(?<=Error aspect \d+:[ \n]*).*?(?=\n)", output)
        error_explanations = re.findall(
            r"(?<=Explanation \d+:[ \n]*).*?(?=\n)", output)
        error_severities = re.findall(r"(?<=Severity \d+:[ \n]*).*?(?=\n)", output)
        score_reductions = re.findall(
            r"(?<=\nScore reduction \d+:[ \n]*)(\d+\.\d+|\d+)", output)
        assert len(error_locations) == len(error_aspects) == len(error_explanations) == len(error_severities) == len(score_reductions), \
            "The number of errors does not match."
        for i in range(len(error_locations)):
            error = {}
            error['location'] = error_locations[i].strip("\n ")
            error['aspect'] = error_aspects[i].strip("\n ")
            error['explanation'] = error_explanations[i].strip("\n ")
            error['severity'] = error_severities[i].strip("\n ")
            error['score_reduction'] = score_reductions[i].strip("\n ")
            result['errors'][f"error_{i}"] = error
        return result

    def _run_batch(self, prompts: List[str], **generate_kwargs):
        """Internal function to score a batch of inputs.
        Args:
            prompts (List[str]):
                a list of prompts.
            generate_kwargs:
                keyword arguments for the model.generate() method. 
                See https://huggingface.co/transformers/main_classes/model.html
        Returns:
            completions (List[str]):
        """
        encodings = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=self.tokenizer.model_max_length)
        input_ids = encodings["input_ids"].to(self.model.device)
        attention_mask = encodings["attention_mask"].to(self.model.device)
        gen_params = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": 1024,
            "do_sample": False,
            "top_p": 1.0,
            "temperature": 0.7,
            "num_return_sequences": 1,
        }
        gen_params.update(generate_kwargs)
        outputs = self.model.generate(**gen_params)

        # input_len = input_ids.shape[1]
        # completion_ids = [output[input_len:] for output in outputs]
        completion_ids = outputs
        completions = [self.tokenizer.decode(
            completion, skip_special_tokens=True) for completion in completion_ids]
        return completions

    def score(
        self,
        insts: List[str],
        hypo_outputs: List[str],
        input_contexts: List[str]=None,
        batch_size: int = 2,
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
            insts:
                a list of instruction strings; One instruction example is:
                "Translate the following text from German to English."
                A instruction is a short description of the task. 
                It contains specific requirements for the model-generated output.
            hypo_outputs:
                a list of hypothesis outputs; One hypothesis output example is the model-generated English translation.
            input_contexts:
                a list of input contexts; One input context example is the source German text.
            batch_size:
                batch size for scoring. 
            use_vllm:
                if True, use VLLM to inference.
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
        assert len(insts) == len(input_contexts) == len(hypo_outputs), \
            "The number of inputs does not match."
        prompt_template = self.template
        prompts = [
            prompt_template.substitute(
                generation_instruction=inst,
                input_context=input_context,
                hypothesis_output=hypo_output
            ).strip("\n ")
            for inst, input_context, hypo_output in zip(insts, input_contexts, hypo_outputs)
        ]
        
        if self.use_llamacpp:
            gen_params = {
                "max_tokens": generate_kwargs.get("max_new_tokens", 1024),
                "top_p": generate_kwargs.get("top_p", 1.0),
                "top_k": generate_kwargs.get("top_k", 40),
                "temperature": generate_kwargs.get("temperature", 0.7),
                "frequency_penalty": generate_kwargs.get("frequency_penalty", 0.0),
                "presence_penalty": generate_kwargs.get("presence_penalty", 0.0),
                "echo": False,
                "stream": generate_kwargs.get("stream", False),
            }
            unused_params = [key for key in generate_kwargs.keys() if key not in gen_params]
            if len(unused_params) > 0:
                print(f"Warning: the following parameters are not used in llamacpp inference: {unused_params}")
            outputs = []
            for prompt in tqdm(prompts, desc="TIGERScore (llamacpp) Batch Scoring"):
                output = self.model(prompt, **gen_params)
                outputs.append(output)
            completions = [output['choices'][0]['text'] for output in outputs]
        elif self.use_vllm:
            import vllm
            sampling_params = vllm.SamplingParams(
                max_tokens=1024,
                top_p=1.0,
                temperature=0.7,
                n=1,
            )
            for key, value in generate_kwargs.items():
                if hasattr(sampling_params, key):
                    setattr(sampling_params, key, value)
            vllm_outputs = self.model.generate(
                prompts,
                sampling_params=sampling_params,
            )
            completions = [output.outputs[0].text for output in vllm_outputs]
        else:
            completions = []
            for i in tqdm(
                range(0, len(prompts), batch_size),
                desc="TIGERScore Batch Scoring",
                total=len(prompts) // batch_size + 1
            ):
                batch_prompts = prompts[i:i+batch_size]
                batch_completions = self._run_batch(batch_prompts, **generate_kwargs)
                completions.extend(batch_completions)
        
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

if __name__ == "__main__":
    instruction = "Write an apology letter."
    input_context = "Reason: You canceled a plan at the last minute due to illness."
    hypo_output = "Hey [Recipient],\n\nI'm really sorry for ditching our plan. I suddenly got an opportunity for a vacation so I took it. I know this might have messed up your plans and I regret that.\n\nDespite being under the weather, I would rather go for an adventure. I hope you can understand my perspective and I hope this incident doesn't change anything between us.\n\nWe can reschedule our plan for another time. Sorry again for the trouble.\n\nPeace out,\n[Your Name]\n\n---"
    
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B")
    scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", quantized=True)
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B", use_vllm=True)
    # scorer = TIGERScorer(model_name="TIGER-Lab/TIGERScore-7B-GGUF", use_llamacpp=True)
    results = scorer.score([instruction], [hypo_output], [input_context])
    print(results)
    # {
    #     "num_errors": 2,
    #     "score": -8.0,
    #     "errors": {
    #         "error_0": {
    #             "location": "I suddenly got an opportunity for a vacation so I took it.",
    #             "aspect": "Incorrect reasoning",
    #             "explanation": "The error is in the reasoning provided for the cancellation. The original reason was due to illness, but the model generated an apology letter implying that the cancellation was due to a vacation opportunity, which is incorrect. The correction would be to maintain the original reason for the cancellation.",
    #             "severity": "Major",
    #             "score_reduction": "4.0"
    #         },
    #         "error_1": {
    #             "location": "Hey [Recipient]",
    #             "aspect": "Inappropriate language or tone",
    #             "explanation": "The opening salutation used by the model is too informal and not appropriate for an apology letter. The correction would be to use a more formal and respectful salutation such as \"Dear [Recipient]\".",
    #             "severity": "Major",
    #             "score_reduction": "4.0"
    #         }
    #     },
    # }
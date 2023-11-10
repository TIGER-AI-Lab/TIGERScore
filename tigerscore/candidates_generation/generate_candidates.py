"""
    This file is modified based on This file is modified based on:
    https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/candidate_generation/main_candidate_generation.py
    We thank the authors for sharing their code.
"""
# Generate candidates with the fine-tuned models.

from pathlib import Path
from candidates_generation.model_utils import (
    build_model,
    build_tokenizer,
    build_processor,
)
from common.utils import (
    seed_everything,
    str2bool,
    empty2None,
    empty2Noneint,
    fetch_images,
)
from candidates_generation.engine import (
    beam_search_step,
)
import argparse
import sys
import os
import torch
import json
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GenerationDataset(torch.utils.data.Dataset):
    """
        Dataset for generate candidates for given sources
    """

    def __init__(self, tokenizer, data, prompt_max_length, prefix=None, no_instruction=False, model_name=None):
        self.tokenizer = tokenizer
        self.data = data
        self.prompt_max_length = min(
            prompt_max_length, tokenizer.model_max_length)
        self.prefix = prefix if prefix is not None else ""
        self.no_instruction = no_instruction
        self.model_name = model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # apply the prompt template to get the proper prompt
        item = self.data[idx]
        if self.no_instruction:
            prompt = item['input']
        else:
            prompt = item['instruction'] + item['input']  # + "\n Description:"
        prompt = self.input2chat(prompt)
        encoded_prompt = self.tokenizer(prompt, max_length=self.prompt_max_length, padding='max_length',
                                        truncation=True, return_tensors="pt", return_token_type_ids=False)
        for key in encoded_prompt.keys():
            encoded_prompt[key] = encoded_prompt[key].squeeze(0)
        return {
            "id": item['id'],
            "encodings": encoded_prompt,
            "original": prompt,
        }

    def input2chat(self, _input):
        """
            Convert input to chat, For LLAMAs
        """

        cov_style = {
            "vicuna": ("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. ",
                       "USER:",
                       " ASSISTANT:"),
            "Wizard": ("Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n",
                       "### Instruction:\n",
                       "\n\n### Response: Let's think step by step."),
            "Llama-2": ("[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant.\n<</SYS>>\n\n",
                        "",
                        " [/INST]"),
            "chatglm": ("[Round 0]\n",
                        "问：",
                        "\n答："),
            "stablelm": ("""<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
""",
                         "<|USER|>",
                         "<|ASSISTANT|>"),
        }

        sys_prompt = ""
        role_user = ""
        role_assistant = ""
        for name in cov_style.keys():
            if name in self.model_name:
                sys_prompt, role_user, role_assistant = cov_style[name]
                break
        return sys_prompt + role_user + _input + role_assistant


class Image2TextGenerationDataset(torch.utils.data.Dataset):
    """
        Dataset for generate candidates for given sources
    """

    def __init__(self, tokenizer, processor, data):
        self.tokenizer = tokenizer
        self.processor = processor
        self.data = data

        self.image_urls = [item['input'] for item in data]
        self.images = fetch_images(self.image_urls, 4, retries=2, timeout=10)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # apply the prompt template to get the proper prompt
        item = self.data[idx]
        image = self.images[idx]
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.squeeze(0)
        return {
            "id": item['id'],
            "encodings": {
                "pixel_values": pixel_values
            }
        }


def get_model_size(n_param):
    """
        Get the size of the model in MB
    """
    units = ["K", "M", "B", "T"]
    unit = 0
    while n_param > 1000 and unit < len(units) - 1:
        n_param /= 1000
        unit += 1
    return "{:.2f}{}".format(n_param, units[unit])


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


def main(args):
    seed_everything(args.seed)
    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))
    # tokenizer
    tokenizer = build_tokenizer(
        args.model, cache_dir=args.cache_dir, resume_download=True, trust_remote_code=True)
    # model

    model_kwargs = {
        "torch_dtype": get_torch_dtype(args.dtype),
        "cache_dir": args.cache_dir,
        "resume_download": True,
        "trust_remote_code": True,
        "device_map": "auto"
    }
    if "7b" in args.model:
        model_kwargs.pop("device_map")

    model = build_model(
        args.model_type,
        args.model,
        **model_kwargs
    )
    if "device_map" not in model_kwargs:
        model = model.to(device)
    # model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe {} has {} trainable parameters".format(
        args.model, get_model_size(n_params)))

    datasets = args.dataset.split(',')
    sets = args.set.split(',')
    for dataset_name in datasets:
        for set in sets:
            print("\nGenerating candidates for {}-{}".format(dataset_name, set))

            data_file = Path(args.data_dir) / \
                dataset_name.replace(":", "/") / f"{set}_data.json"
            save_file = Path(args.data_dir) / dataset_name.replace(":", "/") / "candidates" / \
                set / args.decoding_method / \
                f"{args.model.split('/')[-1]}.json"
            # data
            data = json.load(open(data_file, 'r'))
            if args.end_idx is not None:
                data = data[:args.end_idx]
            if args.start_idx is not None:
                data = data[args.start_idx:]

            if isinstance(args.max_size, int) and args.max_size > 0:
                print("Truncating data from {} to {}".format(
                    len(data), args.max_size))
                data = data[:args.max_size]
            if len(data) == 0:
                print("No data to generate")
                return

            if args.start_idx is not None or args.end_idx is not None:
                if args.start_idx is None:
                    file_postfix = "0-{}".format(args.end_idx)
                elif args.end_idx is None:
                    file_postfix = "{}-{}".format(args.start_idx,
                                                  len(data) + args.start_idx)
                else:
                    file_postfix = "{}-{}".format(args.start_idx, args.end_idx)
                save_file = save_file.parent / \
                    "{}_{}.json".format(save_file.stem, file_postfix)
            save_file.parent.mkdir(parents=True, exist_ok=True)
            # check if the data have already been generated
            if os.path.exists(save_file):
                print("Found existing candidates.")
                if args.overwrite:
                    print("Overwriting existing data")
                else:
                    print("Not overwriting existing data. Finishing generating")
                    continue
            else:
                print("No existing candidates found. Generating candidates")

            if not args.image2text:
                dataset = GenerationDataset(
                    tokenizer, data, args.prompt_max_length, no_instruction=args.no_instruction, model_name=args.model)
            else:
                # image processor
                processor = build_processor(
                    args.model_type, args.model, cache_dir=args.cache_dir)
                dataset = Image2TextGenerationDataset(
                    tokenizer, processor, data)
            print("Total size of dataset: {}".format(len(dataset)))
            # data loader
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=args.inference_bs, shuffle=False)

            # summary generation
            candidates = []
            with torch.no_grad():
                for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating candidates"):
                    for k in batch['encodings'].keys():
                        batch['encodings'][k] = batch['encodings'][k].to(
                            device)
                    # generate candidates
                    outputs = beam_search_step(
                        inputs=batch['encodings'],
                        tokenizer=tokenizer,
                        base_model=model,
                        args=args,
                        # pad_token_id=tokenizer.pad_token_id, # debug for alpaca
                    )
                    _candidates = outputs['generated']
                    _logprobs = outputs['logprobs']
                    for id, _c, _l in zip(batch['id'], _candidates, _logprobs):
                        candidates.append({
                            "id": id,
                            "candidates": [
                                {
                                    "text": _c[i].strip(' \n'),
                                    "scores": {
                                        "logprobs": _l[i]
                                    }
                                }
                                for i in range(len(_c))
                            ]
                        })

            print("Total # of candidates: {}".format(len(candidates)))
            print("# of candidates per example: {}".format(
                len(list(candidates[0].values())[0])))
            # save
            json.dump(candidates, open(save_file, 'w'),
                      indent=4, ensure_ascii=False)
            print("Saved candidates to {}".format(save_file))

    print("Done generating candidates!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=str2bool, default=True)

    # data
    parser.add_argument('--data_dir', type=str, default='../../data')
    parser.add_argument('--dataset', type=empty2None, required=True)
    parser.add_argument('--set', type=str, default="test")
    parser.add_argument('--max_size', type=int, default=None)

    # model
    parser.add_argument('--model_type', type=str, default="flan-t5",)
    parser.add_argument('--model', type=str, default="google/flan-t5-xxl")
    parser.add_argument('--dtype', type=str, default="float32",
                        choices=["float32", "float16", "bfloat16", "int8"])
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--image2text', type=str2bool, default=False)

    # candidate generation
    parser.add_argument('--inference_bs', type=int, default=2)
    parser.add_argument('--decoding_method', type=str, default="diverse_beam_search",
                        choices=["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
    parser.add_argument('--num_return_sequences',
                        type=int, default=1)  # default: 1
    parser.add_argument('--num_beams', type=int, default=1)  # for beam search
    parser.add_argument('--num_beam_groups', type=int,
                        default=1)  # for diverse beam search
    parser.add_argument('--diversity_penalty', type=float,
                        default=1.0)  # for diverse beam search
    parser.add_argument('--top_p', type=float,
                        default=1.0)  # for top-p sampling
    parser.add_argument('--top_k', type=int, default=50)  # for top-k sampling
    # for top-p and top-k sampling
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--stemmer', type=str2bool, default=True)

    # generation config
    parser.add_argument('--prompt_max_length', type=int, default=512)
    parser.add_argument('--output_max_length', type=int, default=512)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--no_repeat_ngram_size', type=int, default=0)
    parser.add_argument('--no_instruction', type=str2bool, default=False)

    parser.add_argument('--start_idx', type=empty2Noneint, default=None)
    parser.add_argument('--end_idx', type=empty2Noneint, default=None)

    parser.add_argument('--overwrite', type=str2bool, default=True)
    args = parser.parse_args()

    if args.cache_dir is None:
        args.cache_dir = Path(os.path.abspath(
            __file__)).parent.parent.parent / "hf_models"
    if args.dataset is None:
        print("No dataset specified. Exiting")
    print("*" * 50)
    print(args)

    main(args)

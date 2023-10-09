"""
    This script downloads the dataset and splits it into train, val, test
    This file is modified based on:
    https://github.com/Ravoxsg/SummaReranker-ACL-22-/blob/main/src/candidate_generation/main_download_dataset.py
    We thank the authors for sharing their code.
"""

import os
import argparse
import sys
import datasets
import json
import PIL
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Union
from datasets import load_dataset_builder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import (
    generate_hash_code,
    str2bool,
    empty2None,
    fetch_single_image
)
from common.datasets_config import DATASETS_CONFIG
from pathlib import Path
from typing import List, Tuple, Union

def preprocess_example(
    ex: dict,
    task: str,
    dataset: str,
    dataset_version: str,
    input_key: str,
    output_key: str,
    id_key: str=None,
    dataset_cache_dir: str=None,
):
    """
    This function preprocesses the batch data according to the task and datasets
    Args:
        ex: a batch of data
        task: the task of the dataset
        dataset: the name of the dataset
        dataset_version: the version of the dataset
    Returns:
        id: the id of the example
        inst: the instruction of the example
        input: the input of the example
        output: the output of the example
    """
    if dataset_cache_dir is not None:
        dataset_cache_dir = Path(dataset_cache_dir)
    else:
        dataset_cache_dir = Path.cwd() / "../../data" / dataset / dataset_version / "cache"
    inst = None
    # task and dataset specific processing
    if task == "translation":
        input = ex["translation"][input_key]
        output = ex["translation"][output_key]
    elif task == "data2text":
        if "dart" in dataset:
            input = tub2str_dart(ex[input_key])
            output = ex[output_key]["text"][-1]
        elif "totto" in dataset: 
            from preprocess_utils_totto import get_highlighted_subtable
            subtable = get_highlighted_subtable(table=ex['table'], cell_indices=ex['highlighted_cells'], with_heuristic_headers=True)
            input = process_totto(subtable,ex)
            output = ex[output_key]["final_sentence"][-1]
        elif "logicnlg" in dataset:
            input = process_logicnlg(ex)
            output = ex[output_key]                   
        elif "wikitabletext" in dataset:
            input = process_wikitabletext(ex)
            output = ex[output_key]
        elif "web_nlg" in dataset:
            input = "\n".join([f"({s})" for s in ex[input_key]["mtriple_set"][0]]).strip()
            
            output = ex[output_key]["text"][-1]
        else:
            input = ex[input_key]
            output = ex[output_key]
    elif task == "image_captioning":
        dataset_cache_dir.mkdir(parents=True, exist_ok=True)
        input = ex[input_key]
        output = ex[output_key]
        if isinstance(input, str):
            # input is a url
            # filter out urls that are invalid
            try:
                image = fetch_single_image(input, retries=0, timeout=3)
            except Exception as e:
                return None, None, None
        elif isinstance(input, PIL.Image.Image):
            # input is PIL image
            image = input.convert("RGB")
            # save image 
            image_path = Path(dataset_cache_dir) / f"{generate_hash_code(output)}.jpg"
            image.save(image_path)
            input = image_path.absolute().as_posix()
        else:
            raise ValueError("Invalid input type: {}".format(type(input)))
    elif task == "long-form QA":
        if "asqa" in dataset:
            input = ex['ambiguous_question']
            knowledges = [x['content'] for x in ex['annotations'][0]["knowledge"] if x['content'] is not None]
            output = ex['annotations'][0]['long_answer']
        elif "natural_questions" in dataset:
            input = ex[input_key]
            output = ex[output_key]
            if len(output) <= 100:
                output = None # skip this example, we want long answer
        elif "cosmos_qa" in dataset:
            input = "Context: " + ex['context'] + "\n" + "Question: " + ex['question'] + "\n" + "Answer: "
            output = ex[f"answer{ex['label']}"]
            if "None of the above choices ." in output or len(output) <= 50:
                output = None # skip this example, we want long answer
        elif "eli5" in dataset:
            input = ex[input_key]
            if len(ex[output_key]) > 0:
                highest_score_idx = np.argmax(ex[output_key]['score'])
                answer = ex['answers']['text'][highest_score_idx]
                output = answer
            else:
                output = None
        elif "FeTaQA" in dataset:
            input = process_fetaqa(ex)
            output = ex[output_key]
            if len(output) <= 100:
                output = None
            pass
        else:
            input = ex[input_key]
            output = ex[output_key]
    elif task == "story_generation":
        if "roc" in dataset:
            input = "Title: " + ex['storytitle'] + "\n" + "Story: \n"
            for i in range(1, 5):
                input += "Sentence " + str(i) + ": " + ex['sentence' + str(i)] + "\n"
            input += "Sentence 5 (Ending): "
            output = ex['sentence5']
        elif "hellaswag" in dataset:
            input = "Context" + ex['ctx']
            output = ex['endings'][int(ex['label'])]
        elif dataset == "swag":
            input = ex['startphrase']
            output = ex['ending{}'.format(ex['label'])]
        else:
            input = ex[input_key]
            output = ex[output_key]
    elif task == "other":
        if "common_gen" in dataset:
            input = "Concepts: " + ", ".join(ex['concepts'])
            output = ex['target']
        elif "alpaca-gpt4" in dataset:
            input = ex["instruction"] + "\n" + ex["input"]
            output = ex[output_key]
        else:
            input = ex[input_key]
            output = ex[output_key]
    elif task == "mathQA":
        if "math_qa" in dataset:
            input = ex[input_key] #+ "\n Options: " + ex['options']
            output = ex[output_key]
        else:
            input = ex[input_key]
            output = ex[output_key]
        # input += "Let's think step by step.\n" # We can move this into instruction instead of input
    elif task == "code":
        if "code_contests" in dataset:
            input = ex[input_key]
            solution = ex[output_key]["solution"]
            solution = sorted(solution, key=lambda x:len(x))
            output = solution[0] if len(solution) > 0 else None
        else:
            input = ex[input_key]
            output = ex[output_key]
    elif task == "instruction-following":
        if "lima" in dataset:
            if len(ex['conversations']) != 2:
                input = None
                output = None
            else:
                input = ex['conversations'][0]
                output = ex['conversations'][1]
        elif "alpaca" in dataset or "Guanaco" in dataset:
            inst = ex["instruction"]
            input = ex["input"]
            output = ex["output"]
        elif "oasst1_en" in dataset:
            if len(ex['messages']) != 2:
                input = None
                output = None
            else:
                input = ex['messages'][0]['content']
                output = ex['messages'][1]['content']
        elif "dolly" in dataset:
            inst = ex["instruction"]
            input = ex["context"]
            output = ex["response"]
    else:
        input = ex[input_key]
        output = ex[output_key]

    # print("input: ", input)
    # print("output: ", output)
    if id_key is not None:
        id = ex[id_key]
    elif input is not None and output is not None:
        id = generate_hash_code(input + output)
    else:
        id = None

    return id, inst, input, output

def download_dataset(
    task: str,
    dataset: str,
    dataset_version: str,
    split: str,
    input_key: str,
    output_key: str,
    set_name: str,
    instruction: str=None,
    save_dir: Union[str, Path]=None,
    use_auth_token: dict = None,
    streaming: bool = False,
    shuffle: bool = False,
    seed: int = 42,
    overwrite: bool = False,
    id_key: str = None,
) -> None:
    data_dir = save_dir
    if data_dir is None:
        data_dir = Path(os.path.dirname(__file__)).parent.parent / "data" 
    dataset_dir = data_dir / dataset / (dataset_version or "")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    save_file = dataset_dir / f"{set_name}_data.json"
    print("Downloading {}:{}:{}".format(dataset, dataset_version, split))
    print("To save at: {}".format(save_file))
    if Path(save_file).exists():
        print("File already exists.")
        if not overwrite:
            print("Skipping download...")
            return
        else:
            print("Overwriting file at: {}".format(save_file))

    # check split
    if len(split.split("+")) > 1:
        splits = split.split("+")
    else:
        splits = [split]
    examples = []

    for _split in splits:
        start_idx = _split[_split.find("[")+1:_split.find(":")]
        end_idx = _split[_split.find(":")+1:_split.find("]")]
        __split = _split[:_split.find("[")]
                                  
        if start_idx == "":
            start_idx = 0
        else:
            start_idx = int(start_idx)
        if end_idx == "":
            end_idx = None
            num_examples = None
        else:
            end_idx = int(end_idx)
            num_examples = end_idx - start_idx
            if num_examples < 0:
                raise ValueError("0 examples selected. Please check your split: {}".format(_split))
        
        # load dataset
        try:
            # Try Streaming
            _streaming = streaming
            DS = datasets.load_dataset(
                dataset, dataset_version,
                split=__split, streaming=_streaming,
                use_auth_token=use_auth_token)
            _exs = [x for x in DS.take(100)]
        except Exception as e:
            print(e)
            print("Error loading dataset. Trying again without streaming...")
            _streaming = False
            DS = datasets.load_dataset(
                dataset, dataset_version,
                split=__split, streaming=_streaming, verification_mode='no_checks',
                use_auth_token=use_auth_token)
        print("Dataset loaded.")
        if shuffle:
            DS = DS.shuffle(seed=seed)
        
        if _streaming:
            iter_DS = DS.skip(start_idx)
        else:
            if start_idx > 0:
                iter_DS = [ex for ex in DS][start_idx:]
            else:
                iter_DS = DS
        # Process the examples
        unique_ids = set()
        with tqdm(desc="Processing {}:{}:{}".format(dataset, dataset_version, _split), 
            total=num_examples,
        ) as pbar:
            for i, ex in enumerate(iter_DS):
                # task and dataset specific processing
                dataset_cache_dir = dataset_dir / "cache"
                _id, _inst, _input, _output = preprocess_example(
                    ex, task, dataset, dataset_version, input_key, output_key, id_key, str(dataset_cache_dir))

               
                if _input is None or _output is None:
                    continue
                _datasource = dataset
                _datasource += f":{dataset_version}" if dataset_version else ""
                               
                if _id not in unique_ids:
                    examples.append({
                        "id": _id,
                        "instruction": _inst or instruction,
                        "input": _input, # input after applying the template
                        "output": _output,
                        "data_source": _datasource,
                        "task": task,
                    })
                    unique_ids.add(_id)
                    pbar.update(1)
                if num_examples is not None and len(unique_ids) >= num_examples:
                    break
    
    # Save the dataset
    with open(save_file, "w") as f:
        json.dump(examples, f, indent=4, ensure_ascii=False)    
    print(f"Saved {len(examples)} examples to {save_file}")
    
def tub2str_dart(tub: List[List[str]]):
    tub_str = ""
    for row in tub:
        tub_str += "( "
        for col in row[:-1]:
            tub_str += col + ", "
        tub_str += row[-1]
        tub_str += ")\n"
    return tub_str #+ '\n Start describing : '

def process_logicnlg(ex:dict):
    d = eval(ex['table'])
    tmp = ''
    
    for i in range(1,len(d)):
        tmp += 'In row {} , '.format(i)
        for j, s in enumerate(d[i]):
            entity = str(s)
            tmp += 'the {} is {} , '.format(d[0][j], entity)
        tmp = tmp[:-3] + ' . '

    tmp_prefix = 'Given the table title of "{}" . '.format(ex['title'])
    return tmp_prefix + tmp #+ '\n Start describing : '

def process_wikitabletext(ex:dict):
    d = eval(ex['content'])
    header = eval(ex['headers'])
    tmp = ''
    
    for i,s in enumerate(header):
        tmp += 'the {} is {} , '.format(s, d[i])
    tmp = tmp[:-3] + ' . '
    return tmp #+ '\n Start describing : '

def process_totto(subtable:list,ex:dict):
    table_str = ""
    
    if ex['table_page_title']:
        table_str += "Given the table title of {} . ".format(ex['table_page_title'])
    if ex['table_section_title']:
        table_str += "Given the table section title of {} . ".format(ex['table_section_title'])


    for item in subtable:
        cell = item["cell"]
        row_headers = item["row_headers"]
        col_headers = item["col_headers"]

        # The value of the cell.
        item_str = "Given the cell value of {} . ".format(cell["value"])

        # All the column headers associated with this cell.
        if col_headers:
            item_str += "The column header : "           
            col_headers_set = set()
            
            for col_header in col_headers:
                col_headers_set.add(col_header["value"])
            
            for col_header in col_headers_set:
              item_str += " {} , ".format(col_header)

            item_str = item_str[:-3] + " . "
        if row_headers:
            item_str += "The row header : "
            row_headers_set = set()
            
            for row_header in row_headers:
                row_headers_set.add(row_header["value"])
            for row_header in row_headers_set:
                item_str += " {} , ".format(row_header)
                
            item_str = item_str[:-3] + " . "
        
        
        # All the row headers associated with this cell.
        for row_header in row_headers:
          item_str += "The row header is {} . ".format(row_header["value"])
        table_str += item_str

    return table_str #+ '\n Start describing : '

def process_fetaqa(ex:dict):
    d = ex['table_array']
    highlights = ex['highlighted_cell_ids']
    highlights.sort()
    tmp = ''
    old_row = -1
    
    for row,col in highlights:
        if row != old_row:
            tmp = tmp[:-3] + ' . ' 
            tmp += 'In row {} , '.format(row)
            old_row = row

        tmp += 'the {} is {} , '.format(d[0][col], d[row][col])
        
    tmp = tmp[3:]
    tmp = tmp[:-3] + ' . '

    tmp_prefix = 'Given the table title of "{}" . '.format(ex['table_section_title'])
    question = ex['question']
    return tmp_prefix + tmp + question


def main(args):
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(os.path.dirname(__file__)).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets_config = DATASETS_CONFIG[args.task]
    for key in datasets_config.keys():
        if ':' not in key:
            dataset, dataset_version = key, None
        else:
            dataset, dataset_version = key.split(":")
        print("Loading dataset: {}:{}...".format(dataset, dataset_version))
        print("Dataset Details:")
        print("Dataset Name: {}".format(dataset))
        print("Dataset Version: {}".format(dataset_version))
        split_info = datasets_config[key]['split_info']
        instruction = datasets_config[key]['instruction']
        input_key = datasets_config[key]['input_key']
        output_key = datasets_config[key]['output_key']

        for set_name, split in split_info.items():
            download_dataset(
                task=args.task,
                dataset=dataset,
                dataset_version=dataset_version,
                split=split,
                instruction=instruction,
                input_key=input_key,
                output_key=output_key,
                set_name=set_name,
                save_dir=data_dir,
                use_auth_token=args.hf_use_auth_token,
                streaming=args.streaming,
                shuffle=args.shuffle,
                seed=args.seed,
                overwrite=args.overwrite,
            )

if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type = int, default = 42)
    # data
    parser.add_argument('--overwrite', type = str2bool, default = False)
    parser.add_argument('--streaming', type = str2bool, default = True)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--shuffle', type = str2bool, default = True)
    parser.add_argument('--task', type = str, default = None)
    parser.add_argument('--hf_use_auth_token', type = str, default = None,
                        help = "Huggingface auth token for downloading datasets; Load from env var if not provided.")
    
    args = parser.parse_args()
    if args.hf_use_auth_token is None:
        args.hf_use_auth_token = os.environ.get("HF_USE_AUTH_TOKEN", None)

    print(args)
    main(args)
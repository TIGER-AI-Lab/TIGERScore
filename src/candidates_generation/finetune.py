'''
From https://github.com/huggingface/notebooks/blob/main/examples/summarization.ipynb
'''

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
########!!!!!!!!!!!!!########
# set WANDB_DISABLED=True for offline logging
########!!!!!!!!!!!!!########

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import evaluate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
import torch
import argparse

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
    Trainer,
    TrainingArguments
)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.utils import (
    seed_everything,
    str2bool,
    empty2None,
    empty2Noneint,
    fetch_images,
)
from model_utils import (
    build_model,
    build_tokenizer,
    build_processor,
)
from pathlib import Path
from generate_candidates import get_model_size, get_torch_dtype,GenerationDataset
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

def main(args):
    # seed
    
    seed_everything(args.seed)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))
    
    # tokenizer
    tokenizer = build_tokenizer(args.model, cache_dir=args.cache_dir, resume_download=True)
    
    #dataset
    dataset_name = args.dataset
    train_data_file = Path(args.data_dir) / dataset_name.replace(":", "/") / f"train_data.json"
    val_data_file = Path(args.data_dir) / dataset_name.replace(":", "/") / f"validation_data.json"
    
    
    # train_data = json.load(open(train_data_file, 'r'))
    # val_data = json.load(open(val_data_file, 'r'))
    train_dataset = load_dataset('json', data_files=str(train_data_file), split="train")
    val_dataset = load_dataset('json', data_files=str(val_data_file),split="train")
    
    def preprocess_function(examples):
        max_input_length = args.prompt_max_length
        max_target_length = args.prompt_max_length
        model_inputs = tokenizer(examples['input'], max_length=max_input_length, truncation=True, padding="max_length")
        # Setup the tokenizer for targets
        # label_ids = tokenizer.encode(examples['output'], max_length=max_target_length, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["output"], max_length=max_target_length, truncation=True, padding="max_length")
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
        model_inputs["labels"] = labels["input_ids"]
        if model_inputs["labels"] == None:
            print("BOOM")
        return model_inputs
    
    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
    )
    tokenized_val_dataset = val_dataset.map(
        preprocess_function,
        batched=True,
    )
    
       
    
    # print("Dataset loaded: ", dataset)
    metric = evaluate.load(str(args.metric_path)+'/'+args.metric_name+".py")
    # model
    model = build_model(
        args.model_type,
        args.model,
        torch_dtype=get_torch_dtype(args.dtype),
        device_map="auto",
        cache_dir=args.cache_dir, resume_download=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\nThe {} has {} trainable parameters".format(args.model, get_model_size(n_params)))
        
   

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
    batch_size= 16
    training_args = Seq2SeqTrainingArguments(
        str(os.path.join(args.output_dir, args.output_name)),
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=1,
        predict_with_generate=True,
        fp16=True)
    
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, label_pad_token_id=label_pad_token_id,pad_to_multiple_of=8)
    
    trainer =  Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )   
    trainer.train()
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--cuda', type = str2bool, default = True)

    # data
    parser.add_argument('--data_dir', type = str, default = '../../data')
    parser.add_argument('--dataset', type = empty2None, required=True)
    # parser.add_argument('--set', type = str, default = "test")
    # parser.add_argument('--max_size', type = int, default = None)
     # model
    parser.add_argument('--model_type', type = str, default = "t5",)
    parser.add_argument('--model', type = str, default = "google/flan-t5-xxl")
    parser.add_argument('--dtype', type = str, default = "float32",
                        choices = ["float32", "float16", "bfloat16", "int8"])
    parser.add_argument('--cache_dir', type = str, default = None)
    parser.add_argument('--prefix', type = empty2None, default = None)
    parser.add_argument('--output_dir', type = str, default = None)
    parser.add_argument('--prompt_max_length', type = int, default = 1024)
    parser.add_argument('--no_instruction', type = str2bool, default = False)
    parser.add_argument('--output_name', type = str, default = None)
    parser.add_argument('--metric_path', type = str, default = None)
    parser.add_argument('--metric_name', type = str, default = 'rouge')
    
    args = parser.parse_args()
    
    if args.cache_dir is None:
        args.cache_dir = Path(os.path.abspath(__file__)).parent.parent.parent / "hf_models"
    if args.output_dir is None:
        args.output_dir = Path(os.path.abspath(__file__)).parent.parent.parent / "finetune_models"
    if args.metric_path is None:
        args.metric_path = Path(os.path.abspath(__file__)).parent.parent.parent / "hf_metrics"
    if args.dataset is None:
        print("No dataset specified. Exiting")
    if args.output_name is None:
        args.output_name = args.model + "_" + args.dataset
    print("*"*50)
    print(args)

    main(args)
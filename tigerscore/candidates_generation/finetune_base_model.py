"""
    This file is to finetune basic models for candidates generation.
    Code based on Huggingface Turorial.
"""
from common.evaluation import overall_eval
from model_utils import (
    build_model,
    build_tokenizer,
)
from typing import Optional, Sequence, Dict, List
from generate_candidates import get_model_size, get_torch_dtype
from dataclasses import dataclass, field
from transformers import (
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
import numpy as np
import logging
import transformers
import torch
import json
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append("..")
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_type: str
    model_name_or_path: str
    dtype: str = "float32"
    cache_dir: Optional[str] = None


@dataclass
class DataArguments:
    data_dir: str
    train_file: str
    eval_file: str = None
    eval_metrics: List[str] = field(default_factory=lambda: ["bleu", "rouge"])
    input_max_length: int = 512
    output_max_length: int = 128
    with_instruction: bool = False


def load_dataset(data_args):
    with open(data_args.train_file, 'r') as f:
        train_data = json.load(f)
    if data_args.eval_file:
        with open(data_args.eval_file, 'r') as f:
            eval_data = json.load(f)
    else:
        eval_data = None

    return train_data, eval_data


class SupervisedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings["input_ids"])


def preprocess_function(examples, tokenizer, data_args):
    if data_args.with_instruction:
        inputs = [x["instruction"] + "\n" + x["input"] for x in examples]
    else:
        inputs = [x["input"] for x in examples]
    inputs = [x.strip(' \n') for x in inputs]
    outputs = [x["output"] for x in examples]

    logging.warning("# of examples: {}".format(len(inputs)))
    logging.warning("Example of inputs:")
    print(inputs[0])
    logging.warning("Example of outputs:")
    print(outputs[0])

    model_inputs = tokenizer(
        inputs, max_length=data_args.input_max_length, truncation=True)
    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            outputs, max_length=data_args.output_max_length, truncation=True)

    logging.warning("Example of model inputs:")
    print("input_ids", model_inputs['input_ids'][0])
    print("attention_mask", model_inputs['attention_mask'][0])
    logging.warning("Example of labels:")
    print(labels['input_ids'][0])
    labels["input_ids"] = [
        [(_l if _l != tokenizer.pad_token_id else IGNORE_INDEX) for _l in label] for label in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return SupervisedDataset(model_inputs)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([torch.tensor(
            instance[key]) for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        # print(self.tokenizer.batch_decode(input_ids))
        # print(self.tokenizer.batch_decode(labels.masked_fill(labels == IGNORE_INDEX, self.tokenizer.pad_token_id)))
        # print("##" * 30)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def main(
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
):

    model = build_model(
        model_args.model_type,
        model_args.model_name_or_path,
        torch_dtype=get_torch_dtype(model_args.dtype),
        device_map="auto",
        cache_dir=model_args.cache_dir, resume_download=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.warning("The {} has {} trainable parameters".format(
        model_args.model_name_or_path, get_model_size(n_params)))
    tokenizer = build_tokenizer(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, resume_download=True)
    logging.warning("Loading dataset...")

    train_data, eval_data = load_dataset(data_args)
    logging.warning("Dataset loaded.")
    logging.warning("Preprocessing dataset...")
    train_dataset = preprocess_function(train_data, tokenizer, data_args)
    eval_dataset = preprocess_function(eval_data, tokenizer, data_args)
    logging.warning("Dataset preprocessed.")
    logging.warning("Loading data collator...")
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    logging.warning("Data collator loaded.")
    logging.warning("Loading trainer...")

    def compute_metrics(eval_pred):

        logits, labels = eval_pred
        labels[labels == IGNORE_INDEX] = tokenizer.pad_token_id
        logits[logits == IGNORE_INDEX] = tokenizer.pad_token_id
        predictions = tokenizer.batch_decode(logits, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        logging.warning("Example of predictions:")
        print(predictions[:3])
        logging.warning("Example of labels:")
        print(labels[:3])
        scores = overall_eval(predictions, labels,
                              metrics=data_args.eval_metrics)
        return {
            key: np.mean(value) for key, value in scores.items()
        }

    training_args.evaluation_strategy = "epoch"
    training_args.weight_decay = 0.01
    training_args.save_total_limit = 5
    training_args.predict_with_generate = True
    training_args.generation_num_beams = 4
    training_args.generation_max_length = data_args.output_max_length
    training_args.load_best_model_at_end = True
    logging.warning("Training arguments:")
    print(training_args)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logging.warning("Trainer loaded.")
    logging.warning("Training...")
    trainer.train()
    logging.warning("Training finished.")
    logging.warning("Saving model...")
    trainer.save_model(output_dir=os.path.join(
        training_args.output_dir, "checkpoint-best"))
    logging.warning("Model saved.")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, data_args, training_args)

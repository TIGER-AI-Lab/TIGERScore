# Set dataset configurations here
# the slices of each split should be all positive integers
NUM_TRAIN_EXAMPLES = 5000
NUM_VAL_EXAMPLES = 200
NUM_TEST_EXAMPLES = 200  # human evaluation 1000 * 5 = 5000 > 1000/500
DATASETS_CONFIG = {
    "summarization": {
        "cnn_dailymail:3.0.0": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "article",
            "output_key": "highlights",
            "instruction": "Summarize the news article.",
        },
        "xsum": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "document",
            "output_key": "summary",
            "instruction": "Generate an abstractive summary for the following article.",
        },
        "gigaword": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "document",
            "output_key": "summary",
            "instruction": "Generate a title for this article.",
        },
        "samsum": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "dialogue",
            "output_key": "summary",
            "instruction": "Generate a summary for the following dialogue.",
        },
    },
    "translation": {
        "wmt19:zh-en": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
            },
            "input_key": "zh",
            "output_key": "en",
            "instruction": "Translate the following Chinese text into English.",
        },
        "wmt19:de-en": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
            },
            "input_key": "en",
            "output_key": "de",
            "instruction": "Translate the following English text into German.",
        },
        "wmt19:cs-en": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
            },
            "input_key": "cs",
            "output_key": "en",
            "instruction": "Translate the following Czech text into English.",
        },
        "wmt19:ru-en": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
            },
            "input_key": "en",
            "output_key": "ru",
            "instruction": "Translate the following English text into Russian.",
        },
        "wmt19:fr-de": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
            },
            "input_key": "fr",
            "output_key": "de",
            "instruction": "Translate the following French text into German.",
        }
    },
    "data2text": {
        "dart": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "tripleset",
            "output_key": "annotations",
            "instruction": "Generate a description for the following tuples.",
        },
        "totto": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "tripleset",
            # The input is complex, see the python file for more details
            "output_key": "sentence_annotations",
            "instruction": "Generate a description for the following table.",
        },
        "kasnerz/wikitabletext": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "table",
            # The input is complex, see the python file for more details
            "output_key": "reference",
            "instruction": "Generate a description for the following table.",
        },
        "web_nlg:release_v2.1": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"dev[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "modified_triple_sets",
            # The input is complex, see the python file for more details
            "output_key": "lex",
            "instruction": "Generate a description for the following table.",
        }
    },
    "long-form QA": {
        "din0s/asqa": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"dev[:{NUM_VAL_EXAMPLES}]",
                "test": f"dev[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "ambiguous_question",
            "output_key": "annotations",
            "instruction": "Answer the following ambiguous factoid question by introducing additional knowledge, clarifying the relationship between multiple possible answers (if any) and resolving the ambiguity"
        },  # use llm to generate the answer
        "DongfuTingle/FeTaQA": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[:{NUM_TEST_EXAMPLES}]",
            },
            # special format
            "input_key": None,
            "output_key": "answer",
            "id_key": "feta_id",
            "instruction": "Given a table title and some highlighed cells (rows), answer the following question in several sentences with proper reasoning.",
        },  # using llm to generate the answer
        "cosmos_qa": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES//2}]",
                "finetune": f"train[{NUM_TRAIN_EXAMPLES//2}:]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": None,
            "output_key": None,
            "instruction": "Given the context below, answer the following question in 1 or 2 sentences.",
        },  # try fine-tuning on flan-t5 or using llm directly
        "eli5": {
            "split_info": {
                "train": f"train_eli5[:{NUM_TRAIN_EXAMPLES}]",
                "finetune": f"train_eli5[{NUM_TRAIN_EXAMPLES}:]",
                "validation": f"validation_eli5[:{NUM_VAL_EXAMPLES}]",
                "test": f"test_eli5[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": 'title',
            "output_key": 'answers',
            "instruction": "Answer the following question step by step in multiple sentences.",
        }  # try fine-tuning on flan-t5 or using llm directly
    },
    "mathQA": {
        "gsm8k:main": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"test[:{NUM_VAL_EXAMPLES}]",
                "test": "test[:]",
            },
            "input_key": "question",
            "output_key": "answer",
            "instruction": "Answer the following math word problem step-by-step.",
            # "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n",
        },  # use dslack/GSM8K-Flan-T5-Large
        # "math_qa": {
        #     "split_info": {
        #         "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
        #         "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
        #         "test": f"test[:{NUM_TEST_EXAMPLES}]",
        #     },
        #     "input_key": "Problem",
        #     "output_key": "Rationale",
        #     # "instruction": "Please, answer the following math problem step-by-step:",
        #     # "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n",
        #     "instruction": "Solve the following multiple choice math word problem step-by-step.",
        # }, # train our own, rationale
        # "qwedsacf/grade-school-math-instructions": {  # It's gsm8k with instructions
        #     "split_info": {
        #         "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
        #         "validation": f"train[:{NUM_VAL_EXAMPLES}]",
        #         "test": f"train[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
        #     }, # This one is special, only contains about 8k examples
        #     "input_key": "INSTRUCTION",
        #     "output_key": "RESPONSE",
        #     "instruction": "Please, answer the following math problem step-by-step:",
        # }, # use jordiclive/alpaca_gpt4-dolly_15k-vicuna-lora-7b.
        # "competition_math": { # TOO HARD
        #     "split_info": {
        #         "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
        #         "validation": f"test[:{NUM_VAL_EXAMPLES}]",
        #         "test": f"test[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
        #     },
        #     "input_key": "problem",
        #     "output_key": "solution",
        #     "instruction": "Please, answer the following math problem step-by-step:",
        # }, # try chavinlo/alpaca-native
    },
    "code": {
        "deepmind/code_contests": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"test[:{NUM_VAL_EXAMPLES}]",
                "test": f"test[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "description",
            "output_key": "solutions",
            "instruction": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:",
        },  # use dslack/GSM8K-Flan-T5-Large
    },
    "instruction-following": {
        "GAIR/lima": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": "test[:]",
                "test": "test[:]",
            },
            "input_key": "conversations",
            "output_key": None,
            "instruction": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        },
        "tatsu-lab/alpaca_farm:alpaca_instructions": {
            "split_info": {
                "train": f"sft[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"val[:{NUM_VAL_EXAMPLES}]",
            },
            "input_key": "instruction+input",  # special format
            "output_key": "output",
            "instruction": None,  # fill in the dataset instruction
        },
        "HuggingFaceH4/oasst1_en": {
            "split_info": {
                "train": f"train_ift[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"test_ift[:{NUM_VAL_EXAMPLES}]",
                "test": f"test_ift[:{NUM_TEST_EXAMPLES}]",
            },
            "input_key": "messages",  # special format
            "output_key": None,
            "instruction": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
        },
        "JosephusCheung/GuanacoDataset": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "instruction+input",  # special format
            "output_key": "output",
            "instruction": None,  # fill in the dataset instruction
        },
        "databricks/databricks-dolly-15k": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "instruction+input",  # special format
            "output_key": "output",
            "instruction": None,  # fill in the dataset instruction
        }
    },
    "story_generation": {
        "wza/roc_stories": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": None,
            "output_key": None,
            "instruction": "Generate a reasonable ending for the following story.",
        },  # special format
        "hellaswag": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                # hellaswag did not provide the label for the test set
                "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": None,
            "output_key": None,
            "instruction": "Generate a reasonable ending for the following story.",
            # startphrase is the first sentence. 4
        },  # special format. try flan-t5. it has been trained on hellaswag in a mutli-option setting. We could try if it works in the generation setting.
        # "swag": {
        #     "split_info": {
        #         "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
        #         "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
        #         "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
        #     },
        #     "input_key": None,
        #     "output_key": None,
        #     "instruction": "Generate a reasonable ending for the following story.",
        # }, # special format
    },  # I have a feel that these story generation task is too easy and the reference are not long answers. We can consider removing them.

    "other": {  # Need to fill
        "common_gen": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "concepts",
            "output_key": "target",
            "instruction": "Generate a reasonable sentence using the following concepts.",
        },  # special format, use flan-t5.
        # "lighteval/lsat_qa:all": {
        #     "split_info": {
        #         "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
        #         "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
        #         "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
        #     },
        #     "input_key": "concepts",
        #     "output_key": None,
        #     "instruction": "Answer the following question given the context carefully, in detail, in more than 50 words.",
        # },
        "vicgalle/alpaca-gpt4": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "concepts",
            "output_key": "output",
            "instruction": "",
        },
        "xnli:en": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "premise",
            "output_key": "hypothesis",
            "instruction": "Generate a reasonable hypothesis given the premise.",
        },
        "knkarthick/dialogsum": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"validation[:{NUM_VAL_EXAMPLES}]",
                "test": f"validation[{NUM_VAL_EXAMPLES}:{NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "input_key": "dialogue",
            "output_key": "summary",
            "instruction": "Summarize the dialogue.",
        },
    },  # do not use
    "image_captioning": {
        "conceptual_captions:labeled": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "id_key": "image_url",
            "input_key": "image_url",
            "output_key": "caption",
            "instruction": ""
        },
        "facebook/winoground": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "id_key": None,
            "input_key": "image_0",
            "output_key": "caption_0",
            "instruction": "",
        },
        "ChristophSchuhmann/MS_COCO_2017_URL_TEXT": {
            "split_info": {
                "train": f"train[:{NUM_TRAIN_EXAMPLES}]",
                "validation": f"train[{NUM_TRAIN_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}]",
                "test": f"train[{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES}:{NUM_TRAIN_EXAMPLES+NUM_VAL_EXAMPLES+NUM_TEST_EXAMPLES}]",
            },
            "id_key": None,
            "input_key": "URL",
            "output_key": "TEXT",
            "instruction": "",
        }
    },  # do not use
}

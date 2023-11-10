from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoModel,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)
decoder_only_models = ["alpaca", "llama", "opt", "bloom",
                       "gpt", "vicuna", "koala", "Wizard", "stablelm"]


def build_model(model_type, model_name, **kwargs):
    """
        Build the model from the model name
    """
    if any([x in model_type for x in decoder_only_models]) or any([x in model_name for x in decoder_only_models]):
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    elif model_type in ["vit"]:
        model = VisionEncoderDecoderModel.from_pretrained(model_name)
    elif model_type in ["bart", "t5", "mbart", "m2m100", "nllb", "opus_mt", "unifiedqa", "opus-mt", "pegasus"]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModel.from_pretrained(model_name, **kwargs)

    return model


def build_tokenizer(model_name, **kwargs):
    """
        Build the tokenizer from the model name
    """

    if "vicuna" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", use_fast=False, **kwargs)
    # elif "Wizard" in model_name:
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", return_token_type_ids=False, **kwargs)
    elif any([x in model_name for x in decoder_only_models]):
        # padding left
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left", **kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, **kwargs)  # , use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def build_processor(model_type, model_name, **kwargs):
    """
        Build the processor from the model name
    """
    if model_type in ["vit"]:
        processor = ViTImageProcessor.from_pretrained(model_name, **kwargs)
    else:
        raise NotImplementedError
    return processor

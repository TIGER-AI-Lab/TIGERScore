import os
import json
import json5
import logging
from dataclasses import dataclass
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Union
from itertools import chain
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class XPGTItem():
    task: str
    instruction: str
    input: str
    ref_output: Union[str, List[str]]
    hypo_output: str

# Message map functions


def default_msg_map(cur_message: dict, messages: List[dict]):
    """ Map the text and old messages to the new messages for query
    Args:
        text (str): the prompt text
        messages (List[dict]): the messages list before this query
    Returns:
        prompt (str): the prompt text
    """
    new_messages = messages + [{
        "role": cur_message['role'],
        "content": cur_message['content']}
    ]
    return new_messages

# Postprocess functions


def default_postprocess(content: str):
    return content


def json_postprocess(content: str):
    try:
        # find the json content
        json_content = content[content.find("{"):content.rfind("}") + 1]
        json_content = json.loads(json_content)
        return json_content
    except json.decoder.JSONDecodeError:
        try:
            json_content = json5.loads(json_content)
            return json_content
        except Exception:
            return content


tokenizer = None


def truncate_texts(texts: Union[List[str], List[List[str]]], max_length: int = None):
    """
    Truncate the texts to the max length.
    Args:
        texts (List[str] or List[List[str]]): The list of texts.
        max_length (int): The max length.
    Returns:
        List[str]: The truncated texts.
    """
    if max_length is None:
        return texts
    if isinstance(texts[0], list) and \
        (
            all([len(x) == 0 for x in texts]) or
            all([x is None for x in list(chain(*texts))])
    ) or isinstance(texts[0], str) and \
            all([x is None for x in list(chain(texts))]):
        logging.warning("All texts are None, skip truncating")
        return texts
    # using llama tokenizer by default
    global tokenizer
    disable_tqdm = len(texts) < 1000
    logging.warning(f"Truncating texts to max length {max_length}")
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", use_auth_token=True)
    # ...
    token_ids = []
    for text in tqdm(texts, desc="Truncating texts (tokenizing)", disable=disable_tqdm):
        if isinstance(text, list):
            token_ids.append(
                [tokenizer.encode(x, add_special_tokens=False) for x in text])
        else:
            token_ids.append(tokenizer.encode(text, add_special_tokens=False))
    # ...
    truncated_texts = []
    for i, _token_ids in tqdm(enumerate(token_ids), desc="Truncating texts (truncating)", disable=disable_tqdm):
        if (len(_token_ids)) and isinstance(_token_ids[0], list):
            truncated_texts.append([])
            for _token_id in _token_ids:
                if len(_token_id) > max_length:
                    truncated_text = tokenizer.decode(
                        _token_id[:max_length], skip_special_tokens=True)
                    truncated_text = truncated_text + " ..."
                else:
                    truncated_text = tokenizer.decode(
                        _token_id, skip_special_tokens=True)
                truncated_texts[i].append(truncated_text)
        else:
            if len(_token_ids) > max_length:
                truncated_text = tokenizer.decode(
                    _token_ids[:max_length], skip_special_tokens=True)
                truncated_text = truncated_text + " ..."
            else:
                truncated_text = tokenizer.decode(
                    _token_ids, skip_special_tokens=True)

            truncated_texts.append(truncated_text)
    return truncated_texts


def truncate_items(items: List[XPGTItem], max_lengths):
    """
    Truncate the texts in the items to the max length.
    Args:
        items (List[XPGTItem]): The list of items.
        max_length (int): The max length.
    Returns:
        List[XPGTItem]: The truncated items.
    """
    truncated_inputs = truncate_texts(
        [item.input for item in items], max_lengths.get("input", None))
    truncated_insts = truncate_texts(
        [item.instruction for item in items], max_lengths.get("instruction", None))
    truncated_ref_outputs = truncate_texts(
        [item.ref_output for item in items], max_lengths.get("ref_output", None))
    truncated_hypo_outputs = truncate_texts(
        [item.hypo_output for item in items], max_lengths.get("hypo_output", None))
    for i, item in enumerate(items):
        item.instruction = truncated_insts[i]
        item.input = truncated_inputs[i]
        item.ref_output = truncated_ref_outputs[i]
        item.hypo_output = truncated_hypo_outputs[i]
    return items


def get_query_messages(messages: List[dict], queried_messages: List[dict]):
    """
    Args:
        messages (List[dict]): the messages list to add for query
        queried_messages (List[dict]): the messages list already queried, which contains the query responses also,
    Returns:
        new_messages (List[dict]): the new messages list to query
        postprocess (function): the postprocess function for the query response
    """
    if len(queried_messages) == 0:
        last_prompt_idx = -1
    else:
        assert len(
            queried_messages) >= 2, "queried_messages should have at least 2 messages, i.e., the user (system) and the response"
        last_prompt = queried_messages[-2]['content']
        prompt_texts = [x['content'] for x in messages]
        last_prompt_idx = prompt_texts.index(last_prompt)
    if last_prompt_idx == len(messages) - 1:
        return None
    new_messages = queried_messages.copy()
    for idx in range(last_prompt_idx + 1, len(messages)):
        new_messages = messages[idx]["map_func"](messages[idx], new_messages)
        if messages[idx]["do_query"]:
            break
    return new_messages, messages[idx]["postprocess"]


def get_xgptscore_from_json(json_content: dict):
    """
    Args:
        json_content (dict): the json content
    Returns:
        xgptscore (float): the xgptscore, i.e. the sum of the reduction scores for all the errors
    """
    if isinstance(json_content, str):
        return None
    try:
        xgptscore = 0
        for error in json_content['errors'].values():
            if error['score_reduction'] == "N/A":
                continue
            xgptscore -= error['score_reduction']
        return xgptscore
    except Exception:
        return None


def get_xgptscore_from_json_star(json_content: dict):
    """
    Args:
        json_content (dict): the json content
    Returns:
        xgptscore (float): the xgptscore, i.e. the sum of the reduction scores for all the errors
    """
    xgptscore = 0
    res = {}
    for aspect_key, aspect in json_content.items():
        if isinstance(aspect, dict):
            score = aspect['Score']
            try:
                score = float(score)
            except Exception:
                score = 0
            xgptscore += score
            res["xgptscore_" + aspect_key] = score
    res["xgptscore"] = xgptscore
    return res


def get_xgptscore_from_json_per_aspect(json_content: dict):
    """
    Args:
        json_content (dict): the json content
    Returns:
        xgptscore (float): the xgptscore, i.e. the sum of the reduction scores for all the errors
    """
    if not isinstance(json_content, dict):
        return None
    xgptscore = 0
    res = {}
    for error in json_content['errors'].values():
        if error['error_aspect'] is not None:
            if ("xgptscore_" + error['error_aspect'] not in res):
                res["xgptscore_" + error['error_aspect']] = 0
            res["xgptscore_" + error['error_aspect']] -= error['score_reduction']
            xgptscore -= error['score_reduction']
    res["xgptscore"] = xgptscore
    return res

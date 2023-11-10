"""
This file contains the process functions for each chatgpt query mode.
Add your own process function for a mode here.
"""
import logging
import random
from .templates import *
from .process_utils import *
from .constants import *
from string import Template
MODE_TEMPLATE_MAP = {}
MODE_PROCESS_MAP = {}

# the decorator for registering a process function


def process_register(name=None):
    def register_func(func, name=name):
        if not name:
            # default name is the function name without the suffix "_process"
            name = func.__name__
            name = name.replace("_process", "")
        MODE_PROCESS_MAP[name] = func
        logging.info(f"Register process function: {name}")
        return func
    return register_func

# First, develop a process function for your mode.


@process_register()
@process_register(name="default")
def ea_process(item: XPGTItem):
    """
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(DEFAULT_SYSTEM_MESSAGE).substitute(task=item.task)
    prompt1 = Template(EA_TEMPLATE[0]).substitute(
        generation_instruction=item.instruction,
        input_context=item.input,
        reference_output=item.ref_output,
        hypothesis_output=item.hypo_output,
        task=item.task,
    )
    prompt2 = Template(EA_TEMPLATE[1]).substitute(
        aspect_descriptions="[{}]".format(
            ",".join(EVAL_ASPECTS[item.task].keys()))
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None  # no response to postprocess
    messages[1]['postprocess'] = default_postprocess
    messages[2]['postprocess'] = json_postprocess
    return messages


@process_register()
def old_ea_process(item: XPGTItem):
    """
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    prompt1 = Template(OLD_EA_TEMPLATES[0]).substitute(
        generation_instruction=item.instruction,
        input_context=item.input,
        reference_output=item.ref_output,
        hypothesis_output=item.hypo_output,
        task=item.task,
    )
    prompt2 = Template(OLD_EA_TEMPLATES[1]).substitute(
        aspect_descriptions="[{}]".format(
            ",".join(EVAL_ASPECTS[item.task].keys()))
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None  # no response to postprocess
    messages[1]['postprocess'] = default_postprocess
    messages[2]['postprocess'] = json_postprocess
    return messages


@process_register()
def wmt_mqm_process(item: XPGTItem):
    """
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(DEFAULT_SYSTEM_MESSAGE).substitute(task=item.task)
    if isinstance(item.ref_output, list) and len(item.ref_output) > 1:
        refs_str = "\n".join(["Reference Translation {}: {}".format(
            i+1, ref) for i, ref in enumerate(item.ref_output)])
    elif isinstance(item.ref_output, list) and len(item.ref_output) == 1:
        refs_str = "Reference Translation: " + item.ref_output[0]
    elif isinstance(item.ref_output, list) and len(item.ref_output) == 0:
        refs_str = "No Reference Translation Provided"
    elif isinstance(item.ref_output, str):
        refs_str = "Reference Translation: " + item.ref_output
    prompt1 = Template(WMT_MQM_TEMPLATES[0]).substitute(
        generation_instruction=item.instruction,
        input_context=item.input,
        reference_output=refs_str,
        hypothesis_output=item.hypo_output,
        task=item.task,
    )
    prompt2 = Template(WMT_MQM_TEMPLATES[1]).substitute(
        aspect_descriptions="[{}]".format(
            ",".join(EVAL_ASPECTS[item.task].keys()))
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None  # no response to postprocess
    messages[1]['postprocess'] = default_postprocess
    messages[2]['postprocess'] = json_postprocess
    return messages


@process_register()
def stars_process(item: XPGTItem):
    """
    A prompt template for Summarization Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(DEFAULT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(STARS_TEMPLATE[0]).substitute(
        input_context=item.input,
        reference_output=item.ref_output,
        hypothesis_output=item.hypo_output,
    )
    prompt1 = Template(STARS_TEMPLATE[1]).substitute(
        task=item.task,
        aspects_descriptions="\n".join(
            [f"- {k}: {v}" for k, v in EVAL_ASPECTS[item.task].items()]),
    )
    prompt2 = Template(STARS_TEMPLATE[2]).substitute(
        aspects_list=",".join(EVAL_ASPECTS[item.task].keys())
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def multi_aspects_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(MULTI_ASPECTS_TEMPLATE[0]).substitute(
        input_context=item.input,
        reference_output=zip_reference_string_non_task(item.ref_output),
        hypothesis_output=item.hypo_output,
    )
    prompt1 = Template(MULTI_ASPECTS_TEMPLATE[1]).substitute(
        task=item.task,
        # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]),
        aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
        # aspects_descriptions = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    prompt2 = Template(MULTI_ASPECTS_TEMPLATE[2]).substitute(
        aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
        # aspects_list = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def old_multi_aspects_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(OLD_MULTI_ASPECTS_TEMPLATE[0]).substitute(
        input_context=item.input,
        reference_output=item.ref_output,
        hypothesis_output=item.hypo_output,
    )
    prompt1 = Template(OLD_MULTI_ASPECTS_TEMPLATE[1]).substitute(
        task=item.task,
        # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]),
        aspects_descriptions=", ".join(
            ["\"" + i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    prompt2 = Template(OLD_MULTI_ASPECTS_TEMPLATE[2]).substitute(
        # aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
        aspects_list=", ".join(
            ["\"" + i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def one_shot_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    one_shot_prompt = Template(ONE_SHOT_TEMPLATE[0]).substitute(
        task=item.task,
        input_context=item.input,
        task_instruction=item.instruction,
    )
    io_prompt = Template(ONE_SHOT_TEMPLATE[1]).substitute(
        input_context=item.input,
        reference_output=item.ref_output,
        hypothesis_output=item.hypo_output,
    )
    prompt1 = Template(ONE_SHOT_TEMPLATE[2]).substitute(
        task=item.task,
        # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]),
        aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
    )
    prompt2 = Template(ONE_SHOT_TEMPLATE[3]).substitute(
        aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": one_shot_prompt, "do_query": True},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = default_postprocess
    messages[2]['postprocess'] = None
    messages[3]['postprocess'] = default_postprocess
    messages[4]['postprocess'] = json_postprocess
    return messages


def zip_reference_string_non_task(reference):
    if isinstance(reference, str):
        return "Reference: " + reference
    if isinstance(reference, list):
        reference = list(set(reference))
        return "\n".join(["Reference {}: {}".format(i + 1, ref) for i, ref in enumerate(reference)]).strip()
    raise TypeError("Reference is not a string or a list of strings")


def choose_only_one_reference(reference):
    if isinstance(reference, str):
        return reference
    if isinstance(reference, list):
        return sorted(list(set(reference)), key=len, reverse=True)[0]
    raise TypeError("Reference is not a string or a list of strings")


def d2t_task_instruction(instruction: str):
    return instruction.lower().replace("following", "Source").replace("below", "").strip(".").strip()


def zip_reference_string(reference, task: str):
    if isinstance(reference, str):
        return "Reference " + task + " Output: " + reference
    if isinstance(reference, list):
        return "\n".join(["Reference " + task + " Output {}: {}".format(i + 1, ref) for i, ref in enumerate(reference)]).strip()
    raise TypeError("Reference is not a string or a list of strings")

# @process_register()
# def summarization_process(item: XPGTItem):
#     """
#     A prompt template for Multi Aspects Quality Estimation
#     Args:
#         task: a task dict
#         inst (str): the instruction
#         input (str): the input context
#         ref_output (str): the reference output
#         hypo_output (str): the hypothesis output
#     Returns:
#         results (List[dict]):
#             List of prompt messages, each message contains the role and content, map_func and do_query.
#                 map_func: the function that map the current message content and the previous messages to the next message content
#                 do_query: whether to query the model for the current message
#     """
#     sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)

#     io_prompt = Template(SUMMARIZATION_TEMPLATE[0]).substitute(
#         input_context=item.input,
#         reference_output=zip_reference_string(item.ref_output,item.task),
#         hypothesis_output=item.hypo_output,
#         generation_instruction=item.instruction,
#     )
#     prompt1 = Template(SUMMARIZATION_TEMPLATE[1]).substitute(
#         task = item.task,
#         # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]),
#         aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
#         # aspects_descriptions = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
#     )
#     prompt2 = Template(SUMMARIZATION_TEMPLATE[2]).substitute(
#         aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
#         # aspects_list = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
#     )
#     messages = [
#         {"role": "system", "content": sys_prompt, "do_query": False},
#         {"role": "user", "content": io_prompt, "do_query": False},
#         {"role": "user", "content": prompt1, "do_query": True},
#         {"role": "user", "content": prompt2, "do_query": True},
#     ]
#     for msg in messages:
#         msg['map_func'] = default_msg_map
#     messages[0]['postprocess'] = None
#     messages[1]['postprocess'] = None
#     messages[2]['postprocess'] = default_postprocess
#     messages[3]['postprocess'] = json_postprocess
#     return messages


@process_register()
def align_score_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(ALIGN_SCORE_TEMPLATE[0]).substitute(
        input_context=item.input,
        reference_output=item.ref_output,
        hypothesis_output=item.hypo_output,
    )
    prompt1 = Template(ALIGN_SCORE_TEMPLATE[1]).substitute(
        task=item.task,
        aspects_descriptions="\n".join(
            [f"- {k}: {v}" for k, v in EVAL_ASPECTS[item.task].items()]).strip(),
        # aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
    )
    prompt2 = Template(ALIGN_SCORE_TEMPLATE[2]).substitute(
        aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
        # aspects_list = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def kb_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = "You are a helpful assistant to help user find information"
    aspects = EVAL_ASPECTS[item.task]
    shuffle_aspect_names = list(aspects.keys())
    random.shuffle(shuffle_aspect_names)
    selected_aspect_names = shuffle_aspect_names[:random.randint(
        1, len(aspects))]
    error_req_msg = ""

    total_num_major_errors = 0
    total_num_minor_errors = 0
    for aspect_name in selected_aspect_names:
        aspect_definition = aspects[aspect_name]
        num_major_errors = random.randint(1, 2)
        num_minor_errors = random.randint(0, 1)
        total_num_major_errors += num_major_errors
        total_num_minor_errors += num_minor_errors
        req_mag = "- contains {} major errors and {} minor errors for aspect {}".format(
            num_major_errors, num_minor_errors, aspect_name
        )
        error_req_msg += req_mag + "\n"
    error_req_msg += "Thus the total number of major errors is {} and the total number of minor errors is {} for all the error aspects above".format(
        total_num_major_errors, total_num_minor_errors
    )

    if isinstance(item.ref_output, list):
        ref_output = random.choice(item.ref_output)
    elif isinstance(item.ref_output, str):
        ref_output = item.ref_output
    else:
        raise TypeError(
            "Reference output is not a string or a list of strings")
    io_prompt = Template(KB_TEMPLATES[0]).substitute(
        generation_instruction=item.instruction,
        input_context=item.input,
        reference_output=ref_output,
        error_requirements=error_req_msg,
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = json_postprocess
    return messages


@process_register()
def d2t_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(D2T_TEMPLATE[0]).substitute(
        input_context=item.input,
        reference_output=zip_reference_string_non_task(item.ref_output),
        hypothesis_output=item.hypo_output,
        generation_instruction=d2t_task_instruction(item.instruction),
    )
    prompt1 = Template(D2T_TEMPLATE[1]).substitute(
        task=item.task,
        generation_instruction=d2t_task_instruction(item.instruction),
        # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]).strip(),
        aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
    )
    prompt2 = Template(D2T_TEMPLATE[2]).substitute(
        aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
        # aspects_list = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def kb_txt_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = "You are a helpful assistant to help user find information"
    aspects = EVAL_ASPECTS[item.task]
    shuffle_aspect_names = list(aspects.keys())
    random.shuffle(shuffle_aspect_names)
    selected_aspect_names = shuffle_aspect_names[:random.randint(
        1, len(aspects))]
    error_req_msg = ""

    for aspect_name in selected_aspect_names:
        aspect_definition = aspects[aspect_name]
        req_mag = "- contains at least 1 error for aspect {}: {}".format(
            aspect_name, aspect_definition
        )
        error_req_msg += req_mag + "\n"

    if isinstance(item.ref_output, list):
        ref_output = random.choice(item.ref_output)
    elif isinstance(item.ref_output, str):
        ref_output = item.ref_output
    else:
        raise TypeError(
            "Reference output is not a string or a list of strings")
    io_prompt = Template(KB_TXT_TEMPLATES[0]).substitute(
        generation_instruction=item.instruction,
        input_context=item.input,
        reference_output=ref_output,
        error_requirements=error_req_msg,
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = default_postprocess
    return messages


@process_register()
def instruction_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """
    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(INSTRUCTION_TEMPLATE[0]).substitute(
        input_context=item.input,
        generation_instruction=item.instruction,
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = default_postprocess
    return messages


def joint_instruction_and_source(input_context: str, generation_instruction: str):
    if (not input_context) or input_context.strip() == "":
        return "Source:" + generation_instruction
    else:
        return "Task Instruction:" + generation_instruction + "\nSource:" + input_context


@process_register()
def longform_qa_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """

    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(LONGFORM_QA_TEMPLATE[0]).substitute(
        generation_instruction_and_source=joint_instruction_and_source(
            item.input, item.instruction),
        input_context=item.input,
        reference_output=zip_reference_string_non_task(item.ref_output),
        hypothesis_output=item.hypo_output,
        generation_instruction=item.instruction,
    )
    prompt1 = Template(LONGFORM_QA_TEMPLATE[1]).substitute(
        task=item.task,
        generation_instruction=d2t_task_instruction(item.instruction),
        # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]).strip(),
        aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
    )
    prompt2 = Template(LONGFORM_QA_TEMPLATE[2]).substitute(
        aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
        # aspects_list = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def instruction_following_process(item: XPGTItem):
    """
    A prompt template for Multi Aspects Quality Estimation 
    Args:
        task: a task dict
        inst (str): the instruction
        input (str): the input context
        ref_output (str): the reference output
        hypo_output (str): the hypothesis output
    Returns:
        results (List[dict]):
            List of prompt messages, each message contains the role and content, map_func and do_query.
                map_func: the function that map the current message content and the previous messages to the next message content
                do_query: whether to query the model for the current message
    """

    sys_prompt = Template(CHATGPT_SYSTEM_MESSAGE).substitute(task=item.task)
    io_prompt = Template(INSTRUCTION_FOLLOWING_TEMPLATE[0]).substitute(
        generation_instruction_and_source=joint_instruction_and_source(
            item.input, item.instruction),
        input_context=item.input,
        reference_output=zip_reference_string_non_task(item.ref_output),
        hypothesis_output=item.hypo_output,
        generation_instruction=item.instruction,
    )
    prompt1 = Template(INSTRUCTION_FOLLOWING_TEMPLATE[1]).substitute(
        task=item.task,
        generation_instruction=d2t_task_instruction(item.instruction),
        # aspects_descriptions="\n".join([f"- {k}: {v}" for k,v in EVAL_ASPECTS[item.task].items()]).strip(),
        aspects_descriptions=", ".join(EVAL_ASPECTS[item.task].keys()),
    )
    prompt2 = Template(INSTRUCTION_FOLLOWING_TEMPLATE[2]).substitute(
        aspects_list=", ".join(EVAL_ASPECTS[item.task].keys())
        # aspects_list = ", ".join(["\""+ i + "\"" for i in EVAL_ASPECTS[item.task].keys()]),
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = None
    messages[2]['postprocess'] = default_postprocess
    messages[3]['postprocess'] = json_postprocess
    return messages


@process_register()
def paraphrase_process(item: XPGTItem):
    sys_prompt = "You are a helpful assistant to help user find information"

    if isinstance(item.ref_output, list):
        ref_output = random.choice(item.ref_output)
    elif isinstance(item.ref_output, str):
        ref_output = item.ref_output
    else:
        raise TypeError(
            "Reference output is not a string or a list of strings")
    io_prompt = Template(PARAPHRASE_TEMPLATES[0]).substitute(
        generation_instruction=item.instruction,
        input_context=item.input,
        reference_output=ref_output,
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": io_prompt, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = default_postprocess
    return messages


@process_register()
def mathqa_process(item: XPGTItem):
    sys_prompt = "You are a helpful assistant to help user find information"

    if isinstance(item.ref_output, list):
        ref_output = random.choice(item.ref_output)
    elif isinstance(item.ref_output, str):
        ref_output = item.ref_output
    else:
        raise TypeError(
            "Reference output is not a string or a list of strings")

    prompt1 = Template(MATHQA_TEMPLATES[0]).substitute(
        task=item.task,
        generation_instruction=item.instruction.strip("\n "),
        input_context=item.input.strip("\n "),
        reference_output=ref_output.strip("\n "),
        hypothesis_output=item.hypo_output.strip("\n "),
        aspects_list="\n".join(
            [f"{i+1}. {aspect_name}" for i, aspect_name in enumerate(EVAL_ASPECTS[item.task].keys())]),
    )
    prompt2 = Template(MATHQA_TEMPLATES[1]).substitute(
        aspects_list="[{}]".format(
            ", ".join([f'"{aspect_name}"' for i, aspect_name in enumerate(
                EVAL_ASPECTS[item.task].keys())])
        )
    )
    messages = [
        {"role": "system", "content": sys_prompt, "do_query": False},
        {"role": "user", "content": prompt1, "do_query": True},
        {"role": "user", "content": prompt2, "do_query": True},
    ]
    for msg in messages:
        msg['map_func'] = default_msg_map
    messages[0]['postprocess'] = None
    messages[1]['postprocess'] = default_postprocess
    messages[2]['postprocess'] = json_postprocess
    return messages

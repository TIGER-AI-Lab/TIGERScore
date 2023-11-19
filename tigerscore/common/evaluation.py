
import psutil
import os
import numpy as np
import spacy
import torch
from copy import deepcopy
from evaluate import load
from sacrebleu import sentence_bleu
from nltk import word_tokenize
from typing import List, Union, Dict
from absl import logging
from tqdm import tqdm
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from tqdm.contrib.concurrent import process_map
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu
from collections import defaultdict

logging.set_verbosity(logging.WARNING)


SUPPORTED_METRICS = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bleu', 'bleurt', "cider", "spice",
                     "bleu4", "bertscore", "prism", "comet", "bart_score", "bart_score_cnn", "bart_score_para", "chrf",
                     "chatgpt_zero_shot", "gpt4_zero_shot"]
METRIC_WEIGHTS = {
    "rouge1": 1.0,
    "rouge2": 1.0,
    "rougeL": 1.0,
    "rougeLsum": 1.0,
    "bleu": 0.01,
    "bleu4": 0.01,
    "bleurt": 1.0,
    "cider": 0.01,
    "spice": 0.01,
    "bertscore": 1.0,
    "chrf": 1.0,
    "prism": 1.0,
    "comet_da": 0.01,
    "cometkiwi_da": 0.01,
    "unieval_sum": 0.01,
    "unieval_dialogue": 0.01,
    "unieval_data2text": 0.01,
    "unieval_fact": 0.01,
    "novel_unigram": 1.0,
    "bart_score": 1.0,
    "bart_score_cnn": 1.0,
    "bart_score_para": 1.0,
    "chatgpt_zero_shot": 1.0,
    "gpt4_zero_shot": 1.0,
}  # scale to 0-1


def pre_rouge_processing(summary):
    summary = summary.replace("<n>", " ")
    summary = "\n".join(sent_tokenize(summary))
    return summary


def eval_rouge(
    hypotheses: List[List[str]],
    references: List[List[str]],
    rouge_types: List[str] = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
) -> Dict[str, float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
        rouge_types: the rouge types to be used.

    Returns:
        A dict of rouge scores.
        key is the rouge type, value is the rouge score, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(references)
    assert set(rouge_types) <= set(['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
                                   ), "Rouge types should be in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']"
    scorer = rouge_scorer.RougeScorer(
        rouge_types, use_stemmer=True, split_summaries=True)
    rouge_scores = {rouge_type: [[] for _ in range(
        len(hypotheses))] for rouge_type in rouge_types}
    with tqdm(total=len(hypotheses), desc="Evaluating rouge") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            for hypo in hypo_group:
                scores = scorer.score_multi(ref, pre_rouge_processing(hypo))
                for rouge_type in rouge_types:
                    rouge_scores[rouge_type][i].append(
                        scores.get(rouge_type).fmeasure)
            pbar.update(1)
    return rouge_scores


def eval_bleu(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    assert len(hypotheses) == len(
        references), f"Length of hypotheses {len(hypotheses)} and references {len(references)} should be the same."
    bleu_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleu") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleu_scores.append([])
            for hypo in hypo_group:
                bleu_scores[i].append(sentence_bleu(hypo, ref).score)
            pbar.update(1)
    return bleu_scores


def eval_bleurt(
    hypotheses: List[List[str]],
    references: List[List[str]]
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    assert len(hypotheses) == len(references)
    bleurt_scorer = load('bleurt')
    bleurt_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bleurt") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            bleurt_scores.append([])
            for hypo in hypo_group:
                _t_scores = []
                for _ref in ref:
                    result = bleurt_scorer.compute(
                        predictions=[hypo], references=[_ref])
                    _t_scores.append(result['scores'][0])
                bleurt_scores[i].append(max(_t_scores))
            pbar.update(1)
    return bleurt_scores


def eval_bleu4(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references

    Returns:
        A list of bleu scores, in same shape with hypotheses.
    """
    print("Evaluating bleu4")
    assert len(hypotheses) == len(references)
    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join(
                [token.text for token in nlp(hypotheses[i][j])])
        for j in range(len(references[i])):
            references[i][j] = " ".join(
                [token.text for token in nlp(references[i][j])])

    bleu4_scorer = Bleu(4)
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = ref
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = bleu4_scorer.compute_score(gts, res)
    for method in zip(("Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"), score):
        print("%s: %0.3f" % method)
    bleu4_scores = scores[3]
    bleu4_scores = [[bleu4_scores[hypo_id] * 100 for hypo_id in hypo_ids]
                    for hypo_ids in hypo_ids_per_ref]
    return bleu4_scores


def eval_chrf(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating chrf")
    assert len(hypotheses) == len(references)
    from sacrebleu import sentence_chrf
    chrf_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating chrf") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            chrf_scores.append([])
            for hypo in hypo_group:
                chrf_scores[i].append(sentence_chrf(hypo, ref).score)
            pbar.update(1)
    return chrf_scores


def eval_cider(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating cider")
    assert len(hypotheses) == len(references)

    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join(
                [token.text for token in nlp(hypotheses[i][j])])
        for j in range(len(references[i])):
            references[i][j] = " ".join(
                [token.text for token in nlp(references[i][j])])

    cider_scorer = Cider()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0

    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = ref
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = cider_scorer.compute_score(gts, res)
    cider_scores = [[scores[hypo_id] * 10 for hypo_id in hypo_ids]
                    for hypo_ids in hypo_ids_per_ref]
    return cider_scores


def eval_spice(
    hypotheses: List[List[str]],
    references: List[List[str]]
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    print("Evaluating spice")
    assert len(hypotheses) == len(references)
    # tokenization
    nlp = spacy.load("en_core_web_sm")
    disable_pipes = list(nlp.pipe_names)
    disable_pipes.remove('tagger')
    nlp.disable_pipes(*disable_pipes)
    for i in tqdm(range(len(hypotheses)), desc="Tokenizing"):
        for j in range(len(hypotheses[i])):
            hypotheses[i][j] = " ".join(
                [token.text for token in nlp(hypotheses[i][j])])
        for j in range(len(references[i])):
            references[i][j] = " ".join(
                [token.text for token in nlp(references[i][j])])

    spice_scorer = Spice()
    gts = {}
    res = {}
    hypo_ids_per_ref = []
    id = 0
    for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
        hypo_ids_per_ref.append([])
        for hypo in hypo_group:
            gts[id] = ref
            res[id] = [hypo]
            hypo_ids_per_ref[i].append(id)
            id += 1

    score, scores = spice_scorer.compute_score(gts, res)
    spice_scores = [[scores[hypo_id]['All']['f'] * 100.0 for hypo_id in hypo_ids]
                    for hypo_ids in hypo_ids_per_ref]
    return spice_scores


def eval_prism(
    hypotheses: List[List[str]],
    references: List[List[str]],
) -> List[float]:
    # We do not src
    from prism import Prism
    from pathlib import Path
    model_path = str(Path(os.path.abspath(__file__)
                          ).parent / Path("models/m39v1"))
    prism = Prism(model_dir=model_path, lang='en', temperature=1.0)
    prism_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating prism") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            # prism_scores.append([])
            # for hypo in hypo_group:
            prism_scores.append(
                list(prism.score(hypo_group, ref, segment_scores=True)[-1]))
            pbar.update(1)
    return prism_scores


def eval_comet_da(
    hypotheses: List[List[str]],
    references: List[List[str]],
    sources: List[str],
) -> List[float]:
    from comet import download_model, load_from_checkpoint
    model_path = download_model('Unbabel/wmt22-comet-da')
    model = load_from_checkpoint(model_path)
    samples = []
    assert len(hypotheses) == len(references) == len(sources)
    for i, (hypo_group, ref_group, source) in enumerate(zip(hypotheses, references, sources)):
        for hypo in hypo_group:
            samples.append({
                "src": source,
                "ref": ref_group,
                "mt": hypo,
            })
    batch_size = 8
    res = model.predict(samples, gpus=1, batch_size=batch_size)
    comets = res[0]
    wrapped_comets = []
    idx = 0
    for i in range(len(hypotheses)):
        wrapped_comets.append([])
        for j in range(len(hypotheses[i])):
            wrapped_comets[i].append(comets[idx])
            idx += 1
    return wrapped_comets


def eval_cometkiwi_da(
    hypotheses: List[List[str]],
    sources: List[str],
) -> List[float]:
    from comet import download_model, load_from_checkpoint
    model_path = download_model('Unbabel/wmt22-cometkiwi-da')
    model = load_from_checkpoint(model_path)
    samples = []
    assert len(hypotheses) == len(sources)
    for i, (hypo_group, source) in enumerate(zip(hypotheses, sources)):
        for hypo in zip(hypo_group):
            samples.append({
                "src": source,
                "mt": hypo,
            })
    batch_size = 8
    res = model.predict(samples, gpus=1, batch_size=batch_size)
    comets = res[0]
    wrapped_comets = []
    idx = 0
    for i in range(len(hypotheses)):
        wrapped_comets.append([])
        for j in range(len(hypotheses[i])):
            wrapped_comets[i].append(comets[idx])
            idx += 1
    return wrapped_comets


def eval_bertscore(
    hypotheses: List[List[str]],
    references: List[List[str]],
    model_type="bert-base-multilingual-cased",
    lang="en",
) -> List[float]:
    """
    Evaluate the hypothesis and reference using bertscore.
    BertScore officially recommends using microsoft/deberta-xlarge-mnli as the model.
    the default multilingual model is bert-base-multilingual-cased.

    Args:
        hypotheses: the hypotheses
        references: the references
    """
    import bert_score
    print("Evaluating bertscore")
    assert len(hypotheses) == len(references)
    flatten_hypotheses = []
    flatten_references = []
    for i in range(len(hypotheses)):
        flatten_hypotheses.extend(hypotheses[i])
        flatten_references.extend([references[i]] * len(hypotheses[i]))

    P, R, F1 = bert_score.score(flatten_hypotheses, flatten_references,
                                lang=lang, verbose=True, model_type=model_type, batch_size=16)
    flatten_scores = F1.numpy().tolist()
    scores = []
    idx = 0
    for i in range(len(hypotheses)):
        scores.append([])
        for j in range(len(hypotheses[i])):
            scores[i].append(flatten_scores[idx])
            idx += 1
    return scores


def eval_bartscore(
    hypotheses: List[List[str]],
    references: List[List[str]],
    metric_name="bart_score",
) -> List[float]:
    """
    Evaluate the hypothesis and reference using the metric.

    Args:
        hypotheses: the hypotheses
        references: the references
        metric_name: bart_score or bart_score_cnn or bart_score_para
    """
    assert len(hypotheses) == len(references)
    from bart_score import BARTScorer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if metric_name == "bart_score_cnn":
        bart_scorer = BARTScorer(
            device=device, checkpoint='facebook/bart-large-cnn')
    elif metric_name == "bart_score_para":
        bart_scorer = BARTScorer(
            device=device, checkpoint='facebook/bart-large-cnn')
        from pathlib import Path
        model_path = Path(os.path.abspath(__file__)).parent / \
            Path("models/bart_score_para.pth")
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading bart_score_para model")
            os.system(
                "gdown https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m -O %s" % model_path)
        else:
            print("bart_score_para model exists")
        bart_scorer.load(model_path)
    elif metric_name == "bart_score":
        bart_scorer = BARTScorer(
            device=device, checkpoint='facebook/bart-large')
    else:
        raise ValueError(
            "metric_name should be bart_score or bart_score_cnn or bart_score_para")

    bart_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bartscore") as pbar:
        for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
            _temp_scores = []
            if not isinstance(ref, list):
                ref = [ref]
            for ref_i in ref:
                hypo_ref_scores = np.array(bart_scorer.score(
                    hypo_group, [ref_i] * len(hypo_group), batch_size=4))
                ref_hypo_scores = np.array(bart_scorer.score(
                    [ref_i] * len(hypo_group), hypo_group, batch_size=4))
                _temp_scores.append(
                    ((hypo_ref_scores + ref_hypo_scores) / 2).tolist())
                # _temp_scores.append(hypo_ref_scores.tolist())
            # get max ref score
            _temp_scores = np.array(_temp_scores).max(axis=0)
            bart_scores.append(_temp_scores)
            pbar.update(1)
            assert len(bart_scores[i]) == len(hypo_group)
    return bart_scores


def eval_bartscore_src_hypo(
    hypotheses: List[List[str]],
    sources: List[str],
    metric_name="bart_score",
) -> List[float]:
    """
    Evaluate the hypothesis with source using the metric.

    Args:
        hypotheses: the hypotheses
        sources: the sources
        metric_name: bart_score or bart_score_cnn or bart_score_para
    """
    assert len(hypotheses) == len(sources)
    from bart_score import BARTScorer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if metric_name == "bart_score_cnn":
        bart_scorer = BARTScorer(
            device=device, checkpoint='facebook/bart-large-cnn')
    elif metric_name == "bart_score_para":
        bart_scorer = BARTScorer(
            device=device, checkpoint='facebook/bart-large-cnn')
        from pathlib import Path
        model_path = Path(os.path.abspath(__file__)).parent / \
            Path("models/bart_score_para.pth")
        if not model_path.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading bart_score_para model")
            os.system(
                "gdown https://drive.google.com/uc?id=1_7JfF7KOInb7ZrxKHIigTMR4ChVET01m -O %s" % model_path)
        else:
            print("bart_score_para model exists")
        bart_scorer.load(model_path)
    else:
        bart_scorer = BARTScorer(
            device=device, checkpoint='facebook/bart-large')

    bart_scores = []
    with tqdm(total=len(hypotheses), desc="Evaluating bartscore") as pbar:
        for i, (hypo_group, source) in enumerate(zip(hypotheses, sources)):
            src_hypo_scores = bart_scorer.score(
                [source] * len(hypo_group), hypo_group, batch_size=4)
            bart_scores.append(src_hypo_scores)
            pbar.update(1)
            assert len(bart_scores[i]) == len(hypo_group)
    return bart_scores


def eval_unieval(
    hypotheses: List[List[str]],
    references: List[List[str]],
    sources: List[str],
    task: str,
):
    from unieval.utils import convert_to_json
    from unieval.metric.evaluator import get_evaluator
    assert task in ['summarization', 'dialogue', 'data2text', 'fact']
    if isinstance(hypotheses[0], list):
        # flatten
        flatten_hypotheses = []
        flatten_references = [] if references is not None else None
        flatten_sources = [] if sources is not None else None
        for i in range(len(hypotheses)):
            for hypo in hypotheses[i]:
                if hypo:
                    flatten_hypotheses.append(hypo)
                    if references is not None:
                        if isinstance(references[i], list):
                            flatten_references.append(references[i][0])
                        elif isinstance(references[i], str):
                            flatten_references.append(references[i])
                        else:
                            raise ValueError(
                                "references should be a list of list of str or a list of str")
                    if sources is not None:
                        flatten_sources.append(sources[i])
        data = convert_to_json(output_list=flatten_hypotheses,
                               src_list=flatten_sources, ref_list=flatten_references)
    elif isinstance(hypotheses[0], str):
        output_list = [x for x in hypotheses if x]
        src_list = [x for i, x in enumerate(sources) if hypotheses[i]]
        ref_list = [x for i, x in enumerate(references) if hypotheses[i]]
        data = convert_to_json(output_list=output_list,
                               src_list=src_list, ref_list=ref_list)
    else:
        raise ValueError(
            "hypotheses should be a list of list of str or a list of str")
    # compute
    evaluator = get_evaluator(task)
    eval_scores = evaluator.evaluate(data, print_result=True)
    aspect_scores = defaultdict(list)
    # unflatten
    aspects = list(eval_scores[0].keys())
    if isinstance(hypotheses[0], list):
        idx = 0
        for i, hypo_group in enumerate(hypotheses):
            for aspect in aspects:
                aspect_scores[aspect].append([])
            for hypo in hypo_group:
                if hypo:
                    for aspect in aspects:
                        aspect_scores[aspect][i].append(
                            eval_scores[idx][aspect])
                    idx += 1
                else:
                    for aspect in aspects:
                        aspect_scores[aspect][i].append(0)
    else:
        for aspect in aspects:
            idx = 0
            for hypo in hypotheses:
                if hypo:
                    aspect_scores[aspect].append(eval_scores[idx][aspect])
                    idx += 1
                else:
                    aspect_scores[aspect].append(0)
    return aspect_scores


def eval_instructscore(
    hypotheses: List[List[str]],
    references: List[List[str]],
):
    from InstructScore import InstructScore
    scorer = InstructScore()
    assert len(hypotheses) == len(references)
    flatten_hypotheses = []
    flatten_references = []
    for i in range(len(hypotheses)):
        flatten_hypotheses.extend(hypotheses[i])
        if isinstance(references[i], list):
            flatten_references.extend([references[i][0]] * len(hypotheses[i]))
        elif isinstance(references[i], str):
            flatten_references.extend([references[i]] * len(hypotheses[i]))
        else:
            raise ValueError(
                "references should be a list of list of str or a list of str")
    batch_outputs, scores_ls = scorer.score(
        flatten_references, flatten_hypotheses)
    idx = 0
    instruct_scores = []
    for i in range(len(hypotheses)):
        instruct_scores.append([])
        for j in range(len(hypotheses[i])):
            instruct_scores[i].append(scores_ls[idx])
            idx += 1
    return instruct_scores


def eval_gptscore_ref_ist(
    hypotheses: List[List[str]],
    references: List[List[str]],
    task="summarization",
    metric_name="flan_base_score",
):
    """
    Instruction (IST) only version gptscore
    """
    asp_definitions = {
        "Semantic Coverage (COV)": "How many semantic content units from the reference text are covered by the generated text?",
        "Factuality (FAC)": "Does the generated text preserve the factual statements of the source text?",
        "Consistency (CON)": "Is the generated text consistent in the information it provides?",
        "Informativeness (INF)": "How well does the generated text capture the key ideas of its source text?",
        "Coherence (COH)": "How much does the generated text make sense?",
        "Relevance (REL)": "How well is the generated text relevant to its source text?",
        "Fluency (FLU)": "Is the generated text well-written and grammatical?",
        "Accuracy (ACC)": "Are there inaccuracies, missing, or unfactual content in the generated text?",
        "Multidimensional Quality Metrics (MQM)": "How is the overall quality of the generated text?",
        "Interest (INT)": "Is the generated text interesting?",
        "Engagement (ENG)": "Is the generated text engaging?",
        "Specific (SPE)": "Is the generated text generic or specific to the source text?",
        "Correctness (COR)": "Is the generated text correct or was there a misunderstanding of the source text?",
        "Semantically appropriate (SEM)": "Is the generated text semantically appropriate?",
        "Understandability (UND)": "Is the generated text understandable?",
        "Error Recovery (ERR)": "Is the system able to recover from errors that it makes?",
        "Diversity (DIV)": "Is there diversity in the system responses?",
        "Depth (DEP)": "Does the system discuss topics in depth?",
        "Likeability (LIK)": "Does the system display a likeable personality?",
        "Flexibility (FLE)": "Is the system flexible and adaptable to the user and their interests?",
        "Inquisitiveness (INQ)": "Is the system inquisitive throughout the conversation?",
    }
    asp_names = list(asp_definitions.keys())
    task_asp_map = {
        "Summ": asp_names[:7],
        "D2T": [asp_names[3]] + asp_names[5:7],
        "Dial": asp_names[2:4] + asp_names[5:7] + asp_names[9:],
        "MT": asp_names[6:9],
    }
    assert task in task_asp_map.keys()
    final_gpt_scores = {}
    from mosestokenizer import MosesDetokenizer
    detokenizer = MosesDetokenizer('en')

    def add_dot(text):
        if len(text.strip()) == 0:
            return '.'
        if text.strip()[-1] != '.':
            text = text.strip() + ' .'
        new_text = text
        return new_text

    def detokenize(text: str):
        # words = text.split(" ")
        words = text.split()
        return detokenizer(words)
    if "flan" in metric_name:
        from flan_score import FLANScorer
        import time
        metric2checkpoint = {
            "flan_small_score": "google/flan-t5-small",
            "flan_base_score": "google/flan-t5-base",
            "flan_large_score": "google/flan-t5-large",
            "flan_xl_score": "google/flan-t5-xl",
            "flan_xxl_score": "google/flan-t5-xxl",
        }
        print('metric_name: ', metric_name)
        checkpoint = metric2checkpoint[metric_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        flan_scorer = FLANScorer(device=device, checkpoint=checkpoint)
        print('FLANScorer setup finished. Begin calculating FLANScorer.')
        start = time.time()
        for e_asp in task_asp_map[task]:
            print("evaluting ", e_asp)
            # Evaluation is cheap when using non-GPT3 models, so here, we evaluate all aspects by default.
            print('num of examples: ', len(hypotheses))
            asp_df = asp_definitions[e_asp]
            asp_df = asp_df.strip().replace(':', '. ')
            print('asp_df: ', asp_df)
            # perfix
            prefix = asp_df + '\n'
            template = "XXXXX In other words , YYYYY"
            # ref->hypo
            gpt_scores = []

            with tqdm(total=len(hypotheses), desc="Evaluating gptscore (Flan)") as pbar:
                for i, (hypo_group, ref) in enumerate(zip(hypotheses, references)):
                    _temp_scores = []
                    if not isinstance(ref, list):
                        ref = [ref]
                    for ref_i in ref:
                        hyps = [add_dot(detokenize(hypo))
                                for hypo in hypo_group]
                        ref_i_ = add_dot(detokenize(ref_i))
                        hypo_ref_scores = np.array(flan_scorer.score(
                            [prefix + template.replace('XXXXX', x).replace('YYYYY', '')
                             for x in hyps],
                            [ref_i_] * len(hypo_group), batch_size=4)
                        )
                        ref_hypo_scores = np.array(flan_scorer.score(
                            [prefix + template.replace('XXXXX', ref_i_).replace('YYYYY', '')] * len(hypo_group), hyps, batch_size=4))
                        _temp_scores.append(
                            ((hypo_ref_scores + ref_hypo_scores) / 2).tolist())
                        # _temp_scores.append(hypo_ref_scores.tolist())
                    # get max ref score
                    _temp_scores = np.array(_temp_scores).max(axis=0)
                    gpt_scores.append(_temp_scores)
                    pbar.update(1)
                    assert len(gpt_scores[i]) == len(hypo_group)
            final_gpt_scores[e_asp] = gpt_scores
        end = time.time()
        print(f'FLANScorer finished. Time: {end-start}')
    else:
        raise NotImplementedError
    # final_gpt_scores["avg_all_asps"] = np.mean([np.array(final_gpt_scores[asp]) for asp in task_asp_map[task]], axis=0).tolist()
    scores = [final_gpt_scores[asp] for asp in task_asp_map[task]]
    avg_scores = [sum(x) / len(x) for x in zip(*scores)]
    final_gpt_scores["avg_all_asps"] = avg_scores
    return final_gpt_scores

def eval_GPT_zero_shot(
    hypotheses: List[List[str]],
    references: List[List[str]],
    sources: List[str],
    model_name="ChatGPT",
):
    from xgptscore.openai_utils import openai_completions, _chatml_to_prompt
    def get_zero_shot_prompt(source, hypo):
        _prompt = "Instruction: \n{}\n".format(source)
        _prompt += "Model-generated Output: \n{}\n".format(hypo)
        _prompt += "Rate the quality of the above output on a scale of 1-5 (Answer me the score only):\n"
        message = [{"role": "user", "content": _prompt}]
        _prompt = _chatml_to_prompt(message)
        return _prompt
    assert len(hypotheses) == len(references) == len(sources), "length of hypotheses, references and sources should be the same"
    scores = []
    prompts = [get_zero_shot_prompt(source, hypo) for hypo_group, source in zip(hypotheses, sources) for hypo in hypo_group]
    completions = openai_completions(prompts, model_name=model_name)
    scores = [float(c) for c in completions['completions']]
    if isinstance(hypotheses[0], list):
        # wrap scores to the original shape
        idx = 0
        new_scores = []
        for hypo_group in hypotheses:
            new_scores.append(scores[idx:idx+len(hypo_group)])
            idx += len(hypo_group)
        scores = new_scores
    return scores

def eval_gptscore_src_ist(
    hypotheses: List[List[str]],
    sources: List[str],
    task="summarization",
    metric_name="flan_base_score",
):
    """
    Instruction (IST) only version gptscore
    """
    asp_definitions = {
        "Semantic Coverage (COV)": "How many semantic content units from the reference text are covered by the generated text?",
        "Factuality (FAC)": "Does the generated text preserve the factual statements of the source text?",
        "Consistency (CON)": "Is the generated text consistent in the information it provides?",
        "Informativeness (INF)": "How well does the generated text capture the key ideas of its source text?",
        "Coherence (COH)": "How much does the generated text make sense?",
        "Relevance (REL)": "How well is the generated text relevant to its source text?",
        "Fluency (FLU)": "Is the generated text well-written and grammatical?",
        "Accuracy (ACC)": "Are there inaccuracies, missing, or unfactual content in the generated text?",
        "Multidimensional Quality Metrics (MQM)": "How is the overall quality of the generated text?",
        "Interest (INT)": "Is the generated text interesting?",
        "Engagement (ENG)": "Is the generated text engaging?",
        "Specific (SPE)": "Is the generated text generic or specific to the source text?",
        "Correctness (COR)": "Is the generated text correct or was there a misunderstanding of the source text?",
        "Semantically appropriate (SEM)": "Is the generated text semantically appropriate?",
        "Understandability (UND)": "Is the generated text understandable?",
        "Error Recovery (ERR)": "Is the system able to recover from errors that it makes?",
        "Diversity (DIV)": "Is there diversity in the system responses?",
        "Depth (DEP)": "Does the system discuss topics in depth?",
        "Likeability (LIK)": "Does the system display a likeable personality?",
        "Flexibility (FLE)": "Is the system flexible and adaptable to the user and their interests?",
        "Inquisitiveness (INQ)": "Is the system inquisitive throughout the conversation?",
    }
    asp_names = list(asp_definitions.keys())
    task_asp_map = {
        "Summ": asp_names[:7],
        "D2T": [asp_names[3]] + asp_names[5:7],
        "Dial": asp_names[2:4] + asp_names[5:7] + asp_names[9:],
        "MT": asp_names[6:9],
    }
    assert task in task_asp_map.keys()
    final_gpt_scores = {}
    from mosestokenizer import MosesDetokenizer
    detokenizer = MosesDetokenizer('en')

    def add_dot(text):
        if len(text.strip()) == 0:
            return '.'
        if text.strip()[-1] != '.':
            text = text.strip() + ' .'
        new_text = text
        return new_text

    def detokenize(text: str):
        # words = text.split(" ")
        words = text.split()
        return detokenizer(words)
    if "flan" in metric_name:
        from flan_score import FLANScorer
        import time
        metric2checkpoint = {
            "flan_small_score": "google/flan-t5-small",
            "flan_base_score": "google/flan-t5-base",
            "flan_large_score": "google/flan-t5-large",
            "flan_xl_score": "google/flan-t5-xl",
            "flan_xxl_score": "google/flan-t5-xxl",
        }
        print('metric_name: ', metric_name)
        checkpoint = metric2checkpoint[metric_name]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        flan_scorer = FLANScorer(device=device, checkpoint=checkpoint)
        print('FLANScorer setup finished. Begin calculating FLANScorer.')
        start = time.time()
        for e_asp in task_asp_map[task]:
            print("evaluting ", e_asp)
            # Evaluation is cheap when using non-GPT3 models, so here, we evaluate all aspects by default.
            print('num of examples: ', len(hypotheses))
            asp_df = asp_definitions[e_asp]
            asp_df = asp_df.strip().replace(':', '. ')
            print('asp_df: ', asp_df)
            # perfix
            prefix = asp_df + '\n'
            template = "XXXXX In other words , YYYYY"
            # ref->hypo
            gpt_scores = []

            with tqdm(total=len(hypotheses), desc="Evaluating gptscore (Flan)") as pbar:
                for i, (hypo_group, source) in enumerate(zip(hypotheses, sources)):
                    hyps = [add_dot(detokenize(hypo)) for hypo in hypo_group]
                    src = add_dot(detokenize(source))
                    src_hypo_scores = np.array(flan_scorer.score(
                        [prefix + template.replace('XXXXX', src).replace('YYYYY', '')] * len(hypo_group), hyps, batch_size=4))
                    gpt_scores.append(src_hypo_scores)
                    pbar.update(1)
                    assert len(gpt_scores[i]) == len(hypo_group)
            final_gpt_scores[e_asp] = gpt_scores
        end = time.time()
        print(f'FLANScorer finished. Time: {end-start}')
    else:
        raise NotImplementedError
    # final_gpt_scores["avg_all_asps"] = np.mean([np.array(final_gpt_scores[asp]) for asp in task_asp_map[task]], axis=0).tolist()
    scores = [final_gpt_scores[asp] for asp in task_asp_map[task]]
    avg_scores = [sum(x) / len(x) for x in zip(*scores)]
    final_gpt_scores["avg_all_asps"] = avg_scores
    return final_gpt_scores


def compute_new_n_gram(source: str, candidate: str):
    """
        computer the new n-grams in the candidate compared to source text
    """
    # text
    text = source.lower()
    text_words = word_tokenize(text)
    text_bigrams = [[text_words[j], text_words[j + 1]]
                    for j in range(len(text_words) - 1)]
    text_trigrams = [[text_words[j], text_words[j + 1],
                      text_words[j + 2]] for j in range(len(text_words) - 2)]
    text_quadrigrams = [[text_words[j], text_words[j + 1], text_words[j + 2],
                         text_words[j + 3]] for j in range(len(text_words) - 3)]

    # candidate
    candidate = candidate.lower().replace("<n>", " ")
    candidate_words = word_tokenize(candidate)

    unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
    for j in range(len(candidate_words)):
        if not (candidate_words[j] in text_words):
            unigrams += 1
        if j < len(candidate_words) - 1:
            bigram = [candidate_words[j], candidate_words[j + 1]]
            if not (bigram in text_bigrams):
                bigrams += 1
        if j < len(candidate_words) - 2:
            trigram = [candidate_words[j],
                       candidate_words[j + 1], candidate_words[j + 2]]
            if not (trigram in text_trigrams):
                trigrams += 1
        if j < len(candidate_words) - 3:
            quadrigram = [candidate_words[j], candidate_words[j + 1],
                          candidate_words[j + 2], candidate_words[j + 3]]
            if not (quadrigram in text_quadrigrams):
                quadrigrams += 1
    new_unigram, new_bigram, new_trigram, new_quadrigram = 0, 0, 0, 0
    if len(candidate_words) > 0:
        new_unigram = unigrams / (len(candidate_words) - 0)
    if len(candidate_words) > 1:
        new_bigram = bigrams / (len(candidate_words) - 1)
    if len(candidate_words) > 2:
        new_trigram = trigrams / (len(candidate_words) - 2)
    if len(candidate_words) > 3:
        new_quadrigram = quadrigrams / (len(candidate_words) - 3)
    return new_unigram, new_bigram, new_trigram, new_quadrigram


def eval_novel_n_gram(
    sources: List[str],
    hypotheses: Union[List[List[str]], List[str]],
) -> List[float]:
    """
        evaluate the novel n-gram in the hypotheses compared to the origianl soiurce
    """
    print("Evaluating novel n-gram")
    assert len(hypotheses) == len(sources)
    for i in range(len(hypotheses)):
        if isinstance(hypotheses[i], str):
            hypotheses[i] = [hypotheses[i]]

    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i, (source, hypo_group) in tqdm(enumerate(zip(sources, hypotheses)), desc="evaluate novel n-grams"):
        new_unigrams.append([])
        new_bigrams.append([])
        new_trigrams.append([])
        new_quadrigrams.append([])
        for hypo in hypo_group:
            new_unigram, new_bigram, new_trigram, new_quadrigram = \
                compute_new_n_gram(source, hypo)
            new_unigrams[i].append(new_unigram)
            new_bigrams[i].append(new_bigram)
            new_trigrams[i].append(new_trigram)
            new_quadrigrams[i].append(new_quadrigram)

    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("New unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(
        m_uni, m_bi, m_tri, m_quadri))
    # nested remove list with single element
    if all([len(score) == 1 for score in new_unigrams]):
        new_unigrams = [score[0] for score in new_unigrams]
    if all([len(score) == 1 for score in new_bigrams]):
        new_bigrams = [score[0] for score in new_bigrams]
    if all([len(score) == 1 for score in new_trigrams]):
        new_trigrams = [score[0] for score in new_trigrams]
    if all([len(score) == 1 for score in new_quadrigram]):
        new_quadrigram = [score[0] for score in new_quadrigram]
    return new_unigrams, new_bigrams, new_trigrams, new_quadrigrams


def eval_distinct_n_grams(texts: Union[List[List[str]], List[str]]):
    print("evaluating distinct n-grams")
    for i in range(len(texts)):
        if isinstance(texts[i], str):
            texts[i] = [texts[i]]

    uni_unigrams, uni_bigrams, uni_trigrams, uni_quadrigrams = [], [], [], []
    for i, text_group in tqdm(enumerate(texts), desc='evaluting distinct n-grams'):
        unigrams = []
        bigrams = []
        trigrams = []
        quadrigrams = []
        for text in text_group:
            text = text.lower()
            text_words = word_tokenize(text)
            text_bigrams = [(text_words[j], text_words[j + 1])
                            for j in range(len(text_words) - 1)]
            text_trigrams = [(text_words[j], text_words[j + 1], text_words[j + 2])
                             for j in range(len(text_words) - 2)]
            text_quadrigrams = [(text_words[j], text_words[j + 1], text_words[j + 2],
                                 text_words[j + 3]) for j in range(len(text_words) - 3)]
            unigrams.extend(text_words)
            bigrams.extend(text_bigrams)
            trigrams.extend(text_trigrams)
            quadrigrams.extend(text_quadrigrams)
        unigrams = set(unigrams)
        bigrams = set(unigrams)
        trigrams = set(trigrams)
        quadrigrams = set(quadrigrams)
        uni_unigrams.append(len(unigrams))
        uni_bigrams.append(len(bigrams))
        uni_trigrams.append(len(trigrams))
        uni_quadrigrams.append(len(quadrigrams))
    print(f"Mean unique 1-grams: {np.mean(uni_unigrams)}")
    print(f"Mean unique 2-grams: {np.mean(uni_bigrams)}")
    print(f"Mean unique 3-grams: {np.mean(uni_trigrams)}")
    print(f"Mean unique 4-grams: {np.mean(uni_quadrigrams)}")
    return uni_unigrams, uni_bigrams, uni_trigrams, uni_quadrigrams


def eval_self_bleu(texts: List[List[str]]):
    print("evaluating self bleu")
    for i in range(len(texts)):
        assert isinstance(texts[i], list)

    self_bleus = []
    for i, text_group in tqdm(enumerate(texts), desc='evaluting distinct n-grams'):
        group_self_bleus = []
        for j in range(len(text_group)):
            hypo = text_group[j]
            refs = text_group[:j] + text_group[j + 1:]
            group_self_bleus.append(sentence_bleu(
                hypothesis=hypo, references=refs).score)
        self_bleus.append(np.mean(group_self_bleus))
    print(f"self BLEUs mean: {np.mean(self_bleus)}")
    return self_bleus


def _overall_eval_multi_process(data):
    s = psutil.Process(os.getpid())
    cpu_id = s.cpu_num()
    print("Worker {} is evaluating".format(cpu_id))
    return _overall_eval(*data)


def _overall_eval(candidates, targets, metrics: List[str], sources=None):
    do_flatten = False
    # deepcopy in case it will make change to the passed in candidates and targets
    candidates = deepcopy(candidates)
    targets = deepcopy(targets)
    assert len(candidates) == len(
        targets), f"candidates and targets should have the same length, but got {len(candidates)} and {len(targets)}"
    # if there are no available targets or sources, return None
    if (all([target == '' for target in targets]) or \
            all([target == [] for target in targets])) and \
        (not sources or all([source == '' for source in sources])):
        return {
            metric: [
                [0 for _ in range(len(candidates[i]))]
                for i in range(len(candidates))]
            for metric in metrics
        }
    for i in range(len(candidates)):
        if isinstance(candidates[i], str):
            do_flatten = True
            candidates[i] = [candidates[i]]
        if isinstance(targets[i], str):
            targets[i] = [targets[i]]

    assert isinstance(sources, list) and isinstance(
        sources[0], str), "sources should be a list of str"

    if "rouge" in metrics:
        metrics.remove("rouge")
        metrics.extend(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'])
        logging.warning(
            "Rouge is a group of metrics, using rouge1, rouge2, rougeL, rougeLsum instead")
    scores = {}
    rouge_tyeps = [metric for metric in metrics if metric.startswith('rouge')]
    if rouge_tyeps:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        rouge_scores = eval_rouge(
            _candidates, _targets, rouge_types=rouge_tyeps)
        scores.update(rouge_scores)
    if 'bleu' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bleu_scores = eval_bleu(_candidates, _targets)
        scores.update({'bleu': bleu_scores})
    if 'bleu4' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bleu4_scores = eval_bleu4(_candidates, _targets)
        scores.update({'bleu4': bleu4_scores})
    if 'bleurt' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bleurt_scores = eval_bleurt(_candidates, _targets)
        scores.update({'bleurt': bleurt_scores})
    if 'cider' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        cider_scores = eval_cider(_candidates, _targets)
        scores.update({'cider': cider_scores})
    if 'spice' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        spice_scores = eval_spice(_candidates, _targets)
        scores.update({'spice': spice_scores})
    if 'bertscore' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bert_scores = eval_bertscore(_candidates, _targets)
        scores.update({'bertscore': bert_scores})
    if 'chrf' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        chrf_scores = eval_chrf(_candidates, _targets)
        scores.update({'chrf': chrf_scores})
    if 'prism' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        prism_scores = eval_prism(_candidates, _targets)
        scores.update({'prism': prism_scores})
    if 'comet_da' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        comet_scores = eval_comet_da(_candidates, _targets, sources)
        scores.update({'comet_da': comet_scores})
    if 'cometkiwi_da' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        comet_scores = eval_cometkiwi_da(_candidates, sources)
        scores.update({'cometkiwi_da': comet_scores})

    if 'bart_score' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bart_scores = eval_bartscore(
            _candidates, _targets, metric_name="bart_score")
        scores.update({'bart_score': bart_scores})
    if 'bart_score_cnn' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bart_scores = eval_bartscore(
            _candidates, _targets, metric_name="bart_score_cnn")
        scores.update({'bart_score_cnn': bart_scores})
    if 'bart_score_para' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bart_scores = eval_bartscore(
            _candidates, _targets, metric_name="bart_score_para")
        scores.update({'bart_score_para': bart_scores})
    if 'bart_score_src_hypo' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bart_scores = eval_bartscore_src_hypo(
            _candidates, sources, metric_name="bart_score")
        scores.update({'bart_score_src_hypo': bart_scores})
    if 'bart_score_cnn_src_hypo' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bart_scores = eval_bartscore_src_hypo(
            _candidates, sources, metric_name="bart_score_cnn")
        scores.update({'bart_score_cnn_src_hypo': bart_scores})
    if 'bart_score_para_src_hypo' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        bart_scores = eval_bartscore_src_hypo(
            _candidates, sources, metric_name="bart_score_para")
        scores.update({'bart_score_para_src_hypo': bart_scores})

    if 'unieval_sum' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        unieval_sum_scores = eval_unieval(
            _candidates, _targets, sources, task='summarization')
        for aspect in unieval_sum_scores:
            scores.update(
                {f'unieval_sum_{aspect}': unieval_sum_scores[aspect]})
    if 'unieval_dialogue' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        unieval_dislogue_scores = eval_unieval(
            _candidates, _targets, sources, task='dialogue')
        for aspect in unieval_dislogue_scores:
            scores.update(
                {f'unieval_dialogue_{aspect}': unieval_dislogue_scores[aspect]})
    if 'unieval_fact' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        unieval_fact_scores = eval_unieval(
            _candidates, _targets, sources, task='fact')
        for aspect in unieval_fact_scores:
            scores.update(
                {f'unieval_fact_{aspect}': unieval_fact_scores[aspect]})
    if 'unieval_d2t' in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        unieval_intermediate_scores = eval_unieval(
            _candidates, _targets, sources, task='data2text')
        for aspect in unieval_intermediate_scores:
            scores.update(
                {f'unieval_d2t_{aspect}': unieval_intermediate_scores[aspect]})

    if "instructscore" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        instruct_scores = eval_instructscore(_candidates, _targets)
        scores.update({'instructscore': instruct_scores})

    if "gptscore_flan_sum" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_ref_ist(
            _candidates, _targets, task="Summ", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update({f'gptscore_flan_sum_{aspect}': gpt_scores[aspect]})
    if "gptscore_flan_d2t" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_ref_ist(
            _candidates, _targets, task="D2T", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update({f'gptscore_flan_d2t_{aspect}': gpt_scores[aspect]})
    if "gptscore_flan_dial" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_ref_ist(
            _candidates, _targets, task="Dial", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update({f'gptscore_flan_dial_{aspect}': gpt_scores[aspect]})
    if "gptscore_flan_mt" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_ref_ist(
            _candidates, _targets, task="MT", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update({f'gptscore_flan_mt_{aspect}': gpt_scores[aspect]})

    if "gptscore_flan_sum_src_hypo" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_src_ist(
            _candidates, sources, task="Summ", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update(
                {f'gptscore_flan_sum_src_hypo_{aspect}': gpt_scores[aspect]})
    if "gptscore_flan_d2t_src_hypo" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_src_ist(
            _candidates, sources, task="D2T", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update(
                {f'gptscore_flan_d2t_src_hypo_{aspect}': gpt_scores[aspect]})
    if "gptscore_flan_dial_src_hypo" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_src_ist(
            _candidates, sources, task="Dial", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update(
                {f'gptscore_flan_dial_src_hypo_{aspect}': gpt_scores[aspect]})
    if "gptscore_flan_mt_src_hypo" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        gpt_scores = eval_gptscore_src_ist(
            _candidates, sources, task="MT", metric_name="flan_base_score")
        for aspect in gpt_scores:
            scores.update(
                {f'gptscore_flan_mt_src_hypo_{aspect}': gpt_scores[aspect]})
    if "chatgpt_zero_shot" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        chatgpt_scores = eval_GPT_zero_shot(_candidates, _targets, sources, model_name="ChatGPT")
        scores.update({'chatgpt_zero_shot': chatgpt_scores})
    if "gpt4_zero_shot" in metrics:
        _candidates, _targets = deepcopy(candidates), deepcopy(targets)
        chatgpt_scores = eval_GPT_zero_shot(_candidates, _targets, sources, model_name="gpt-4")
        scores.update({'gpt4_zero_shot': chatgpt_scores})

    if len(scores) == 0:
        logging.warning(
            "No metric is matched in {}, please check the spelling".format(metrics))
    if do_flatten:
        for metric in scores:
            assert all([len(score) == 1 for score in scores[metric]])
            scores[metric] = [score[0] for score in scores[metric]]
    return scores


def overall_eval(
    candidates: Union[List[List[str]], List[str]],
    targets: Union[List[str], List[List[str]]],
    metrics: List[str],
    sources: List[str] = None,
    num_workers: int = 1,
) -> Dict[str, List[float]]:
    """
    Args:
        candidates: the candidates
        targets: the targets
        metrics: the metrics to be evaluated
        num_workers: the number of workers to be used
    Return:
        A dict of scores, same shape with candidates for each metric
    """
    if num_workers > 1:
        cpu_num = psutil.cpu_count(logical=False)
        num_workers = min(num_workers, cpu_num)
        print("Using {} workers to evaluate".format(num_workers))
        chunk_size = len(candidates) // num_workers + 1
        candidates_chunks = [candidates[i:i + chunk_size]
                             for i in range(0, len(candidates), chunk_size)]
        targets_chunks = [targets[i:i + chunk_size]
                          for i in range(0, len(targets), chunk_size)]
        if sources is not None:
            sources_chunks = [sources[i:i + chunk_size]
                              for i in range(0, len(sources), chunk_size)]
            datas = [(candidates_chunks[i], targets_chunks[i], metrics,
                      sources_chunks[i]) for i in range(len(candidates_chunks))]
        else:
            datas = [(candidates_chunks[i], targets_chunks[i], metrics)
                     for i in range(len(candidates_chunks))]
        scores_chunks = process_map(
            _overall_eval_multi_process, datas, chunksize=1, max_workers=num_workers)
        scores = {}
        for chunk in scores_chunks:
            for k, v in chunk.items():
                scores[k] = scores.get(k, []) + v
    else:
        scores = _overall_eval(candidates, targets, metrics, sources=sources)
    return scores

import os,sys,json
from pathlib import Path
from datasets import load_dataset
import numpy as np


def get_ranks_from_chatgpt_cmps(ds_data):

    # transform chatgpt cmp_results to [bz, c, c]
    bz = len(ds_data)
    c = len(ds_data[0]['candidates'])

    chatgpt_cmp_results = np.zeros((bz, c, c))
    _models = [c['model'] for c in ds_data[0]['candidates']]
    for i, d in enumerate(ds_data):
        models = [c['model'] for c in d['candidates']]
        assert models == _models, f"models not match: {models} vs {_models}"
        if isinstance(d['cmp_results'],str):
            if d['cmp_results'] == "null":
                continue
            else:
                d['cmp_results'] = eval(d['cmp_results'])
        
        if isinstance(d['cmp_results'], dict):
            for cand in d['candidates']:
                cand["scores"]["gpt_rank_score"] = 0
            for key, value in d['cmp_results'].items():
                idx1, idx2 = models.index(key.split(",")[0]), models.index(key.split(",")[1])
                if value == "A is better":
                    for cand in d['candidates']:
                        if cand['model'] == key.split(",")[0]:
                            cand["scores"]["gpt_rank_score"] += 1
                        if cand['model'] == key.split(",")[1]:
                            cand["scores"]["gpt_rank_score"] -= 1
                    chatgpt_cmp_results[i][idx1][idx2] += 1
                    chatgpt_cmp_results[i][idx2][idx1] -= 1
                elif value == "B is better":
                    for cand in d['candidates']:
                        if cand['model'] == key.split(",")[0]:
                            cand["scores"]["gpt_rank_score"] -= 1
                        if cand['model'] == key.split(",")[1]:
                            cand["scores"]["gpt_rank_score"] += 1
                    chatgpt_cmp_results[i][idx1][idx2] -= 1
                    chatgpt_cmp_results[i][idx2][idx1] += 1
                elif value == "Same good":
                    for cand in d['candidates']:
                        if cand['model'] == key.split(",")[0]:
                            cand["scores"]["gpt_rank_score"] += 0.5
                        if cand['model'] == key.split(",")[1]:
                            cand["scores"]["gpt_rank_score"] += 0.5
                    chatgpt_cmp_results[i][idx1][idx2] += 0.5
                    chatgpt_cmp_results[i][idx2][idx1] += 0.5
                elif value == "Same bad":
                    for cand in d['candidates']:
                        if cand['model'] == key.split(",")[0]:
                            cand["scores"]["gpt_rank_score"] -= 0.5
                        if cand['model'] == key.split(",")[1]:
                            cand["scores"]["gpt_rank_score"] -= 0.5
                    chatgpt_cmp_results[i][idx1][idx2] -= 0.5
                    chatgpt_cmp_results[i][idx2][idx1] -= 0.5
                else:
                    raise ValueError("Unknown value: {}".format(value))
                
            
        else:
            print(d['cmp_results'])

    chatgpt_cmp_ranks = get_ranks_from_cmps(chatgpt_cmp_results)

    model_ranks_map = {}
    for i, model_name in enumerate(_models):
        model_ranks_map[model_name] = chatgpt_cmp_ranks[:, i]
    return model_ranks_map, chatgpt_cmp_results,ds_data


def get_ranks_from_cmps(cmp_results, policy="max_logits"):
    """
    Args:
        cmp_results: ndarray of shape (n, c, c) where n is the number of samples, c is the number of candidates
            for each element, >0 means the first candidate is better than the second one, <0 means the second one is better
    Returns:
        ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    """
    if isinstance(cmp_results, list):
        cmp_results = np.array(cmp_results)
    bz, c, _ = cmp_results.shape
    ranks = np.zeros((bz, c), dtype=np.int32)
    for i in range(bz):
        if policy == "max_logits":
            scores = (cmp_results[i] - cmp_results[i].T).sum(axis=-1)
        elif policy == "max_wins":
            scores = (cmp_results[i] > 0).sum(axis=-1) + (cmp_results[i] < 0).sum(axis=-2)
        _ranks = get_ranks_from_scores(scores)
        ranks[i] = _ranks
    return ranks

def get_ranks_from_scores(scores):
    """
    Args:
        scores: ndarray of shape (n, c) or (c) where n is the number of samples, c is the number of candidates
        Treat same as higher one
        
    Returns:
        ranks: ndarray of shape (n, c) or (c) where n is the number of samples, c is the number of candidates
    """
    if isinstance(scores, list):
        scores = np.array(scores)
    orig_shape = scores.shape
    if len(scores.shape) == 1:
        scores = scores.reshape(1, -1)
    bz, c = scores.shape
    ranks = np.zeros((bz, c), dtype=np.int32)
    for i in range(bz):
        sorted_scores_i = list(sorted(list(scores[i]), reverse=True))
        for j in range(c):
            ranks[i, j] = sorted_scores_i.index(scores[i, j]) + 1
    
    ranks = ranks.reshape(orig_shape)
    return ranks


if __name__ == "__main__":

    output_file = "./test_data_prepared.json"
    dataset = load_dataset("llm-blender/mix-instruct")
    data = [x for x in dataset['test']]
    new_data = []

    for d in data:
        new_data.append({
            "id": d["id"],
            "instruction": d["instruction"] if d["instruction"] else "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.",
            "input": d["input"],
            "output": d["output"],
            "candidates" : d["candidates"],
            "cmp_results": d["cmp_results"],
        })

    _,_,new_data=get_ranks_from_chatgpt_cmps(data)
    
    data = new_data
    need_remove = []
    for item in data:
        for cand in item['candidates']:
            if "gpt_rank_score" not in cand["scores"]:
                need_remove.append(item)
                break

    for item in need_remove:
        data.remove(item)
        
    with open(output_file, 'w') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)
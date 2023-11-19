"""
    Calculate correlations between human scores and metric scores+
"""
import fire
import sys
import os
import json
import scipy
import numpy as np
import itertools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import List
from mt_metrics_eval.stats import Correlation
from collections import defaultdict
import prettytable as pt
from common.evaluation import overall_eval


class MyCorrelation(Correlation):
    """
        From https://github.com/google-research/mt-metrics-eval
    """
    def __init__(self, num_sys: int, gold_scores: List[int], metric_scores: List[int]):
        # remove nan in metrics scores
        none_metric_scores_idxs = [idx for idx,
                                   x in enumerate(metric_scores) if x is None]
        # print("Remove {} nan scores from {} scores".format(
        #     len(none_metric_scores_idxs),
        #     len(metric_scores)
        # ))
        gold_scores = gold_scores.copy()
        # set gold scores to None if metric scores are None
        for idx in none_metric_scores_idxs[::-1]:
            gold_scores[idx] = None
        super().__init__(num_sys, gold_scores, metric_scores)


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


def cor_spearman(hypo_ranks, ref_ranks):
    """
    Args:
        hypo_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_ranks: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean of the diagonal elements of the spearman correlation matrix
    """
    if isinstance(hypo_ranks, list):
        hypo_ranks = np.array(hypo_ranks)
    if isinstance(ref_ranks, list):
        ref_ranks = np.array(ref_ranks)
    assert hypo_ranks.shape == ref_ranks.shape
    bz, c = hypo_ranks.shape
    hypo_ranks = hypo_ranks.reshape(bz, c).T
    ref_ranks = ref_ranks.reshape(bz, c).T
    cor = 0
    for i in range(c):
        cor += scipy.stats.spearmanr(hypo_ranks[i], ref_ranks[i]).correlation
    cor /= c
    return cor


def main(
    input_file: str,
    output_file: str,
    human_score_names: List[str],
    metrics: List[str],
    num_workers: int = 1,
    overwrite: bool = False,
    print_results: bool = False,
    average_by: str = "none",
    as_rank: bool = False,
    add_aggrement: bool = False,
):
    if isinstance(metrics, str):
        metrics = metrics.split(" ")
    else:
        metrics = list(metrics)
    if isinstance(human_score_names, str):
        human_score_names = [human_score_names]
    else:
        human_score_names = list(human_score_names)
    if not os.path.exists(output_file):
        with open(input_file) as f:
            data = json.load(f)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Saved a inital copy to {output_file}")
    else:
        with open(output_file) as f:
            data = json.load(f)
        with open(input_file) as f:
            data_input = json.load(f)
        if len(data) != len(data_input):
            print("Warning: data and data_input have different length")
            with open(output_file, "w") as f:
                json.dump(data_input, f, indent=4, ensure_ascii=False)
                print(f"Saved a inital copy to {output_file}")
            data = data_input

    evaled_metrics = []
    for metric in metrics:
        if any("my_" + metric in score_name for score_name in data[0]['candidates'][0]['scores'].keys()):
            # ######## Temp Code remember delete #########
            # if "_src_hypo" in metric:
            #     continue
            evaled_metrics.append(metric)
    if not overwrite:
        to_eval_metrics = [
            metric for metric in metrics if metric not in evaled_metrics]
    else:
        to_eval_metrics = metrics
    print(metrics)
    print("To eval metrics: {}".format(to_eval_metrics))
    if len(to_eval_metrics) > 0:
        sources = [d["input"] for d in data]
        references = [d.get("output") or d.get('refs') or "" for d in data]
        references = [ref[0] if isinstance(ref, list) else ref for ref in references]
        hypotheses = [[cand['text'] for cand in d['candidates']] for d in data]
        scores = overall_eval(hypotheses, references, to_eval_metrics,
                              sources=sources, num_workers=num_workers)
        for metric, _scores in scores.items():
            for i, item in enumerate(data):
                for j, cand in enumerate(item['candidates']):
                    cand['scores']["my_" + metric] = _scores[i][j]
        with open(output_file, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"Saved to {output_file}")

    if print_results:
        # compute correlations
        human_score_corr_dict = {human_score_name: defaultdict(
            dict) for human_score_name in human_score_names}
        all_scores_names = list(data[0]['candidates'][0]['scores'].keys())
        other_scores_names = [
            score_name for score_name in all_scores_names if score_name not in human_score_names]
        # remove logprobs
        if "logprobs" in other_scores_names:
            other_scores_names.remove("logprobs")
        # remove redundant bart_score_cnn_src_hypo and bart_score_cnn_ref_hypo
        bartscore_redundant_scores_names = [x for x in other_scores_names if x.startswith(
            "bart_score_cnn_src_hypo") or x.startswith("bart_score_cnn_ref_hypo")]
        for score_name in bartscore_redundant_scores_names:
            other_scores_names.remove(score_name)
        for h_name in human_score_names:
            h_scores = [[cand['scores'][h_name]
                         for cand in item['candidates']] for item in data]

            for o_name in other_scores_names:
                o_scores = [[cand['scores'][o_name]
                             for cand in item['candidates']] for item in data]
                if as_rank:
                    # transform scores to ranks before computing correlation
                    o_scores = get_ranks_from_scores(o_scores)
                    # higher is better
                    o_scores = (- np.array(o_scores)).tolist()
                h_scores_flat = []
                h_scores_T = [x for x in zip(*h_scores)]
                for sublist in h_scores_T:
                    h_scores_flat.extend(sublist)
                # h_scores = np.array(h_scores).T.reshape(-1).tolist()
                o_scores_flat = []
                o_scores_T = [x for x in zip(*o_scores)]
                for sublist in o_scores_T:
                    o_scores_flat.extend(sublist)
                # o_scores = np.array(o_scores).T.reshape(-1).tolist()
                corr = MyCorrelation(1, h_scores_flat, o_scores_flat)
                human_score_corr_dict[h_name][o_name]['Pearson'] = [
                    round(x, 4) for x in corr.Pearson(average_by=average_by)]
                human_score_corr_dict[h_name][o_name]['Spearman'] = [
                    round(x, 4) for x in corr.Spearman(average_by=average_by)]
                human_score_corr_dict[h_name][o_name]['Kendall'] = [
                    round(x, 4) for x in corr.Kendall(average_by=average_by)]

        # print table
        table = pt.PrettyTable()
        table.field_names = ["Human Score", "Metric",
                             "Pearson", "Spearman", "Kendall"]
        table.align["Human Score"] = "l"
        table.align["Metric"] = "l"
        table.align["Pearson"] = "l"
        table.align["Spearman"] = "l"
        table.align["Kendall"] = "l"
        # add data
        for h_name in human_score_names:
            other_scores_pearson = [human_score_corr_dict[h_name]
                                    [o_name]['Pearson'][0] for o_name in other_scores_names]
            sorted_other_scores_names = [x for _, x in sorted(
                zip(other_scores_pearson, other_scores_names), reverse=True)]
            for o_name in sorted_other_scores_names:
                table.add_row([
                    h_name,
                    o_name,
                    human_score_corr_dict[h_name][o_name]['Pearson'],
                    human_score_corr_dict[h_name][o_name]['Spearman'],
                    human_score_corr_dict[h_name][o_name]['Kendall'],
                ])
            if h_name != human_score_names[-1]:
                table.add_row(["-", "-", "-", "-", "-"])
        if add_aggrement:
            # add aggrement column, pairwise agreement
            aggrement_columns = []
            for h_name in human_score_names:
                other_scores_pearson = [human_score_corr_dict[h_name]
                    [o_name]['Pearson'][0] for o_name in other_scores_names]
                sorted_other_scores_names = [x for _, x in sorted(
                    zip(other_scores_pearson, other_scores_names), reverse=True)]
                for o_name in sorted_other_scores_names:
                    h_scores = [[cand['scores'][h_name]
                        for cand in item['candidates']] for item in data]
                    o_scores = [[cand['scores'][o_name]
                        for cand in item['candidates']] for item in data]
                    aggreement = 0
                    total = 0
                    for _h_scores, _o_scores in zip(h_scores, o_scores):
                        num_cands = len(_h_scores)
                        for i, j in itertools.combinations(range(num_cands), 2):
                            total += 1
                            if (_h_scores[i] - _h_scores[j]) * (_o_scores[i] - _o_scores[j]) > 0:
                                aggreement += 1
                    aggreement_ratio = aggreement / total
                    aggrement_columns.append(aggreement_ratio)
                if h_name != human_score_names[-1]:
                    aggrement_columns.append("-")
            table.add_column("Aggrement", aggrement_columns)
        
        print("File: {}".format(output_file))
        print("Correlations (Sorted by Pearson):")
        print(table)
        


if __name__ == "__main__":
    fire.Fire(main)

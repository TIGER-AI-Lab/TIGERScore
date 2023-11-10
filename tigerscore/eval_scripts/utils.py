import logging
from mt_metrics_eval.stats import Correlation
from typing import List


class MyCorrelation(Correlation):
    def __init__(self, num_sys: int, gold_scores: List[int], metric_scores: List[int]):
        # remove nan in metrics scores
        none_metric_scores_idxs = [idx for idx,
                                   x in enumerate(metric_scores) if x is None]
        logging.info("Remove {} nan scores from {} scores".format(
            len(none_metric_scores_idxs),
            len(metric_scores)
        ))
        gold_scores = gold_scores.copy()
        # set gold scores to None if metric scores are None
        for idx in none_metric_scores_idxs[::-1]:
            gold_scores[idx] = None
        super().__init__(num_sys, gold_scores, metric_scores)

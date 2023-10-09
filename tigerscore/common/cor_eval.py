import numpy as np
import scipy
def cor_pearson(hypo_scores, ref_scores):
    """
    Args:
        hypo_scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean correlation coefficient
    """
    if isinstance(hypo_scores, list):
        hypo_scores = np.array(hypo_scores)
    if isinstance(ref_scores, list):
        ref_scores = np.array(ref_scores)
    assert hypo_scores.shape == ref_scores.shape
    bz, c = hypo_scores.shape
    hypo_scores = hypo_scores.reshape(bz, c).T
    ref_scores = ref_scores.reshape(bz, c).T
    cor = 0
    for i in range(c):
        cor += np.corrcoef(hypo_scores[i], ref_scores[i])[0, 1]
    cor /= c
    return cor

def cor_spearman(hypo_scores, ref_scores):
    """
    Args:
        hypo_scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean of the diagonal elements of the spearman correlation matrix
    """
    if isinstance(hypo_scores, list):
        hypo_scores = np.array(hypo_scores)
    if isinstance(ref_scores, list):
        ref_scores = np.array(ref_scores)
    assert hypo_scores.shape == ref_scores.shape
    bz, c = hypo_scores.shape
    hypo_scores = hypo_scores.reshape(bz, c).T
    ref_scores = ref_scores.reshape(bz, c).T
    cor = 0
    for i in range(c):
        cor += scipy.stats.spearmanr(hypo_scores[i], ref_scores[i]).correlation
    cor /= c
    return cor

            
def cor_spearman_footrule(hypo_scores, ref_scores):
    """
    Args:
        hypo_scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
        ref_scores: ndarray of shape (n, c) where n is the number of samples, c is the number of candidates
    returns:
        cor: float, the mean of the set of the spearman correlation coefficients
    """
    if isinstance(hypo_scores, list):
        hypo_scores = np.array(hypo_scores)
    if isinstance(ref_scores, list):
        ref_scores = np.array(ref_scores)
    assert hypo_scores.shape == ref_scores.shape
    bz, c = hypo_scores.shape
    hypo_scores = hypo_scores.reshape(bz, c)
    ref_scores = ref_scores.reshape(bz, c)
    return np.abs(hypo_scores - ref_scores).sum(axis=-1).mean()
"""
Edit F1 metric from AmbigQA paper (https://aclanthology.org/2020.emnlp-main.466/)
"""
import numpy as np


def get_edit_f1(original, gold_rewrite, predicted_rewrite, verbose=False):
    original_tokens = original.split(' ')
    gold_tokens = gold_rewrite.split(' ')
    generated_tokens = predicted_rewrite.split(' ')
    reference = _get_edits(original_tokens, gold_tokens)
    prediction = _get_edits(original_tokens, generated_tokens)

    if verbose:
        print(reference)
        print(prediction)
    
    # now compare the reference edits and predicted edits
    if len(reference) == len(prediction) == 0:         # if neither has edits, full score
        edit_f1 = 1
    elif len(reference) == 0 or len(prediction) == 0:  # if only one of them has no edits, zero score
        edit_f1 = 0
    else:                                              # otherwise, compute F1 score
        edit_f1 = _get_f1(reference, prediction)
    
    return edit_f1


def _get_edits(tokens1, tokens2):
    """
    takes the original tokens, tokens1, and the tokens in the rewrite, tokens2
    returns a list of added and deleted tokens
    """
    allCommon = []
    while True:
        commons = list(set(tokens1) & set(tokens2))
        if len(commons) == 0:
            break
        allCommon += commons
        for c in commons:
            ind1, ind2 = tokens1.index(c), tokens2.index(c)
            tokens1 = tokens1[:ind1] + tokens1[ind1+1:]
            tokens2 = tokens2[:ind2] + tokens2[ind2+1:]
    deleted = ["[DELETED]" + token for token in tokens1]
    added = ["[ADDED]" + token for token in tokens2]
    return deleted + added


def _get_f1(reference_edits, predicted_edits):
    occupied_references = [False for _ in reference_edits]   # number of gold edits that were predicted
    occupied_predictions = [False for _ in predicted_edits]   # number of predicted edits that were correct
    
    for i, reference in enumerate(reference_edits):
        for j, prediction in enumerate(predicted_edits):
            if occupied_references[i] or occupied_predictions[j]:
                continue
            em = reference == prediction
            if em:
                occupied_references[i] = True
                occupied_predictions[j] = True
    assert np.sum(occupied_references) == np.sum(occupied_predictions)
    a, b = np.mean(occupied_references), np.mean(occupied_predictions)

    if a+b==0:
        return 0.
    return 2*a*b/(a+b)

"""
Statistics for §3: Does Ambiguity Explain Disagreement?
"""
import math
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
from mturk.utils import get_disambiguation_idxs
from utils.constants import id2label, label2id
import numpy as np
from utils.utils import get_mode

NUM_ANNOTATORS = 9


def compute_agreement_statistics(results_df):
    """
    compute crowdworker interannotator agreement statistics on ambiguous examples and disambiguations
    """
    num_examples = len(results_df.index)

    # NLI annotation for ambiguous examples
    ambiguous_X = np.empty((num_examples, NUM_ANNOTATORS))

    # NLI annotation for disambiguations, num_disambiguations x NUM_ANNOTATORS
    disambiguated_X = []

    for i, row in results_df.iterrows():
        disambiguation_labels = row['labels']
        disambiguation_idxs = get_disambiguation_idxs(row)
        
        num_disambiguations = len(disambiguation_idxs)
        disambiguation_labels = [disambiguation_labels[j] for j in disambiguation_idxs]
        
        ambiguous_X[i,:] = [label2id[l] for l in row['q0_gold']]
        
        for j in range(num_disambiguations):
            disambiguated_X.append([label2id[l] for l in row[f'q{disambiguation_idxs[j]+1}_gold']])

    return {
        'ambiguous_kappa': fleiss_kappa(aggregate_raters(ambiguous_X)[0]),
        'disambiguated_kappa': fleiss_kappa(aggregate_raters(disambiguated_X)[0])
    }


def compute_acceptability_statistics(results_df):
    # accept_each = list of count of annotators who accept each disambiguation (len: num disambiguations)
    # accept_all = list of count of annotators who accept all disambiguations

    accept_each, accept_all, accept_distractor = [], [], []
    for _, row in results_df.iterrows():
        # disambiguation acceptability
        disambiguation_idxs = get_disambiguation_idxs(row)
        disambiguation_judgments = [row[f'd{i+1}_gold'] for i in disambiguation_idxs]
        accept_each_cts = np.sum(disambiguation_judgments, axis=1)
        accept_each.extend(accept_each_cts)
        
        disambiguation_judgments_per_annotator = zip(*disambiguation_judgments)
        accept_all_ct = len([j for j in disambiguation_judgments_per_annotator if np.all(j)])
        accept_all.append(accept_all_ct)

        # distractor acceptability
        if len(disambiguation_idxs) == 2:
            distractor_idx = int(row['distractor_idx'])
            accept_distractor_ct = np.sum(row[f'd{distractor_idx+1}_gold'])
            accept_distractor.append(accept_distractor_ct)
    
    return {
        'disambiguation_acceptability': np.mean(accept_each)/NUM_ANNOTATORS,
        'all_disambiguation_acceptability': np.mean(accept_all)/NUM_ANNOTATORS,
        'distractor_acceptability': np.mean(accept_distractor)/NUM_ANNOTATORS,
    }


def compute_committee_perf(results_df):
    """
    compute human performance on the dataset
    """
    committee_wrong_ids = []
    for _, row in results_df.iterrows():
        disambiguation_idxs = get_disambiguation_idxs(row)
        # if disambiguation is not accepted or gold label not chosen for any of the disambiguations, then the committee is wrong
        if np.any([(get_mode(row[f'd{j+1}_gold']) == False) or (get_mode(row[f'q{j+1}_gold']) != row['labels'][j]) for j in disambiguation_idxs]):
            committee_wrong_ids.append(row['id'])
    
    return {
        'committee_wrong_ids': committee_wrong_ids,
        'committee_perf': (len(results_df.index) - len(committee_wrong_ids))/len(results_df.index)
    }
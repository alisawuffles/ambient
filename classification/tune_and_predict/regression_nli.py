from pathlib import Path
from tqdm import tqdm
from utils.constants import NLI_LABELS, label2id, MAX_DISAMBIGUATIONS
from utils.utils import ensure_dir
import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score
import click
import json
from itertools import combinations


@click.command()
@click.option('--model_path', type=str, default='models/roberta-large-rsnli-unli')
@click.option('--data_path', type=str, default='annotation/AmbiEnt')
def main(model_path: str, data_path: str):
    data_path = Path(data_path)
    dev_df, test_df = pd.read_json(data_path / 'dev.jsonl', lines=True), pd.read_json(data_path / 'test.jsonl', lines=True)
    
    model = RobertaForSequenceClassification.from_pretrained(model_path).to('cuda')
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    # construct true matrix of shape [m, 3] where m = # of dev examples
    # T[i,j] = 1 if example i has label j, else 0
    T = np.zeros((len(dev_df.index), len(NLI_LABELS)))
    for i, ex in dev_df.iterrows():
        for label in ex['labels'].split(', '):
            T[i, label2id[label]] = 1

    # compute prediction matrix M of shape [m, 1]
    # M[i] = r where r is a prediction in [0,1]
    M = np.empty((len(dev_df.index)))
    for i, ex in dev_df.iterrows():
        x = tokenizer(ex['premise'], ex['hypothesis'], return_tensors='pt').to('cuda')
        logit = model(**x).logits.cpu().detach().numpy()
        M[i] = float(logit.squeeze())
    
    # use all pairs of values (a,b) where a<b as possible endpoints
    # for each label and endpoint pair, compute prediction matrix P of shape [m, 1]
    # find best threshold for each label based on F1 score
    thresholds = M.flatten()
    endpoints = [(a,b) for a,b in combinations(thresholds, 2) if a<b]
    best_threshold = {label: None for label in NLI_LABELS}

    for j, label in enumerate(NLI_LABELS):
        endpoint_dict = {}
        for a,b in tqdm(endpoints, desc=label):
            P = np.zeros((len(dev_df.index)))
            for i, ex in dev_df.iterrows():
                if a <= M[i] <= b:
                    P[i] = 1
            endpoint_dict[(a,b)] = f1_score(T[:,j], P)
        t = max(endpoint_dict, key=endpoint_dict.get)
        best_threshold[label] = t
    
    P = np.empty((0, len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)))
    T = np.empty((0, len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)))

    for i, ex in tqdm(test_df.iterrows(), total=len(test_df.index)):
        P_row, T_row = np.zeros(len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)), np.zeros(len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS))
        x = tokenizer(ex['premise'], ex['hypothesis'], return_tensors='pt').to('cuda')
        
        # calculate predictions
        logit = float(model(**x).logits.squeeze(0).cpu().detach().numpy())
        for label, endpoints in best_threshold.items():
            a, b = endpoints
            if a <= logit <= b:
                P_row[label2id[label]] = 1
        
        # store gold labels
        true = [label2id[l] for l in ex['labels'].split(', ')]
        T_row[true] = 1
        
        # calculate predictions, store gold labels for disambiguations
        for j, disam in enumerate(ex['disambiguations']):
            x = tokenizer(disam['premise'], disam['hypothesis'], return_tensors='pt').to('cuda')
            logit = float(model(**x).logits.squeeze(0).cpu().detach().numpy())
            for label, endpoints in best_threshold.items():
                a, b = endpoints
                if a <= logit <= b:
                    P_row[len(NLI_LABELS)*(j+1)+label2id[label]] = 1
            true = label2id[disam['label']]
            T_row[len(NLI_LABELS)*(j+1)+true] = 1
    
        P = np.concatenate((P, np.expand_dims(P_row, axis=0)))
        T = np.concatenate((T, np.expand_dims(T_row, axis=0)))
    
    f1 = f1_score(T[:,:len(NLI_LABELS)], P[:,:len(NLI_LABELS)], average='macro')
    em = np.mean([np.array_equal(T[i,:len(NLI_LABELS)], P[i,:len(NLI_LABELS)]) for i in range(T.shape[0])])
    group_em = np.mean([np.array_equal(T[i,:], P[i,:]) for i in range(T.shape[0])])
    
    res = {
        'logit_threshold': best_threshold,
        'f1': f1,
        'em': em,
        'group_em': group_em
    }

    out_dir = Path(f'results/multilabel/{model_path.split("/")[-2]}')
    ensure_dir(out_dir)
    with open(out_dir / f'{model_path.split("/")[-1]}.json', 'w') as fo:
        json.dump(res, fo, indent=4)


if __name__ == '__main__':
    main()
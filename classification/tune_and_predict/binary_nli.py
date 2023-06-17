from pathlib import Path
from tqdm import tqdm
from utils.constants import NLI_LABELS, label2id, MAX_DISAMBIGUATIONS
from utils.utils import ensure_dir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import f1_score
import click
import json


@click.command()
@click.option('--ent_model_path', type=str)
@click.option('--neu_model_path', type=str)
@click.option('--con_model_path', type=str)
@click.option('--data_path', type=str, default='annotation/AmbiEnt')
@click.option('--do_plot', type=bool, default=False)
def main(ent_model_path: str, neu_model_path: str, con_model_path: str, data_path: str, do_plot: bool):
    data_path = Path(data_path)
    dev_df, test_df = pd.read_json(data_path / 'dev.jsonl', lines=True), pd.read_json(data_path / 'test.jsonl', lines=True)

    tokenizer = RobertaTokenizer.from_pretrained(ent_model_path)
    model_paths = {'entailment': ent_model_path, 'neutral': neu_model_path, 'contradiction': con_model_path}
    binary_models = {}

    for label in NLI_LABELS:
        binary_models[label] = RobertaForSequenceClassification.from_pretrained(model_paths[label]).to('cuda')

    # construct true matrix of shape [m, 3] where m = # of dev examples
    # T[i,j] = 1 if example i has label j, else 0
    T = np.zeros((len(dev_df.index), len(NLI_LABELS)))
    for i, ex in dev_df.iterrows():
        for label in ex['labels'].split(', '):
            T[i, label2id[label]] = 1
        
    # compute logit matrix M of shape [m, 3]
    # M[i,j] = l for l is the logit for example i corresponding to label j
    M = np.empty((len(dev_df.index), len(NLI_LABELS)))
    for i, ex in dev_df.iterrows():
        x = tokenizer(ex['premise'], ex['hypothesis'], return_tensors='pt').to('cuda')
        for j, (label, binary_model) in enumerate(binary_models.items()):
            positive_logit = binary_model(**x).logits.cpu().detach().numpy().squeeze()[0]
            M[i, j] = positive_logit
    
    # use all logit values as possible thresholds
    # for each threshold, compute prediction matrix P of shape [m, 3]
    # P[i,j] = l if M[i,j] > t
    thresholds = M.flatten()
    threshold_dict = {}

    for t in thresholds:
        P = np.zeros((len(dev_df.index), len(NLI_LABELS)))
        for i, ex in dev_df.iterrows():
            for j, l in enumerate(NLI_LABELS):
                if M[i, j] > t:
                    P[i,j] = 1
        threshold_dict[t] = f1_score(T, P, average='macro')    

    if do_plot:
        fig, ax = plt.subplots()
        sns.lineplot(threshold_dict.keys(), threshold_dict.values())
        ax.set_ylabel('Macro F1')
        ax.set_xlabel('Logit threshold')
        ax.set_title('Three binary models model')
    
    t = max(threshold_dict, key=threshold_dict.get)

    # apply threshold to do evaluation on test set
    P = np.empty((0, len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)))
    T = np.empty((0, len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)))

    for i, ex in tqdm(test_df.iterrows(), total=len(test_df.index)):
        P_row, T_row = np.zeros(len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)), np.zeros(len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS))
        x = tokenizer(ex['premise'], ex['hypothesis'], return_tensors='pt').to('cuda')
        for j, (label, binary_model) in enumerate(binary_models.items()):
            positive_logit = binary_model(**x).logits.cpu().detach().numpy().squeeze()[0]
            if positive_logit > t:
                P_row[j] = 1
        true = [label2id[l] for l in ex['labels'].split(', ')] # true labels
        T_row[true] = 1
        
        for j, disam in enumerate(ex['disambiguations']):
            x = tokenizer(disam['premise'], disam['hypothesis'], return_tensors='pt').to('cuda')
            for k, (label, binary_model) in enumerate(binary_models.items()):
                positive_logit = binary_model(**x).logits.cpu().detach().numpy().squeeze()[0]
                if positive_logit > t:
                    P_row[len(NLI_LABELS)*(j+1)+k] = 1
            true = label2id[disam['label']]
            T_row[len(NLI_LABELS)*(j+1)+true] = 1
        
        P = np.concatenate((P, np.expand_dims(P_row, axis=0)))
        T = np.concatenate((T, np.expand_dims(T_row, axis=0)))
    
    f1 = f1_score(T[:,:len(NLI_LABELS)], P[:,:len(NLI_LABELS)], average='macro')
    em = np.mean([np.array_equal(T[i,:len(NLI_LABELS)], P[i,:len(NLI_LABELS)]) for i in range(T.shape[0])])
    group_em = np.mean([np.array_equal(T[i,:], P[i,:]) for i in range(T.shape[0])])

    res = {
        'logit_threshold': t,
        'f1': f1,
        'em': em,
        'group_em': group_em
    }

    out_dir = Path(f'results/multilabel/binary_models')
    ensure_dir(out_dir)
    with open(out_dir / f'{ent_model_path.split("/")[-1]}.json', 'w') as fo:
        json.dump(res, fo, indent=4)


if __name__ == '__main__':
    main()
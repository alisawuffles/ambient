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
@click.option('--model_path', type=str, default='models/roberta-large-wanli-set')
@click.option('--data_path', type=str, default='annotation/AmbiEnt')
@click.option('--do_plot', type=bool, default=False)
def main(model_path: str, data_path: str, do_plot: bool):
    data_path = Path(data_path)
    test_df = pd.read_json(data_path / 'test.jsonl', lines=True)
    
    model = RobertaForSequenceClassification.from_pretrained(model_path).to('cuda')
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    # no threshold tuning required, just make predictions!
    P = np.empty((0, len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)))
    T = np.empty((0, len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)))
    results = []

    for i, ex in tqdm(test_df.iterrows(), total=len(test_df.index)):
        P_row, T_row = np.zeros(len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS)), np.zeros(len(NLI_LABELS)*(1+MAX_DISAMBIGUATIONS))
        x = tokenizer(ex['premise'], ex['hypothesis'], return_tensors='pt').to('cuda')
        logits = model(**x).logits.cpu().detach().numpy().squeeze()
        pred = model.config.id2label[np.argmax(logits)]
        for pred_label in pred.split(', '):
            P_row[label2id[pred_label]] = 1
        true = [label2id[l] for l in ex['labels'].split(', ')] # true labels
        T_row[true] = 1

        results.append({
            'id': ex['id'], 
            'premise': ex['premise'], 
            'hypothesis': ex['hypothesis'], 
            'labels': ex['labels'],
            'prediction': pred
        })
        
        for j, disam in enumerate(ex['disambiguations']):
            x = tokenizer(disam['premise'], disam['hypothesis'], return_tensors='pt').to('cuda')
            logits = model(**x).logits.cpu().detach().numpy().squeeze()
            pred = model.config.id2label[np.argmax(logits)]
            for pred_label in pred.split(', '):
                P_row[len(NLI_LABELS)*(j+1)+label2id[pred_label]] = 1
            true = label2id[disam['label']]
            T_row[len(NLI_LABELS)*(j+1)+true] = 1
        
        P = np.concatenate((P, np.expand_dims(P_row, axis=0)))
        T = np.concatenate((T, np.expand_dims(T_row, axis=0)))

    f1 = f1_score(T[:,:len(NLI_LABELS)], P[:,:len(NLI_LABELS)], average='macro')
    em = np.mean([np.array_equal(T[i,:len(NLI_LABELS)], P[i,:len(NLI_LABELS)]) for i in range(T.shape[0])])
    group_em = np.mean([np.array_equal(T[i,:], P[i,:]) for i in range(T.shape[0])])

    res = {
        'f1': f1,
        'em': em,
        'group_em': group_em
    }
    
    out_dir = Path(f'results/multilabel/{model_path.split("/")[-2]}')
    ensure_dir(out_dir)
    pd.DataFrame(results).to_json(out_dir / 'predictions.jsonl', lines=True, orient='records')
    with open(out_dir / f'{model_path.split("/")[-1]}.json', 'w') as fo:
        json.dump(res, fo, indent=4)


if __name__ == '__main__':
    main()
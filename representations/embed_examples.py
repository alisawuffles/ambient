"""
Create a npy file containing [CLS] token embeddings for a given dataset of examples.
This is used for finding groups of nearest neighbors for prompting InstructGPT to generate unlabeled,
likely-ambiguous NLI examples (§2.2).
"""

import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import torch
import os
import click
from pathlib import Path
from utils.utils import ensure_dir


def get_cls_embedding(model, tokenizer, premise, hypothesis):
    x = tokenizer(premise, hypothesis, return_tensors='pt', max_length=128, truncation=True).to('cuda')
    outputs = model(**x, output_hidden_states=True)
    return outputs.hidden_states[-1][:,0,:]


@click.command()
@click.option('--model_path', type=str, help='path to trained model')
@click.option('--data_file', type=str, help='jsonl file of examples to embed')
@click.option('--dataset_name', type=str, help='for naming output npy file')
def main(model_path: str, data_file: str, dataset_name: str):
    model_path = Path(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    nli_model = RobertaForSequenceClassification.from_pretrained(model_path).to('cuda')
    
    output_dir = model_path / 'representations'
    ensure_dir(output_dir)
    data_df = pd.read_json(data_file, lines=True, orient='records')
    batch_size = 50000
    num_batches = np.ceil(len(data_df.index) / batch_size).astype('int')
    print(f'Total number of batches: {num_batches}')

    # split into batches so we don't run out of memory
    for batch_idx in range(num_batches):
        batch = data_df[batch_idx * batch_size : min(len(data_df.index), (batch_idx+1) * batch_size)]
        with torch.no_grad():
            vectors = []
            for _, row in tqdm(batch.iterrows(), total=len(batch.index), desc=f'Batch {batch_idx}'):
                v = get_cls_embedding(nli_model, tokenizer, row['premise'], row['hypothesis']).squeeze(0)
                vectors.append(v)

        vectors = torch.stack(vectors)
        vectors = torch.nn.functional.normalize(vectors)
        vectors = vectors.cpu().detach().numpy()
        
        # save to file
        with open(output_dir / f'{dataset_name}_{batch_idx}.npy', 'wb') as fo:
            np.save(fo, vectors)

    # combine batches
    vectors_per_batch = []
    for batch_idx in range(num_batches):
        with open(output_dir / f'{dataset_name}_{batch_idx}.npy', 'rb') as fin:
            vectors_per_batch.append(np.load(fin))

    mnli_vectors = np.concatenate(vectors_per_batch)
    with open(output_dir / f'{dataset_name}.npy', 'wb') as fo:
        np.save(fo, mnli_vectors)


if __name__ == '__main__':
    main()

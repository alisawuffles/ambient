"""
heuristic-based filtering
"""

import pandas as pd
import numpy as np
import click
from utils.utils import strip_punctuation_and_casing
import os
from pathlib import Path
from tqdm import tqdm
import json
from utils.constants import PRONOUNS
from utils.utils import strip_punctuation_and_casing

dangling_punctuation = ['"', "'", '(', ')']

forbidden_swaps = PRONOUNS + ['cat', 'dog', 'car', 'man', 'woman'] 

forbidden_phrases = [
    "Mary wants to try a little bit of every country's food on her trip around the world",
    "The novel is set in",
    'to go to the store',
    "After a long day of work",
    "I'm not going to lie",
    'The rock band played all night',
    'The rock concert was so loud',
    "I have a big project due tomorrow",
    'go to the party',
    "I'm going to the store.",
    'I am going to the store.',
    'The novel has been banned in several countries',
    'The novel has been banned in many countries',
    'shook her head',
    'shook his head',
    'shook its head',
    " might ",    # a bunch of weird examples with the word might
    'in a way',
    "I'm going to bed",
    'in a sense',
    "I can't decide whether",
    'said',    # exclude examples that are just about X saying Y
    '?',  # exclude all questions, since ambiguity in questions should be explored separately
]

forbidden_insertions = ['very']

forbidden_stems = [
    'It is generally agreed that',
    'In some cases',
    'They say that',
    'I think',
    'I think that',
    'The study found that',
    'In a sense,',
    'In some ways,',
    'It is important to remember that',
    'I would say that',
    'In my opinion,',
    'The findings of the study suggest that'
]


def clean(sentence):
    sentence = sentence.strip()
    if sentence.count('"') == 2 and sentence[0] == '"' and sentence[-1] == '"':
        sentence = sentence[1:-1]
    if sentence[0] == '(' and sentence[-1] == ')' and (sentence.count('(') + sentence.count(')')) == 2:
        sentence = sentence[1:-1]
    for p in dangling_punctuation:
        if sentence[0] == p and sentence.count(p) == 1:
            sentence = sentence[1:]
        elif sentence[-1] == p and sentence.count(p) == 1:
            sentence = sentence[:-1]
    return sentence


@click.command()
@click.option('--data_file', type=str, help='jsonl file of examples to clean (should be named examples.jsonl)')
@click.option('--seed_data_file', type=str, default='data/mnli/train.jsonl', help='jsonl of examples from original dataset')
def main(data_file: str, seed_data_file: str):
    gen_df = pd.read_json(data_file, lines=True)
    gen_df = gen_df.reset_index().rename({'index': 'id'}, axis=1)
    pre_dedup_len = len(gen_df.index)
    gen_df = gen_df.drop_duplicates(subset=['premise', 'hypothesis'], keep='first')
    num_duplicates = pre_dedup_len - len(gen_df.index)
    
    output_dir = Path(os.path.dirname(data_file))
    dataset_df = pd.read_json(seed_data_file, lines=True, orient='records')

    discards = {
        'short': [],                 # premise or hypothesis has fewer than 2 characters
        'copied_premise': [],        # premise == hypothesis
        'only_pronouns_differ': [],  # P and H only differ in pronouns
        'copied_nn': [],             # examples copied nearest neighbor
        'forbidden_phrase': []       # examples contain phrase from instructions
    }

    to_drop = []
    for idx, row in tqdm(gen_df.iterrows(), total=len(gen_df.index)):
        if row['premise'] is None or row['hypothesis'] is None:
            discards['short'].append(idx)
            continue
        premise, hypothesis = row['premise'].strip(), row['hypothesis'].strip()
        # 1. filter examples where premise or hypothesis is too short
        if min(len(premise), len(hypothesis)) < 5:
            discards['short'].append(idx)
            continue
        # 2. filter examples where hypothesis == premise, ignoring punctuation and casing
        if strip_punctuation_and_casing(premise) == strip_punctuation_and_casing(hypothesis):
            discards['copied_premise'].append(idx)
            continue
        # 3. filter examples that contain a redundant pattern
        if np.any([x in premise + hypothesis for x in forbidden_phrases]):
            discards['forbidden_phrase'].append(idx)
            continue
        if np.any([strip_punctuation_and_casing(premise) == strip_punctuation_and_casing(b+' '+hypothesis) for b in forbidden_stems]):
            discards['forbidden_phrase'].append(idx)
            continue
        # 4. filter examples where the example copies an in-context example
        copied_nn = False
        for nn in row['nearest_neighbors']:
            nn_row = dataset_df.loc[dataset_df['id'] == nn].iloc[0]
            nn_premise = nn_row['premise'].strip()
            nn_hypothesis = nn_row['hypothesis'].strip()
            if premise == nn_premise and hypothesis == nn_hypothesis:
                copied_nn = True
                break
        if copied_nn:
            discards['copied_nn'].append(idx)
            continue
        # 5. discard examples where P & H only differ by pronouns
        p_words = strip_punctuation_and_casing(premise).split(' ')
        h_words = strip_punctuation_and_casing(hypothesis).split(' ')
        if len(p_words) == len(h_words):
            diffs = [(w1, w2) for w1, w2 in zip(p_words, h_words) if w1 != w2]
            if np.all([(w1 in forbidden_swaps) and (w2 in forbidden_swaps) for w1, w2 in diffs]):
                discards['only_pronouns_differ'].append(idx)
        # clean examples to strip whitespaces and weird punctuation
        gen_df.at[idx, 'premise'] = clean(premise)
        gen_df.at[idx, 'hypothesis'] = clean(hypothesis)
    
    to_drop = [idx for sublist in discards.values() for idx in sublist]
    gen_df = gen_df.drop(to_drop)
    gen_df.to_json(output_dir / f'filtered_examples.jsonl', orient='records', lines=True)
    print(f'Filtered data written to {output_dir}/filtered_examples.jsonl')
    
    for k, v in discards.items():
        print(f'{k}\t\t{len(v)}')
    print(f'duplicates\t\t{num_duplicates}')


if __name__ == "__main__":
    main()
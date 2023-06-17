"""
Utils for reading batches from linguist annotations collected in §2.3
"""
from pathlib import Path
import pandas as pd
from utils.constants import id2label
import os
import csv
import math
from collections import defaultdict
import numpy as np
from datetime import datetime
import locale
locale.setlocale(locale.LC_TIME, 'en_US')

researcher_ids = {
    'alisa': 'MY_AMT_ID'
}

time_format = '%a %b %d %H:%M:%S %Z %Y'

keep_columns = (['WorkerId', 'SubmitTime', 'Answer.ee', 'Input.id', 'Input.premise', 'Input.hypothesis', 'Answer.feedback']
    + [f'Answer.premise{i}' for i in range(1,5)]
    + [f'Answer.hypothesis{i}' for i in range(1,5)])


def check_nan(gold):
    if type(gold) == str:
        return gold == 'nan'
    if type(gold) == float:
        return math.isnan(gold)


def read_batch(batch_id: int, batch_dir='annotation/validation/batches'):
    """
    - read in batch results file
    - make label annotations readable (e.g., 1 -> entailment)
    - keep a subset of relevant columns
    - exclude examples with invalid annotations or from disqualified workers
    
    return batch_df, a cleaned version of the batch results
    """
    def textify(gold_string):
        gold_string = gold_string.replace('.0', '')
        for label_id, label in id2label.items():
            gold_string = gold_string.replace(str(label_id), label)
        return gold_string
    
    batch_dir = Path(f'{batch_dir}/batch_{batch_id}')
    batch_df = pd.read_csv(batch_dir / f'Batch_{batch_id}_batch_results.csv')

    # convert label annotations from integers to text
    gold_columns = batch_df.columns[batch_df.columns.str.contains('_gold')].tolist()
    for gold_col in gold_columns:
        batch_df[gold_col] = batch_df[gold_col].astype(str)
        batch_df[gold_col] = batch_df[gold_col].apply(textify)

    # keep a subset of columns, and rename columns
    batch_df = batch_df[keep_columns + gold_columns]
    column_rename_map = {c: c.split('.')[-1] for c in batch_df.columns if (c.startswith('Answer.') or c.startswith('Input.'))}
    column_rename_map.update({'WorkerId': 'worker_id', 'SubmitTime': 'submit_time', 'Answer.ee': 'time_on_page'})
    batch_df.rename(columns=column_rename_map, inplace=True)

    # exclude examples from disqualified workers
    if os.path.exists(batch_dir/'not_qualified.csv'):
        unqualified_workers = [e[0] for e in list(csv.reader(open(batch_dir/'not_qualified.csv')))]
        batch_df = batch_df.loc[~batch_df['worker_id'].isin(unqualified_workers)]
    
    # keep only rows in balanced_df.jsonl (filtering logic may have changed)
    data_ids = pd.read_json('AmbiEnt/generated_data/balanced_examples.jsonl', lines=True).id.tolist()
    batch_df = batch_df.loc[batch_df['id'].isin(data_ids)]

    # discard rows that do not meet validation
    def check_valid(row):
        premise, hypothesis = row['premise'], row['hypothesis']
        if '|' in row['q0_gold']:
            num_premise_revisions = 0
            num_hypothesis_revisions = 0
            for i in range(1, 5):
                if f'q{i}_gold' in row and not check_nan(row[f'q{i}_gold']):
                    if row[f'premise{i}'] != premise:
                        num_premise_revisions += 1
                    if row[f'hypothesis{i}'] != hypothesis:
                        num_hypothesis_revisions += 1
            if num_premise_revisions == 1 or num_hypothesis_revisions == 1:
                return False
        return True
    
    batch_df = batch_df[batch_df.apply(check_valid, axis=1)]
    
    # create timestamp
    batch_df['submit_time'] = [datetime.strptime(t, time_format) for t in batch_df['submit_time'].tolist()]

    return batch_df


def clean_annotation_batch(batch_df):
    """
    return cleaned_df, where each row is a P/H example (not an annotation) by combining two linguist annotations
    discard examples that either annotator marked for discard
    """

    # combine annotations for each example    
    example_rows = []
    for id, example_df in batch_df.groupby('id'):
        annotations = example_df['q0_gold'].tolist()
        
        # discard examples that one or more annotators marked for discard
        if 'discard' in ' '.join(annotations):
            continue
        
        worker_ids = example_df['worker_id'].tolist()
        example_id, premise, hypothesis = id, example_df.iloc[0].premise, example_df.iloc[0].hypothesis
        
        rewrites = defaultdict(list)
        for _, row in example_df.iterrows():
            if '|' in row['q0_gold']:
                for i in range(1, 5):
                    if f'q{i}_gold' in row and not check_nan(row[f'q{i}_gold']):
                        rewrites[row[f'q{i}_gold']].append({
                            'premise': row[f'premise{i}'],
                            'hypothesis': row[f'hypothesis{i}']
                        })
                    
        example_rows.append({
            'id': example_id,
            'worker_ids': worker_ids,
            'premise': premise,
            'hypothesis': hypothesis,
            'annotations': annotations,
            'disambiguations': rewrites
        })

    return pd.DataFrame(example_rows)


def clean_validation_batch(df):
    # for duplicate examples annotated multiple times in the inter-annotator batch, keep examples annotated by me
    df['alisa'] = df['worker_id'] == researcher_ids['alisa']
    df = df.sort_values(['id', 'alisa'], ascending=False).drop_duplicates(subset='id')
    df = df.drop(columns='alisa')
    df = df.loc[df['q0_gold'] != 'discard']
    
    example_rows = []
    for i, row in df.iterrows():
        rewrites = defaultdict(list)
        if '|' in row['q0_gold']:
            for i in range(1, 5):
                if f'q{i}_gold' in row and not check_nan(row[f'q{i}_gold']):
                    rewrites[row[f'q{i}_gold']].append({
                        'premise': row[f'premise{i}'],
                        'hypothesis': row[f'hypothesis{i}']
                    })
        example_rows.append({
            'id': row['id'],
            'validator_id': row['worker_id'],
            'premise': row['premise'],
            'hypothesis': row['hypothesis'],
            'gold': row['q0_gold'],
            'disambiguations': rewrites
        })
        
    return pd.DataFrame(example_rows)


def statistics_for_worker(batch_df, worker_id):
    person_df = batch_df.loc[batch_df['worker_id'] == worker_id]
    prop_ambiguous = sum(person_df['q0_gold'].str.contains('\|'))/len(person_df)
    prop_discard = sum(person_df['q0_gold'].eq('discard'))/len(person_df)
    
    return {
        'num_examples': len(person_df),
        'median_time': person_df['time_on_page'].quantile(q=0.50),
        'prop_ambiguous': np.round(prop_ambiguous, 3),
        'prop_discard': np.round(prop_discard, 3)
    }

"""
Utils for reading batches from crowdworker experiments (§3)
"""

import os
import numpy as np
from pathlib import Path
import csv
import pandas as pd
from utils.constants import id2label
import ast
from datetime import datetime
import locale
locale.setlocale(locale.LC_TIME, 'en_US')

time_format = '%a %b %d %H:%M:%S %Z %Y'


def read_batch(batch_id: int, batch_dir='annotation/crowdworker_exp/batches'):
    batch_dir = Path(batch_dir)

    def textify_labels(gold_string):
        return id2label[int(gold_string.replace('.0', ''))]

    def textify_disambiguation(answer_string):
        assert answer_string in ['1', '2']
        return True if answer_string == '1' else False
    
    batch_dir = batch_dir / f'batch_{batch_id}'
    batch_df = pd.read_csv( batch_dir / f'Batch_{batch_id}_batch_results.csv')

    for gold_col in [f'Answer.q{i}_gold' for i in range(4)]:
        batch_df[gold_col] = batch_df[gold_col].astype(str).apply(textify_labels)

    for disam_col in [f'Answer.d{i}_gold' for i in range(1,4)]:
        batch_df[disam_col] = batch_df[disam_col].astype(str).apply(textify_disambiguation)
    
    batch_df['Answer.ee'] = batch_df['Answer.ee'].astype(float)
    keep_columns = (['WorkerId', 'SubmitTime', 'Input.id', 'Input.premise', 'Input.hypothesis', 
                    'Input.labels', 'Input.ambiguous_sent', 'Input.distractor_idx', 'Input.ambiguous_sent_html']
                    + batch_df.columns[batch_df.columns.str.contains('Answer.')].tolist()
                    + batch_df.columns[batch_df.columns.str.contains('interpretation')].tolist())
    batch_df = batch_df[keep_columns]
    column_rename_map = {c: c.split('.')[-1] for c in batch_df.columns if (c.startswith('Answer.') or c.startswith('Input.'))}
    column_rename_map.update({
        'WorkerId': 'worker_id', 
        'SubmitTime': 'submit_time',
        'Answer.ee': 'time_on_page'})
    batch_df.rename(columns=column_rename_map, inplace=True)

    # exclude examples that got revised later
    if os.path.exists(batch_dir / 'fixed_examples.txt'):
        fixed_example_ids = [e[0] for e in list(csv.reader(open(batch_dir / 'fixed_examples.txt')))]
        if batch_df.dtypes['id'] == int:
            fixed_example_ids = [int(e) for e in fixed_example_ids]
        batch_df = batch_df.loc[~batch_df['id'].isin(fixed_example_ids)]

    # create timestamp
    batch_df['submit_time'] = [datetime.strptime(t, time_format) for t in batch_df['submit_time'].tolist()]

    batch_df['ambiguous_sent'] = ['premise' if 'premise' in x else 'hypothesis' for x in batch_df['ambiguous_sent_html']]
    batch_df.drop('ambiguous_sent_html', axis=1, inplace=True)

    return batch_df


def clean_batch(batch_df):
    example_rows = []
    for id, example_df in batch_df.groupby('id'):
        dummy_row = example_df.iloc[0]
        ex = {'id': id}
        for col in ['premise', 'hypothesis', 'ambiguous_sent', 'distractor_idx', 'labels']:
            ex[col] = dummy_row[col]
        ex['interpretations'] = dummy_row[[f'interpretation{i}' for i in range(1,4)]].tolist()
        for col in [f'q{i}_gold' for i in range(4)] + [f'd{i}_gold' for i in range(1,4)]:
            ex[col] = example_df[col].tolist()
        example_rows.append(ex)
    
    return pd.DataFrame(example_rows)

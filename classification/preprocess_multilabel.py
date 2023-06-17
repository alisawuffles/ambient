import pandas as pd
from utils.utils import example_untouched
from tqdm import tqdm


split = 'train'
wanli_df = pd.read_json(f'../wanli/data/wanli/{split}.jsonl', lines=True)
wanli_df['pairID'] = wanli_df['pairID'].astype(str)
wanli_ids = wanli_df.id.tolist()
annotations_df = pd.read_json('../wanli/data/wanli/anonymized_annotations.jsonl', lines=True)


# construct multilabel data, where each example can have multiple labels
multilabel_data = []
for train_id in tqdm(wanli_ids):
    sub_df = annotations_df.loc[annotations_df['id'] == train_id]
    row1, row2 = sub_df.iloc[0].to_dict(), sub_df.iloc[1].to_dict()
    if example_untouched(row1) and example_untouched(row2):
        gold1, gold2 = row1['gold'], row2['gold']
        gold = set([gold1, gold2])
        row1['gold'] = gold
        multilabel_data.append({
            'id': row1['id'],
            'premise': row1['premise'],
            'hypothesis': row1['hypothesis'],
            'gold': ', '.join(sorted(list(gold))),   # sort for set classification case
            'genre': 'generated',
            'pairID': row1['nearest_neighbors'][0]
        }) 
    elif not example_untouched(row1):  # if row1 is revised, add row2
        multilabel_data.append({
            'id': row2['id'],
            'premise': row2['revised_premise'],
            'hypothesis': row2['revised_hypothesis'],
            'gold': row2['gold'],
            'genre': 'generated_revised' if row2['revised'] else 'generated',
            'pairID': row2['nearest_neighbors'][0]
        })
    elif not example_untouched(row2):  # if row1 is unrevised and row2 is revised, add row1
        multilabel_data.append({
            'id': row1['id'],
            'premise': row1['revised_premise'],
            'hypothesis': row1['revised_hypothesis'],
            'gold': row1['gold'],
            'genre': 'generated_revised' if row1['revised'] else 'generated',
            'pairID': row1['nearest_neighbors'][0]
        })
    else:
        print("something's wrong!")
        pass

pd.DataFrame(multilabel_data).to_json(f'data/wanli/multilabel_{split}.jsonl', lines=True, orient='records')
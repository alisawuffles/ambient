def get_disambiguation_idxs(row):
    return [j for j, label in enumerate(row['labels']) if label]
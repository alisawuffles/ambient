def create_test_instances(test_df):
    test_instances = []
    for i, row in test_df.iterrows():
        for sentence_key in ['premise', 'hypothesis']:
            if row[f'{sentence_key}_ambiguous']:
                test_instances.append({
                    'id': row['id'],
                    'ambiguous_sentence_key': sentence_key,
                    'ambiguous_sentence': row[sentence_key],
                    'disambiguations': list(set([l[sentence_key] for l in row['disambiguations']])),
                })
    return test_instances

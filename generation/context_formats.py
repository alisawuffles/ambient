import pandas as pd

instruction = 'Write pairs of sentences that are related to each other in the same way.'
def format_incontext_examples(examples: pd.DataFrame):
    """
    format a given group of examples as context for GPT-3, using template for the given label
    """
    examples = examples[::-1]
    context_string = f'{instruction}\n\n'
    # write in context examples
    for i, (_, row) in enumerate(examples.iterrows()):
        # for every chunk_size examples, repeat instructions and enumerate starting from 1
        context_string += f'Sentence 1: {row["premise"]}\nSentence 2: {row["hypothesis"]}\n\n'

    # write final numbering and premise, if provided
    context_string += f'Sentence 1:'

    return context_string

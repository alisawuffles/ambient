import os
import json
import click
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy import spatial
from generation.context_formats import format_incontext_examples
from generation.gpt3_generation import request
from utils.utils import ensure_dir


@click.command()
@click.option('--model_path', type=str, help='contains pre-computed representations and training dynamics')
@click.option('--num_incontext_examples', default=5, type=int)
@click.option('--engine', type=str, default='ada', help='GPT-3 model for generation')
@click.option('--top_p', type=float, default=0.5, help='for nucleus sampling')
@click.option('--num_gens_per_prompt', default=5, type=int, help='')
@click.option('--num_examples', default=10, type=int, 
    help='total number of generated examples desired, including previously generated examples, -1 for using all seeds'
)
def main(
    model_path: str,
    num_incontext_examples: int,
    engine: str,
    top_p: float,
    num_gens_per_prompt: int, 
    num_examples: int,
):
    if engine.startswith('text'):
        engine_name = '-'.join(engine.split('-')[1:])
    
    output_dir = Path(f'generated_data/wanli_disagreement_p{top_p}_{engine_name}')
    model_path = Path(model_path)
    ensure_dir(output_dir)

    generated_examples = []

    # pre-computed embeddings of training examples
    with open(model_path / 'representations/wanli.npy', 'rb') as fin:
        wanli_vectors = np.load(fin)
        tree = spatial.KDTree(wanli_vectors)
    
    # load pool of WANLI data
    train_df = pd.read_json('data/wanli/train.jsonl', lines=True, orient='records')
    ambiguous_train = pd.read_json('data/wanli/subsets/train_disagreements.jsonl', lines=True)

    # write output continuously and flush periodically
    examples_fo = open(output_dir / 'examples.jsonl', 'w')
    lines_per_flush = 100

    # generate examples!
    tot = num_examples//num_gens_per_prompt if num_examples > 0 else len(ambiguous_train.index)
    for _, row in tqdm(ambiguous_train.iterrows(), total=tot):
        i = train_df.loc[train_df['id'] == row['id']].index[0]  # get corresponding row index (NOT example id) in train_df
        # get nearest neighbors
        embedding = wanli_vectors[i,:]
        neighbor_ids = tree.query(embedding, k=num_incontext_examples)[1]
        neighbors_df = train_df.loc[neighbor_ids]
        
        context_string = format_incontext_examples(neighbors_df)
        # write an example context to files
        if not os.path.exists(output_dir / f'sample_context.txt'):
            with open(output_dir / f'sample_context.txt', 'w') as template_fo:
                template_fo.write(context_string)
        
        for i in range(num_gens_per_prompt):
            generation = request(
                context_string, 
                model=engine,
                max_tokens=120,
                top_p=top_p,
                stop='\n\n',
                return_only_text=True,
            )

            try:
                premise, hypothesis = generation.split('\nSentence 2: ')
            except ValueError:
                continue
            generated_ex = {
                'premise': premise,
                'hypothesis': hypothesis,
                'nearest_neighbors': neighbors_df.id.tolist()
            }
            generated_examples.append(generated_ex)
            # write output
            examples_fo.write(json.dumps(generated_ex, default=str) + '\n')
            if len(generated_examples) % lines_per_flush == 0:
                examples_fo.flush()

        if num_examples > 0 and len(generated_examples) >= num_examples:
            break
    
    examples_fo.close()
    
    with open(output_dir / 'examples.json', 'w') as fo:
        json.dump(generated_examples, fo, indent=4)


if __name__ == "__main__":
    main()
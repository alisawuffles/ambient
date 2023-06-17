"""
can't evaluate ChatCompletion-based OpenAI models (ChatGPT, GPT-4) because they don't provide logprobs
"""

import torch
import json
import pandas as pd
from evaluation.pmi import cross_entropy
from evaluation.model_utils import get_model_class, load_model, is_instruct_model
from evaluation.model_utils import GPT3_BATCH_SIZE, LLAMA_BATCH_SIZE, FLAN_BATCH_SIZE
from generation.gpt3_generation import request
import random
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.utils import ensure_dir
from pathlib import Path
from evaluation.distractors import create_distractor
from evaluation.utils import create_test_instances
import numpy as np
import click
from datetime import date

BATCH_SIZES = {'gpt3': GPT3_BATCH_SIZE, 'llama': LLAMA_BATCH_SIZE, 'flan': FLAN_BATCH_SIZE}


def save_example_results(ambiguous_sent: str, continuation_stats: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    fig, ax = plt.subplots()
    for i, (d_key, stats) in enumerate(continuation_stats.items()):
        stat_df = pd.DataFrame(stats)
        # save continuations to jsonl file
        stat_df.to_json(
            out_dir / f'{d_key}.jsonl', 
            lines=True, 
            orient='records',
        )
        # add histogram to plot
        sns.histplot(
            data=stat_df, 
            x='log_odds', 
            color=sns.color_palette('muted')[i], 
            label=d_key, 
            kde=True, 
            stat='density'
        )

    ax.legend()
    ax.set_title(ambiguous_sent)
    ax.set_xlabel('log (c|y_i) - log (c|x)')
    plt.tight_layout()
    plt.savefig(out_dir / 'hist.png', dpi=300)
    plt.close()


def generate_continuations(prompt, model_name, top_p, num_generations, model=None, tokenizer=None):
    model_class = get_model_class(model_name)
    model_batch_size = BATCH_SIZES[model_class]
    batch_sizes = [model_batch_size for _ in range(num_generations//model_batch_size)] + [num_generations % model_batch_size]
    if 0 in batch_sizes: batch_sizes.remove(0)

    if model_class != 'gpt3':
        generation_kwargs = {'do_sample': True, 'max_new_tokens': 60, 'top_p': top_p}

    conts = []
    for batch_size in batch_sizes:
        if model_class == 'gpt3':
            gpt3_gens = request(
                prompt=prompt,  
                top_p=top_p,
                model=model_name,
                n=batch_size,
                stop=['.', '!', '?'],
                logit_bias={50256: -100, 1: -100},
                return_only_text=True,
            )
            gpt3_gens = [g+'.' for g in gpt3_gens]      # add back the stop token
            conts.extend(gpt3_gens)
        elif model_class == 'flan':
            # batch size = 1
            with torch.inference_mode():
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
                generate_ids = model.generate(input_ids, num_return_sequences=batch_size, **generation_kwargs)
                gens = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                gens = [l.replace('<pad>', '').split('.')[0]+'.' for l in gens]
                conts.extend(gens)
        elif model_class == 'llama':
            # batch size = 1
            with torch.inference_mode():
                input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
                full_ids = model.generate(input_ids, num_return_sequences=batch_size, **generation_kwargs)
                repeat_input_ids = input_ids.repeat(full_ids.shape[0], 1)
                generate_ids = [out[len(inp):] for inp, out in zip(repeat_input_ids, full_ids)]
                gens = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                gens = [l.split('.')[0]+'.' for l in gens]    # keep only the first sentence
                conts.extend(gens)
    
    return conts


def continuation_evaluation(
    test_df,
    model_name='davinci', 
    top_p=0.8,
    num_generations=1000,
):
    test_df = test_df[test_df['premise_ambiguous'] | test_df['hypothesis_ambiguous']]
    test_instances = create_test_instances(test_df)
    print(f'Number of test instances: {len(test_instances)}')
    
    model_kwargs = load_model(model_name)
    # NOTE: appending open quotation mark for non-instruct models leads to better topicality
    stem = 'Write a story.\n\n' if is_instruct_model(model_name) else '"'

    # output files
    out_dir = Path(f'results/continuation_evaluation/{model_name}')
    ensure_dir(out_dir)
    out_file = out_dir / 'results.jsonl'
    results_fo = open(out_file, 'w')

    results = []
    for row in tqdm(test_instances, total=len(test_instances)):
        ambiguous_sentence = row['ambiguous_sentence']

        disambiguations = [stem + d for d in row['disambiguations']]
        distractor = stem + create_distractor(ambiguous_sentence)
        ambiguous_sentence = stem + ambiguous_sentence
        disambiguations_dict = {f'y{i}': d for i, d in enumerate(disambiguations)}
        disambiguations_dict['d'] = distractor

        # generate continuations
        generated_continuations = {}
        for d_key in disambiguations_dict:
            conts = generate_continuations(disambiguations_dict[d_key], model_name, top_p, num_generations, **model_kwargs)
            generated_continuations[d_key] = conts

        # clean up continuations
        min_conts = num_generations
        for d_key, continuations in generated_continuations.items():
            non_empty_conts = [c for c in continuations if len(c)>1]    # exclude empty strings and end quotes
            generated_continuations[d_key] = non_empty_conts
            if len(non_empty_conts) < min_conts:
                min_conts = len(non_empty_conts)

        # downsample so that each set of continuations is the same size
        for d_key, continuations in generated_continuations.items():
            generated_continuations[d_key] = random.sample(continuations, min_conts)
        
        # compute cross entropies!
        continuation_stats = {}
        for d_key, conts in generated_continuations.items():
            # calculate CE given appropriate disambiguation, i.e., -log p(c|y_i)
            disambiguation = disambiguations_dict[d_key]
            ce_list_cond = cross_entropy([disambiguation]*min_conts, conts, model_name=model_name, **model_kwargs)

            # calculate CE given ambiguous prompt, i.e., -log p(c|x)
            ce_list_ambig = cross_entropy([ambiguous_sentence]*min_conts, conts, model_name=model_name, **model_kwargs)
            
            assert len(conts) == len(ce_list_cond) == len(ce_list_ambig)
            continuation_stats[d_key] = [
                {'continuation': c, 'ce_cond': x, 'ce_ambig': y, 'log_odds': y-x}
                for c,x,y in zip(conts, ce_list_cond, ce_list_ambig)
            ]
        
        # save results
        save_example_results(
            ambiguous_sentence, continuation_stats, 
            out_dir=out_dir / f'example_dirs/{row["id"]}'
        )

        # write row to full results
        ex = {'id': row['id'], 'ambiguous_sentence': ambiguous_sentence, 'options': {}, 'num_conts': min_conts}

        for d_key, stats in continuation_stats.items():
            ex['options'][d_key] = {
                'sentence': disambiguations_dict[d_key],
                'KL_div': np.mean([s['log_odds'] for s in stats]),
            }

        results_fo.write(json.dumps(ex, default=str) + '\n')
        results_fo.flush()
        results.append(ex)


@click.command()
@click.option('--data_path', type=str, default='annotation/AmbiEnt/test.jsonl')
@click.option('--model_name', type=str, default='davinci')
@click.option('--top_p', type=float, default=0.8)
@click.option('--num_generations', type=int, default=1000)
def main(data_path: str, model_name: str, top_p: float, num_generations: int):
    test_df = pd.read_json(data_path, lines=True)
    continuation_evaluation(test_df, model_name=model_name, top_p=top_p, num_generations=num_generations)


if __name__ == '__main__':
    main()
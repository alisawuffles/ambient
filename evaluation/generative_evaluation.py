"""
evaluate generation of disambiguating rewrites
"""
import torch
from evaluation.edit_f1 import get_edit_f1
from evaluation.model_utils import load_model, get_model_class
from generation.gpt3_generation import request
from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import click

template = {
    'premise': 
        {'instruction': ("In each example, you will be given some context and a claim, where the correctness"
                         " of the claim is affected by some ambiguity in the context. Enumerate two or three"
                         " interpretations of the context that lead to different judgments about the claim."),
         'bridge': "We don\'t know, because the context can be interpreted in many different ways:\n",
        },
    'hypothesis': 
        {'instruction': ("In each example, you will be given some context and a claim. Unfortunately, the"
                         " claim has some ambiguity that affects whether it is correct. Enumerate two or"
                         " three interpretations of the claim that lead to different judgments about its correctness."),
         'bridge': "We don\'t know, because the claim can be interpreted in many different ways:\n",
        }
}

label_verbalizer = {
    'entailment': 'Then the claim is true.',
    'neutral': 'Then the claim is inconclusive.',
    'contradiction': 'Then the claim is false.'
}

question = 'Given the context alone, is this claim true, false, or inconclusive?\n'

def _premise_ambiguous(row):
    """
    returns True if the premise is ambiguous, False otherwise
    """
    return row['premise_ambiguous']


def generate_disambiguations(prompt, model_name, model=None, tokenizer=None):
    model_class = get_model_class(model_name)
    if model_class == 'gpt3':
        generation = request(prompt, model=model_name, stop='\n\n', temperature=0, return_only_text=True)
    elif model_class == 'chat':
        generation = request(
            messages=[{'role': 'user', 'content': prompt}],
            model=model_name,
            stop='\n\n',
            max_tokens=200,
            temperature=0,
            return_only_text=True
        )
    elif model_class == 'flan':
        with torch.inference_mode():
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
            generate_ids = model.generate(input_ids, max_new_tokens=200)
            generation = tokenizer.batch_decode(generate_ids, skip_special_token=True, clean_up_tokenization_spaces=False)[0]
            generation = generation.replace('<pad>', '').replace('</s>', '').strip()
    elif model_class == 'llama':
        with torch.inference_mode():
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
            full_ids = model.generate(input_ids, max_new_tokens=200)
            generate_ids = [out[len(inp):] for inp, out in zip(input_ids, full_ids)]
            generation = tokenizer.batch_decode(generate_ids, skip_special_token=True, clean_up_tokenization_spaces=False)[0]
            generation = generation.split('\n\n')[0]

    return generation


def generation_to_lines(gen):
    gen = gen.strip()
    if gen[:3] != "1. ":  # if the first line is not a disambiguation
        return []
    lines = gen[3:].replace("\n", "").split("2. ")
    if len(lines) >= 2:
        assert len(lines) == 2
        lines[1:] = lines[1].split("3. ")
        assert len(lines) <= 3
    
    lines = [l.strip() for l in lines]
    return lines


def generative_evaluation(test_df, model_name='ada', num_incontext_examples=5):
    test_df = test_df[test_df['premise_ambiguous'] ^ test_df['hypothesis_ambiguous']]   # XOR
    model_kwargs = load_model(model_name)

    results = []

    for i, row in tqdm(test_df.iterrows(), total=len(test_df.index)):
        premise_ambiguous = _premise_ambiguous(row)
        ambiguous_sentence_key = 'premise' if premise_ambiguous else 'hypothesis'    

        # sample in-context examples
        mask = test_df.apply(_premise_ambiguous, axis=1) == premise_ambiguous
        in_context_pool = test_df[mask].drop(i)
        in_context_examples = in_context_pool.sample(min(num_incontext_examples, len(in_context_pool.index)))

        prompt = template[ambiguous_sentence_key]['instruction'] + '\n\n'
        bridge = template[ambiguous_sentence_key]['bridge']

        # create prompt
        for _, ex_row in in_context_examples.iterrows():
            prompt += f'Context: {ex_row["premise"]}\nClaim: {ex_row["hypothesis"]} {question}{bridge}'
            for j, disambiguation in enumerate(ex_row['disambiguations']):
                l = disambiguation['label']
                prompt += f'{j+1}. {disambiguation[ambiguous_sentence_key]} {label_verbalizer[l]}\n'

            prompt += '\n'
        prompt += f'Context: {row["premise"]}\nClaim: {row["hypothesis"]} {question}{bridge}'

        # generate disambiguations
        generation = generate_disambiguations(prompt, model_name, **model_kwargs)
        generated_lines = generation_to_lines(generation)
        
        pred_rewrites = {}
        for line in generated_lines:
            for label, verbalizer in label_verbalizer.items():
                if verbalizer in line:
                    pred_rewrites[label] = line.replace(verbalizer, '').strip()
        
        # calculate Edit F1 score
        gold_rewrites = {
            d['label']: d[ambiguous_sentence_key] for d in row['disambiguations']
        }

        edit_f1s = []
        for label in gold_rewrites:
            if label in pred_rewrites:
                edit_f1s.append(get_edit_f1(row[ambiguous_sentence_key], gold_rewrites[label], pred_rewrites[label]))
            else:
                edit_f1s.append(0)
    
        ## record results
        results.append({
            'id': row['id'],
            'premise': row['premise'],
            'hypothesis': row['hypothesis'],
            'premise_ambiguous': row['premise_ambiguous'],
            'hypothesis_ambiguous': row['hypothesis_ambiguous'],
            'prompt': prompt,
            'generation': generation,
            'gold_rewrites': gold_rewrites,
            'predicted_rewrites': pred_rewrites,
            'edit_f1': np.mean(edit_f1s)
        })
    
    eval_df = pd.DataFrame(results)
    eval_df.to_json(f'results/generative_evaluation/{model_name}-n{num_incontext_examples}.jsonl', lines=True, orient='records')

    return {
        'edit_f1_mean': eval_df.edit_f1.mean(),
        'edit_f1_median': eval_df.edit_f1.median(),
    }


@click.command()
@click.option('--data_path', type=str, default='annotation/AmbiEnt/validated_examples.jsonl')
@click.option('--model_name', type=str, default='davinci')
@click.option('--num_incontext_examples', type=int, default=4)
def main(data_path: str, model_name: str, num_incontext_examples: int):
    test_df = pd.read_json(data_path, lines=True)
    generative_evaluation(test_df, model_name=model_name, num_incontext_examples=num_incontext_examples)


if __name__ == '__main__':
    main()
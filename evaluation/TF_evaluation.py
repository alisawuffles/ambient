import pandas as pd
from generation.gpt3_generation import request
from evaluation.model_utils import get_model_class, load_model, GPT3_BATCH_SIZE, FLAN_BATCH_SIZE, LLAMA_BATCH_SIZE
from evaluation.utils import create_test_instances
from evaluation.TF_utils import gpt3_logprobs_to_TF, chat_token_to_TF, logits_to_TF
from tqdm import tqdm
import click
import random
from datetime import date


templates = {
    'This may mean:': True,
    'This does not necessarily mean:': True,
    'This cannot mean:': False,
    'This can only mean:': False,
}


def TF_query(model_name: str, prompts: list, model=None, tokenizer=None):
    print('Generating output...')
    model_class = get_model_class(model_name)
            
    if model_class == 'gpt3':
        inputs = prompts
        outputs, prob_mass = [], []
        with tqdm(total=len(prompts)) as pbar:
            while len(inputs) > 0:
                result = request(
                    prompt=inputs[:GPT3_BATCH_SIZE],
                    model=model_name,
                    max_tokens=1,
                    logprobs=100,
                    temperature=0,
                    logit_bias={"50256": -100}    # suppress <|endoftext|>
                )
                logprobs_list = [choice['logprobs']['top_logprobs'][0] for choice in result['choices']]     # logprobs for first token
                o, p = gpt3_logprobs_to_TF(logprobs_list)
                outputs += o
                prob_mass += p
                inputs = inputs[GPT3_BATCH_SIZE:]
                pbar.update(GPT3_BATCH_SIZE)
    
    elif model_class in ['chat', 'gpt4']:
        outputs, prob_mass = [], [None]*len(prompts)
        for prompt in tqdm(prompts):
            next_token = request(
                messages=[{'role': 'user', 'content': prompt}],
                model=model_name,
                max_tokens=1,
                temperature=0,
                logit_bias={"50256": -100},    # suppress <|endoftext|>
                return_only_text=True,
            )
            outputs.append(chat_token_to_TF(next_token.strip()))

    elif model_class == 'flan':
        remaining_prompts = prompts
        outputs, prob_mass = [], []
        with tqdm(total=len(prompts)) as pbar:
            while len(remaining_prompts) > 0:
                inputs = tokenizer.batch_encode_plus(remaining_prompts[:FLAN_BATCH_SIZE], return_tensors='pt', padding=True)
                input_ids = inputs.input_ids.to('cuda')
                attention_mask = inputs.attention_mask.to('cuda')
                decoder_input_ids = model._prepare_decoder_input_ids_for_generation(input_ids.size(0))
                logits = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids).logits[:,-1,:]
                o, p = logits_to_TF(logits, model_class)
                outputs += o
                prob_mass += p
                remaining_prompts = remaining_prompts[FLAN_BATCH_SIZE:]
                pbar.update(FLAN_BATCH_SIZE)
    
    elif model_class == 'llama':
        outputs, prob_mass = [], []
        for prompt in tqdm(prompts):
            inputs = tokenizer(prompt, return_tensors='pt', padding=True)
            input_ids = inputs.input_ids.to('cuda')
            attention_mask = inputs.attention_mask.to('cuda')
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:,-1,:]
            o, p = logits_to_TF(logits, model_class)
            outputs += o
            prob_mass += p

    return outputs, prob_mass


@click.command()
@click.option('--data_path', type=str, default='annotation/AmbiEnt/test.jsonl')
@click.option('--model_name', type=str, default='davinci')
def main(data_path, model_name):
    test_df = pd.read_json(data_path, lines=True)
    test_df = test_df[test_df['premise_ambiguous'] | test_df['hypothesis_ambiguous']]
    test_instances = create_test_instances(test_df)
    print(f'Number of test instances: {len(test_instances)}')

    test_examples = []
    for row in test_instances:
        for disambiguation in row['disambiguations']:
            for template_id, (template, answer) in enumerate(templates.items()):
                prompt = f"Q: {row['ambiguous_sentence']} {template} {disambiguation} True or False?\nA:"
                test_examples.append({
                    'example_id': row['id'],
                    'ambiguous_sentence_key': row['ambiguous_sentence_key'],
                    'disambiguation': disambiguation,
                    'prompt': prompt,
                    'template_id': template_id,
                    'answer': answer,
                })

    prompts = [e['prompt'] for e in test_examples]
    print(random.sample(prompts, 1)[0])
    
    model_kwargs = load_model(model_name)
    outputs, prob_mass = TF_query(model_name, prompts, **model_kwargs)

    print('Computing accuracy')

    results = []

    for ex, output, mass in zip(test_examples, outputs, prob_mass):
        ex['prediction'] = output
        ex['TF_prob_mass'] = mass
        results.append(ex)
    
    results_df = pd.DataFrame(results)
    acc = (results_df.prediction == results_df.answer).sum()/len(results_df.index)
    print(f'Accuracy: {acc}')
    print(f'Average probability mass of True, False tokens: {results_df.TF_prob_mass.mean()}')
    today = date.today().strftime('%m-%d')
    results_df.to_json(f'results/TF_evaluation/{model_name}-{today}.jsonl', lines=True, orient='records')


if __name__ == '__main__':
    main()
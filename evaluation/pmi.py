from generation.gpt3_generation import request
from evaluation.model_utils import GPT3_BATCH_SIZE
import random
import torch.nn.functional as F

def cross_entropy(inputs, options, model_name=None, model=None, tokenizer=None, verbose=False):
    """
    get a list of -log P(target|inp) for the inputs and options in inputs, options
    """
    if not model and not tokenizer:
        return gpt3_cross_entropy(inputs, options, model_name, verbose)
    elif model and tokenizer:
        return lm_cross_entropy(inputs, options, model, tokenizer)
    else:
        raise ValueError('Must provide either GPT3 engine or model/tokenizer')


def lm_cross_entropy(inputs, options, model, tokenizer):
    """
    for HF models
    """
    ce_list = []
    for inp, out in zip(inputs, options):
        prompt_input_ids = tokenizer.encode(inp, return_tensors='pt').to(model.device)

        if model.config.is_encoder_decoder:
            output_input_ids = tokenizer.encode(out, return_tensors='pt').to(model.device)
            decoder_input_ids = model._shift_right(output_input_ids)
            lm_logits = model(input_ids=prompt_input_ids, decoder_input_ids=decoder_input_ids).logits # batch_size x sequence length x V
            ce_loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), output_input_ids.view(-1))
        else:    
            full_input_ids = tokenizer.encode(inp + out, return_tensors='pt').to('cuda:0')
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids).loss * (prompt_input_ids.shape[1]-1) # don't include BOS token
            full_loss = model(full_input_ids, labels=full_input_ids).loss * (full_input_ids.shape[1]-1)
            ce_loss = full_loss - prompt_loss
        ce_loss = ce_loss.item()
        ce_list.append(ce_loss)

    return ce_list


def gpt3_cross_entropy(inputs, options, engine='ada', verbose=False):
    """
    for GPT-3 models
    """
    data = [inp + opt for inp, opt in zip(inputs, options)]
    if verbose:
        print(random.sample(data, 1))

    # request GPT-3
    outputs = []
    while len(data) > 0:
        result = request(
            model=engine,
            prompt=data[:GPT3_BATCH_SIZE],   # batching
            max_tokens=0,
            logprobs=1, 
            echo=True,
        )
        outputs += result['choices']
        data = data[GPT3_BATCH_SIZE:]
    
    # calculate cross-entropy
    ce_list = []
    assert len(inputs) == len(outputs)
    for inp, out in zip(inputs, outputs):
        # skip probabilities of words in the prompt
        i = 0
        while out['logprobs']['text_offset'][i] < len(inp):
            i += 1
        # if the input is the empty string, we still need to skip the first token since it has infinite probability
        if i == 0:
            i = 1
        # sum of log probs over the target tokens
        ce = -sum(out['logprobs']["token_logprobs"][i:])
        ce_list.append(ce)
    return ce_list
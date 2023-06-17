import ast
import torch
import numpy as np
from collections import defaultdict
from scipy.special import softmax

TF_token_dict = {
    'gpt3': {True: ' True', False: ' False'},
    'chat': {True: 'True', False: 'False'},
    'flan': {True: 10998, False: 10747}, # 10747: token ID for 'Fal'
    'llama': {True: 5852, False: 7700},
}

def gpt3_logprobs_to_TF(logprobs_list):
    T_token, F_token = TF_token_dict['gpt3'][True], TF_token_dict['gpt3'][False]
    
    TF_list, prob_mass_list = [], []
    for logprobs in logprobs_list:
        
        if T_token in logprobs and F_token in logprobs:
            if logprobs[T_token] > logprobs[F_token]:
                TF_list.append(True)
            else:
                TF_list.append(False)
        else:
            TF_list.append(None)
        
        probs = defaultdict(float)
        for k, v in zip(logprobs.keys(), softmax(list(logprobs.values()))):
            probs[k] = v
        prob_mass_list.append(probs[T_token] + probs[F_token])
    
    return TF_list, prob_mass_list


def chat_token_to_TF(token):
    """
    token: a single token returned by openai.ChatCompletion
    """
    return ast.literal_eval(token) if token in TF_token_dict['chat'].values() else None


def logits_to_TF(logits, model_class):
    """
    logits: batch_size x vocab_size
    """
    TF_tokens = TF_token_dict[model_class]
    TF =  (logits[:, TF_tokens[True]] > logits[:, TF_tokens[False]]).tolist()

    probs = torch.softmax(logits, dim=-1)
    prob_mass = (probs[:, TF_tokens[True]] + probs[:, TF_tokens[False]]).tolist()

    return TF, prob_mass

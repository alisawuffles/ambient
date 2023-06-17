import openai
from utils.constants import OPENAI_API_KEY
from tqdm import tqdm
from evaluation.model_utils import get_model_class
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = OPENAI_API_KEY

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def request(
    prompt=None,
    messages=None,
    model='ada',
    return_only_text=False,
    max_tokens=1000,
    **kwargs,
):
    # retry request (handles connection errors, timeouts, and overloaded API)
    while True:
        try:
            if get_model_class(model) == 'chat':
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            else:
                response = openai.Completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    **kwargs
                )
            break
        except Exception as e:
            tqdm.write(str(e))
            tqdm.write("Retrying...")
    
    if return_only_text:
        if get_model_class(model) == 'chat':
            generations = [gen['message']['content'] for gen in response['choices']]
        else:
            generations = [gen['text'] for gen in response['choices']]

        if len(generations) == 1:
            return generations[0]
        return generations
    
    return response
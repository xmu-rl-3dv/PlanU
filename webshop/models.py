import os
import openai
from openai import OpenAI
from openai import OpenAIError
import backoff 
from transformers import GPT2Tokenizer
import copy
completion_tokens = prompt_tokens = 0
MAX_TOKENS = 15000
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def require_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for the configured WebShop backend")
    return api_key


def configured_api_base() -> str:
    return os.environ.get("OPENAI_API_BASE", "").strip() or DEFAULT_API_BASE


@backoff.on_exception(backoff.expo, OpenAIError)
def completions_with_backoff(**kwargs_origin):
    kwargs = copy.deepcopy(kwargs_origin)
    n = kwargs['n']
    # print("API call with params:", kwargs)
    # input('--------------')
    completion =[]
    kwargs['n'] = min(4,kwargs['n'])
    # for i in range(n):
    #     kwargs['n'] = 1  # Ensure we only request one completion at a time
    #     completion.append(client.chat.completions.create(**kwargs))
    client = OpenAI(api_key=require_api_key(), base_url=configured_api_base())
    completion= client.chat.completions.create(**kwargs)
    # print(completion.choices[0].message.content[:40])
    # input('...')
    return completion


def gpt3(prompt, model="text-davinci-002", temperature=1.0, max_tokens=100, n=1, stop=None) -> list:
    openai.api_key = require_api_key()
    openai.api_base = configured_api_base()
    outputs = []
    for _ in range(n):
        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            n=1,
            stop=stop
        )
        outputs.append(response.choices[0].text.strip())
    return outputs

def gpt(prompt, model="qwen-plus", temperature=0.8, max_tokens=100, n=1, stop=None) -> list:
    if model == "test-davinci-002":
        return gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def gpt4(prompt, model="gpt-4", temperature=0.2, max_tokens=100, n=1, stop=None) -> list:
    if model == "test-davinci-002":
        return gpt3(prompt, model, temperature, max_tokens, n, stop)
    else:
        messages = [{"role": "user", "content": prompt}]
        return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    
def chatgpt(messages, model="qwen-plus", temperature=0.8, max_tokens=100, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 4)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        # outputs.extend([choice["message"]["content"] for choice in res["choices"]])

        # log completion tokens
        # completion_tokens += res["usage"]["completion_tokens"]
        # prompt_tokens += res["usage"]["prompt_tokens"]
        outputs.extend([choice.message.content for choice in res.choices])

        # log completion tokens
        completion_tokens += res.usage.completion_tokens
        prompt_tokens += res.usage.prompt_tokens
    return outputs
    
def gpt_usage(backend="gpt-4"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    elif backend == "gpt-3.5-turbo-16k":
        cost = completion_tokens / 1000 * 0.004 + prompt_tokens / 1000 * 0.003
    elif 'qwen' in backend:
        cost = completion_tokens / 1000 * 0.03 + prompt_tokens / 1000 * 0.015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

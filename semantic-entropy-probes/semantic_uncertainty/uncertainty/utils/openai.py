import os
import logging
import hashlib
from tenacity import (retry, stop_after_attempt,  # for exponential backoff
                      wait_random_exponential)

from openai import OpenAI


os.environ.setdefault("AIHUB_API_KEY", "sk-xxxxx")

# CLIENT = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

client = OpenAI(
    api_key=os.environ['AIHUB_API_KEY'],
    base_url="https://xxxx/v1"
)


# @retry(wait=wait_random_exponential(min=1, max=10))
# def predict(prompt, temperature=1.0, model='gpt-4'):
#     """Predict with GPT-4 model."""
#
#     if isinstance(prompt, str):
#         messages = [
#             {"role": "user", "content": prompt},
#         ]
#     else:
#         messages = prompt
#
#     if model == 'gpt-4':
#         model = 'gpt-4-turbo'  # or 'gpt-4o'
#     elif model == 'gpt-3.5':
#         model = 'gpt-3.5-turbo'
#
#     output = CLIENT.chat.completions.create(
#         model=model,
#         messages=messages,
#         max_tokens=200,
#         temperature=temperature,
#     )
#     response = output.choices[0].message.content
#     return response

@retry(wait=wait_random_exponential(min=1, max=10))
def predict(prompt, temperature=1.0, model='gpt-4'):
    """Predict with LLM model."""

    if isinstance(prompt, str):
        messages = [
            {"role": "user", "content": prompt},
        ]
    else:
        messages = prompt

    # 不要死if，直接允许任何model
    if model == 'gpt-4':
        model = 'gpt-4-turbo'
    elif model == 'gpt-3.5':
        model = 'gpt-3.5-turbo'
    # 其它直接透传，比如gpt-4o、gpt-4o-mini、claude-3等
    # 当然你可以再加一行自动gpt-4o的别名

    output = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=200,
        temperature=temperature,
    )
    response = output.choices[0].message.content
    return response



def md5hash(string):
    return int(hashlib.md5(string.encode('utf-8')).hexdigest(), 16)

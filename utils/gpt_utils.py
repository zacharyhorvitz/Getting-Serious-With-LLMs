''' Some utils for using the openai api '''

import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def hit_openai_chat(
    *,
    prompt,
    max_tokens,
    temperature=0.7,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,
    stop=[". ", ".\n", "\n\n"],
    model="gpt-3.5-turbo",
    device=None,
    tokenizer=None, # for compatibility with other logic
):

    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    return response

def hit_openai_chat_wrapper(**kwargs):
    '''Hits the openai chat api and returns the text response'''
    response = hit_openai_chat(**kwargs)
    response = response.choices[0].message.content
    output = ' '.join(response.split())
    return output



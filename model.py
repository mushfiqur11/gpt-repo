from openai import OpenAI
from utils import *

# def create_gpt_messages(samples=False, input_limit=10):
#     """
#     This function creates the gpt message for the given seq_in, seq_out and intent
#     """
#     messages = []
#     return messages

def calculate_cost(completion_token, prompt_token, unit_cost):
    """
    This function calculates the cost of the gpt response
    """
    return (3* completion_token + prompt_token) * unit_cost

def formulate_gpt_prompt(messages, prompt):
    """
    This function formulates the gpt prompt for the given messages
    """
    prompt_dict = {"role": "system", "content": prompt}
    messages.append(prompt_dict)
    return messages

def add_incontext_samples(messages, samples, input_limit=10):
    """
    This function adds the incontext samples to the messages list
    """
    for i, sample in enumerate(samples):
        # a, b = sample.split("\t")
        a, b = sample
        if i < input_limit:
            messages.append({"role": "user", "content": a})
            messages.append({"role": "assistant", "content": b})
    return messages

def add_task(messages, task):
    """
    This function adds the task sentence to the messages list
    """
    messages.append({"role": "user", "content": task})
    return messages

def get_gpt_response(gpt_config, prompt, task, samples=[], input_limit=10, verbose=False, full_response=False, json_response=False, get_cost=True):
    """
    This function returns the gpt response for the given prompt
    """
    messages = []
    messages = formulate_gpt_prompt(messages, prompt)
    if len(samples) > 0 and input_limit > 0:
        messages = add_incontext_samples(messages, samples, input_limit=input_limit)
    messages = add_task(messages, task)

    api_key = get_api_key(gpt_config["key_path"]) # Legacy code
    client = OpenAI(api_key=api_key) # Legacy code
        
    # client = OpenAI() # Use this if you add the api_key as an environment variable

    if verbose:
        print(messages)

    if json_response:
        response_format = { "type": "json_object" }
    else:
        response_format = { "type": "string" }
    response = client.chat.completions.create(
        model=gpt_config["model"],
        temperature=gpt_config["temperature"],
        seed=gpt_config["seed"],
        messages=messages,
        response_format=response_format
    )
    if full_response:
        return response
    if get_cost:
        cost = calculate_cost(response.usage.completion_tokens, response.usage.prompt_tokens, gpt_config["unit_cost"])
        return {"response": response.choices[0].message.content, "cost": cost}
    return response.choices[0].message.content
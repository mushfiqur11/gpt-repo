from utils import *
from model import *



if __name__ == "__main__":
    gpt_config = retrieve_yaml('gpt_config.yaml')
    prompt = retrieve_prompt('gpt_prompt.yaml')['simpleprob']

    task = 'Be the person you needed when you were younger.'

    gpt_response = get_gpt_response(gpt_config, prompt, task, verbose=True, json_response=True)

    print(gpt_response)
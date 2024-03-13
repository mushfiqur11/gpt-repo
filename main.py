from utils import *
from model import *



if __name__ == "__main__":
    gpt_config = retrieve_yaml('gpt_config.yaml')
    prompt = retrieve_prompt('gpt_prompt.yaml')['simpleprob']
    cost_limit = 0.5

    # task = 'Be the person you needed when you were younger.'

    # gpt_response = get_gpt_response(gpt_config, prompt, task, verbose=True, json_response=True)

    task_list = ['Be the person you needed when you were younger.', 'Be the person you needed when you were younger.', 'Be the person you needed when you were younger.']
    result_list = []
    total_cost = 0
    for task in task_list:
        gpt_response = get_gpt_response(gpt_config, prompt, task, verbose=True, json_response=True, get_cost=True)
        response, cost = gpt_response
        result_list.append(response)
        total_cost += cost
        if total_cost > cost_limit:
            break
    print(result_list)
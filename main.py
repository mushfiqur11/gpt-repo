from utils import *
from model import *
import pandas as pd
from tqdm import tqdm


if __name__ == "__main__":
    gpt_config = retrieve_yaml('gpt_config.yaml')
    prompt = retrieve_prompt('gpt_prompt.yaml')['simpleprob']
    cost_limit = 0.5

    # task = 'Be the person you needed when you were younger.'

    # gpt_response = get_gpt_response(gpt_config, prompt, task, verbose=True, json_response=True)

    task_source = '../data/some_data.txt'
    task_list = []
    with open(task_source, 'r') as file:
        for line in file:
            task_list.append(line.strip())
    print(task_list)
    results = []
    total_cost = 0
    for task in tqdm(task_list):
        if len(task) < 2:
            continue
        gpt_response = get_gpt_response(gpt_config, prompt, task, json_response=True, get_cost=True)
        try:
            output = json_to_dict(gpt_response['response'])
            output = output['simplicity_score']
        except:
            output = 'Failed to parse response'
        
        results.append({'input': task, 'output': output})
        total_cost += gpt_response['cost']
        if total_cost > cost_limit:
            break
    # print(results)
    df = pd.DataFrame(results)
    df.to_csv('output.csv')
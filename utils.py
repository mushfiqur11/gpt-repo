import yaml
import pandas as pd
import json

def retrieve_yaml(filename, directory='.'):
    """
    Retrieve a yaml file from a directory
    """
    file_path = directory + '/' + filename
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data

def retrieve_prompt(filename, directory='.', prompt_name=None):
    """
    Retrieve a prompt from a yaml file
    """
    data = retrieve_yaml(filename, directory)
    prompts = pd.DataFrame.from_records(data['prompts'])
    if prompt_name is not None:
        prompt = prompts[prompts['name'] == prompt_name]['prompt'].values[0]
    else:
        prompt = prompts.set_index('name').T.to_dict('records')[0]
    return prompt
    
def get_api_key(api_key_path):
    with open(api_key_path, "r") as f:
        api_key = f.read()
    return api_key

def json_to_dict(json_response):
    return json.loads(json_response)
# print(retrieve_yaml('gpt_config.yaml'))
# print(retrieve_prompt('gpt_prompt.yaml', prompt_name='base'))
# print(retrieve_prompt('gpt_prompt.yaml'))


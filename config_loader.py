import json

def load_config(path="config.json"):
    with open(path, 'r') as file:
        config = json.load(file)
    return config

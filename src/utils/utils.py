import yaml
import os

def load_config(config_path=None):
    if config_path is None:
        # Construct the path to config.yaml relative to this file
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'config.yaml'))
    
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
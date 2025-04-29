import json
import os

def cfg(file="private/cfg.json"):
    """
    Load a JSON configuration file into a dictionary.
    
    Parameters:
        file (str): Path to the JSON config file (default is "cfg.json").
    
    Returns:
        dict: Configuration data loaded from the file.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file isn't valid JSON.
    """
    if not os.path.isfile(file):
        raise FileNotFoundError(f"Configuration file '{file}' not found.")
    
    with open(file, 'r', encoding='utf-8') as f:
        return json.load(f)
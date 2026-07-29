"""
Inference utilities for SwinLip
"""
import os
import json


def read_txt_lines(filepath):
    """
    Read text file and return lines as list
    
    Args:
        filepath (str): Path to text file
        
    Returns:
        list: List of lines from file
    """
    assert os.path.isfile(filepath), f"File not found: {filepath}"
    with open(filepath) as file:
        content = file.read().splitlines()
    return content


def load_json(json_filepath):
    """
    Load JSON configuration file
    
    Args:
        json_filepath (str): Path to JSON file
        
    Returns:
        dict: JSON content as dictionary
    """
    assert os.path.isfile(json_filepath), f"JSON file not found: {json_filepath}"
    with open(json_filepath, 'r') as file:
        json_content = json.load(file)
    return json_content


def save_as_json(data, filepath):
    """
    Save data as JSON file
    
    Args:
        data (dict): Data to save
        filepath (str): Output file path
    """
    with open(filepath, 'w') as outfile:
        json.dump(data, outfile, indent=4, sort_keys=True)
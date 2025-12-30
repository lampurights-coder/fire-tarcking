import os
from pathlib import Path


import yaml


def load_config(config_path: str = "./configs/config.yaml") -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path (str): Path to the YAML config file. Defaults to './configs/config.yaml'.

    Returns:
        dict: Parsed YAML content as a Python dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            raise e
    return config


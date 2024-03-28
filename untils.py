"""
This module house useful functions that make working with the model easier.
Version: 1.0.0
Author: Deja S.
Created: 26-03-2024
Last Edit: 26-03-2024w
"""

import yaml


def load_config(file_path):
    """
    This function read the given YAML file and return the configs from that file.
    :param file_path: string path to the configs yaml file.
    :return: Dictionary of configurations.
    """
    with open(file_path, 'r') as conf:
        configs = yaml.load(conf, Loader=yaml.FullLoader)
        print(f"--- Loaded {configs['project_name']} configurations.")
        return configs

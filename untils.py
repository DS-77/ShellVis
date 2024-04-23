"""
This module house useful functions that make working with the model easier.
Version: 1.0.0
Author: Deja S.
Created: 26-03-2024
Last Edit: 26-03-2024
"""

import yaml
import cv2 as cv
import numpy as np


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


def add_class(img_path, class_name, conf):
    img = cv.imread(img_path)
    text = f"{class_name}: {conf:.2f}"

    border_size = 100
    h, w = img.shape[:2]
    new_h = h + border_size

    # Add border to image
    new_img = np.zeros((new_h, w, 3), dtype=np.uint8)
    new_img[:h, :] = img

    # Add text to image
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]

    # Center the text horizontally
    text_x = (w - text_size[0]) // 2

    # Center the text vertically
    text_y = h + (border_size - text_size[1]) // 2
    cv.putText(new_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    return new_img

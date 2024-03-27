"""
This module is used to train the Yolo8 Classifier and Object detection Model for ShellVis.
Version: 1.0.0
Author: Deja S.
Created: 26-03-2024
Last Edit: 26-03-2024
"""

import os
import tqdm
import torch
import argparse
import logging
from datetime import datetime
from ultralytics import YOLO

if __name__ == "__main__":

    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weight", required=False, default="./weights/yolo8s.pt", type=str, help="The path to "
                                                                                                    "weights.")
    ap.add_argument("-c", "--config_file", required=True, type=str, help="The path to the yaml configuration file.")
    ap.add_argument("-m", "--mode", required=True, type=str, help="Train the 'classification' or 'object' detection model.")
    opts = vars(ap.parse_args())

    # TODO: Add logger

    # Required Variables
    output_dir = f"./run"
    train_weight_dir = os.path.join(output_dir, "weights")
    out_imgs = os.path.join(output_dir, "images")
    log_dir = os.path.join(output_dir, "logs")
    conf = opts['config_file']
    w_path = opts['weights']

    # Check if input weights directory is valid
    if not os.path.exists(w_path):
        print(f"ERROR: '{w_path}' does not exist!")
        exit()

    # Create required directories
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir, f"{datetime.date()}"))
        os.mkdir(train_weight_dir)
        os.mkdir(out_imgs)
        os.mkdir(log_dir)

    exit()

    # Model Configuration
    model = YOLO(w_path)


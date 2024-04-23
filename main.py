"""
This module is the main file to run the entire ShellVis Pipeline.

Author: Deja S.
Version: 1.0.0
Created: 22-04-2024
Last Edited: 22-04-2024
"""

import os
import torch
import logging
import argparse
from ultralytics import YOLO
from untils import add_class
from datetime import datetime
from untils import load_config

if __name__ == "__main__":

    # TODO: Maybe add the option to input a single directory.

    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input_img", required=True, type=str, help="The path to the input image.")
    ap.add_argument('-m', "--input_mask", required=True, type=str, help="The path to the input mask.")
    ap.add_argument('-o', "--output_dir", required=False, type=str, default="./shellvis_output")
    opts = vars(ap.parse_args())

    # Argparse variables
    input_img = opts['input_img']
    input_mask = opts['input_mask']
    output_dir = opts['output_dir']

    # Model Variables
    cls_weights_path = "./weights/cls_best.pt"
    inpainting_weights_path = ""
    results = None
    prompt = None
    cls_conf = 0
    results_img = None

    # Check if input image path is valid.
    if not os.path.exists(input_img):
        print(f"ERROR: Input image '{input_img}' does not exists!")
        exit()

    # Check if input mask path is valid.
    if not os.path.exists(input_mask):
        print(f"ERROR: Input mask '{input_mask}' does not exist!")
        exit()

    # Create output directory if it does not exist.
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Model definition ========================================================
    cls_model = YOLO(cls_weights_path, task='classify')

    # Classify ================================================================
    results = cls_model.predict(input_img, conf=0.5)
    for r in results:
        index = results[r].probs.top1
        cls_conf = results[r].probs.top1conf
        prompt = results[r].names[index]
    # Restore =================================================================

    # Save Results ============================================================

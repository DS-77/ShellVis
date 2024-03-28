"""
This module is used to train the YOLO model for object detection tasks.
Version: 1.0.0
Author: Deja S.
Created: 28-03-2024
Last Edited: 28-03-2024
"""

import os
import torch
import argparse
from ultralytics import YOLO
from untils import load_config

if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weight", required=False, default="./weights/yolo8s.pt", type=str, help="The path to "
                                                                                                    "weights.")
    ap.add_argument("-c", "--config_file", required=True, type=str, help="The path to the yaml configuration file.")
    ap.add_argument("-m", "--mode", required=True, type=str, help="Continue training or new training session. ('new' "
                                                                  "or 'resume')")
    opts = vars(ap.parse_args())

    # Required Variables
    configs_path = opts['config_file']
    w_path = opts['weight']
    mode = opts['mode']
    results = None

    # Check of input weights path is valid
    if not os.path.exists(w_path):
        print(f"ERROR: '{w_path}' does not exist!")
        exit()

    # Load Configurations
    print("--- Loading Configurations ...")
    conf = load_config(configs_path)

    # Select a device
    device = 0 if torch.cuda.is_available() else 'cpu'

    # Model Configuration
    model = YOLO(w_path, task='detect')

    if mode == 'new':
        print("--- Starting new training session ...")
        results = model.train(data=conf['path'], epochs=conf['epochs'], imgsz=conf['image_size'],
                              device=device, batch=conf['batch'], save=conf['save'], save_period=conf['save_period'],
                              workers=conf['workers'], optimizer=conf['optimiser'], val=conf['val'],
                              plots=conf['plots'],
                              resume=conf['resume'])
    else:
        results = model.train(resume=True)

    # Model Validation
    metrics = model.val()
    print(metrics.top5)

    # Export Model
    # model.export()
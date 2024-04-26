"""
This module is used to train the Yolo8 Classifier Model for ShellVis.
Version: 1.0.0
Author: Deja S.
Created: 26-03-2024
Last Edit: 27-03-2024
"""

import os
import tqdm
import torch
import logging
import argparse
from ultralytics import YOLO
from datetime import datetime
from untils import load_config

if __name__ == "__main__":

    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weight", required=False, default="./weights/yolo8s-cls.pt", type=str, help="The path to "
                                                                                                    "weights.")
    ap.add_argument("-c", "--config_file", required=True, type=str, help="The path to the yaml configuration file.")
    ap.add_argument("-m", "--mode", required=True, type=str, help="Continue training or new training session. ('new' "
                                                                  "or 'resume')")
    opts = vars(ap.parse_args())

    # TODO: Add logger

    # Required Variables
    output_dir = f"./run"
    train_weight_dir = os.path.join(output_dir, "weights")
    out_imgs = os.path.join(output_dir, "images")
    log_dir = os.path.join(output_dir, "logs")
    configs_path = opts['config_file']
    w_path = opts['weight']
    mode = opts['mode']
    results = None

    # Check if input weights directory is valid
    if not os.path.exists(w_path):
        print(f"ERROR: '{w_path}' does not exist!")
        exit()

    # Load Configurations
    print("--- Loading configurations ...")
    conf = load_config(configs_path)

    # Create required directories
    # TODO: may not need this
    # if not os.path.exists(output_dir):
    #     os.makedirs(os.path.join(output_dir, f"{datetime.now()}"))
    #     os.mkdir(train_weight_dir)
    #     os.mkdir(out_imgs)
    #     os.mkdir(log_dir)

    # Select a device
    print(f"--- GPU Available: {torch.cuda.is_available()}")
    device = 0 if torch.cuda.is_available() else 'cpu'

    # Model Configuration
    model = YOLO(w_path, task='classify')

    if mode == "new":
        print("--- Starting a new training session ...")
        results = model.train(data=conf['path'], epochs=conf['epochs'], imgsz=conf['image_size'],
                              device=device, batch=conf['batch'], save=conf['save'], save_period=conf['save_period'],
                              workers=conf['workers'], optimizer=conf['optimiser'], val=conf['val'], plots=conf['plots'],
                              resume=conf['resume'], cos_lr=conf['cos_lr'], dropout=conf['dropout'])
    else:
        results = model.train(resume=True)

    # Model validation
    metrics = model.val()

    print("--- Metrics: ")
    print(metrics)

    # Export model
    # model.export()
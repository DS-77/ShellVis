import os
import tqdm
import torch
import argparse
import cv2 as cv
import numpy as np
from ultralytics import YOLO
from datetime import datetime


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


if __name__ == "__main__":
    # Argument parser
    # TODO: Make an option for batch or directory of images
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weight", required=False, default="./weights/yolo8s-cls.pt", type=str, help="The path to "
                                                                                                        "weights.")
    ap.add_argument("-i", "--input_img", required=True, type=str, help="The path to the input image(s).")
    opts = vars(ap.parse_args())

    input_img = opts['input_img']
    w_path = opts['weight']
    output_dir = "./classification_results"
    results = None

    # Check if input weights directory is valid
    if not os.path.exists(w_path):
        print(f"ERROR: '{w_path}' does not exist!")
        exit()

    # Create output directory if one is not made
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Select a device
    print(f"--- GPU Available: {torch.cuda.is_available()}")
    device = 0 if torch.cuda.is_available() else 'cpu'

    # Model Configuration
    model = YOLO(w_path, task='classify')

    # Predict Class
    results = model.predict(input_img, conf=0.5)

    for r in range(len(results)):
        index = results[r].probs.top1
        conf = results[r].probs.top1conf
        class_name = results[r].names[index]

        result_image = add_class(input_img, class_name, conf)

        # Save image
        file_name = f"output_{datetime.now()}.png"
        cv.imwrite(os.path.join(output_dir, file_name), result_image)
        print(f"DONE: '{file_name}' can be found on path: {output_dir}")

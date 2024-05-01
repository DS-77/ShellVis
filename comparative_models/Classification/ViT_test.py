import os
import tqdm
import torch
import argparse
import cv2 as cv
import numpy as np
from datetime import datetime, date
from transformers import ViTFeatureExtractor, ViTForImageClassification
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference


def add_class(img_path, class_name, conf=None):
    """
    This function creates the result image.
    :param img_path: The string path to the original image.
    :param class_name: The string class name predicted by YOLO8.
    :param conf: The model's prediction confidence value.
    :return: A new image.
    """
    img = cv.imread(img_path)
    text = None

    if conf is not None:
        text = f"{class_name}: {conf:.2f}"
    else:
        text = f"{class_name}"

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


def make_file(files, true_labels, pred_labels) -> None:
    """
    This function makes a txt file of the filename, true labels (gathered from the filename) and the predicted labels.
    :param files: List of filenames.
    :param true_labels: List of true labels.
    :param pred_labels: List of predicted labels.
    :return: None
    """
    outdir = "./output_label_files"
    save_file_name = os.path.join(outdir, f"vit_classification_labels_{datetime.now()}.txt")

    with open(save_file_name, "w") as f:
        f.write("file_name,true_label,predicted_label\n")

        for fn, gt, p in zip(files, true_labels, pred_labels):
            f.write(f"{fn},{gt},{p}\n")


if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--weight", required=False, default="./weights/yolo8s-cls.pt", type=str, help="The path to "
                                                                                                        "weights.")
    ap.add_argument("-i", "--input_path", required=True, type=str, help="The path to the input image(s).")
    ap.add_argument("-b", "--batch", required=False, action="store_true",
                    help="If you want to run a directory instead of a single image.")
    opts = vars(ap.parse_args())

    w_path = opts['weight']
    is_batch = opts['batch']
    input_path = opts['input_path']
    output_dir = f"./vit_runs/test_{date.today()}"
    results = None

    # Store the True and predicted labels
    true_labels = []
    predicted_labels = []

    # Check if the input path is valid
    if not os.path.exists(input_path):
        print(f"ERROR: '{input_path}' does not exist!")
        exit()

    # Check if input weights directory is valid
    if not os.path.exists(w_path):
        print(f"ERROR: '{w_path}' does not exist!")
        exit()

    # Create output directory if one is not made
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If it is in batch mode, it will gather the image paths; else treat it as a single image path.
    if is_batch:
        input_img = os.listdir(input_path)
    else:
        input_img = input_path

    # Select a device
    print(f"--- GPU Available: {torch.cuda.is_available()}")
    device = 0 if torch.cuda.is_available() else 'cpu'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Model Configuration
    classifier = VisionClassifierInference(
        feature_extractor=ViTFeatureExtractor.from_pretrained(w_path),
        model=ViTForImageClassification.from_pretrained(w_path),
    )

    # Predict the Class
    if is_batch:
        # predict for the batch
        for i in tqdm.tqdm(input_img):
            # Get true class label
            splits = i.split("_")
            t_label = f"{splits[0]}_{splits[1]}".lower()
            true_labels.append(t_label)

            i_img = os.path.join(input_path, i)

            # Get results
            pred_label = classifier.predict(img_path=i_img)
            predicted_labels.append(pred_label)
        make_file(input_img, true_labels, predicted_labels)
    else:
        # Predict for a single image
        cls_label = classifier.predict(img_path=input_img)
        result_img = add_class(input_img, cls_label)

        # Save image
        filename = f"output_{datetime.now()}.png"
        cv.imwrite(os.path.join(output_dir, filename), result_img)
        print(f"'{filename}' can be found on path: {output_dir}")

    print("Done.")

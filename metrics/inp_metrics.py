"""
This module is used to measure the metrics of the Inpainting models. It
measures:
- Structural Similarity Index (SSIM)
- Peak Signal-to-noise Ratio (PSNR)
- Mean Square Error (MSE)

Author: Deja S.
Version: 1.0.0
Created: 23-04-2024
Last Edit: 23-04-2024
"""

import os
import tqdm
import argparse
import cv2 as cv
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error


def calculate_ssim(gen_imgs: list, gt_imgs: list, gen_path: str, gt_path: str) -> list:
    """
    This function computes structural similarity index of the list of generated and ground truth images.
    :param gen_imgs: List of generated image names.
    :param gt_imgs: List of ground truth image names.
    :param gt_path: String of the generated image path.
    :param gen_path: String of the ground truth image path.
    :return: List of ssim scores.
    """
    temp_ssim = []

    for gen, gt in tqdm.tqdm(zip(gen_imgs, gt_imgs), total=len(gen_imgs)):
        # Reading in the generated and ground truth images
        img = cv.imread(os.path.join(gen_path, gen))
        gt_img = cv.imread(os.path.join(gt_path, gt))

        # Covert to greyscale
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gt_img = cv.cvtColor(gt_img, cv.COLOR_BGR2GRAY)

        # Calculate SSIM
        ssim_score, diff = structural_similarity(img, gt_img, full=True)

        temp_ssim.append(ssim_score)

    return temp_ssim


def calculate_psnr(gen_imgs: list, gt_imgs: list, gen_path: str, gt_path: str) -> list:
    """
    This function computes peak signal-to-noise ratio of the list of generated and ground truth images.
    :param gen_imgs: List of generated image names.
    :param gt_imgs: List of ground truth image names.
    :param gt_path: String of the generated image path.
    :param gen_path: String of the ground truth image path.
    :return: List of psnr scores.
    """
    temp_psnr = []

    for gen, gt in tqdm.tqdm(zip(gen_imgs, gt_imgs), total=len(gen_imgs)):
        # Reading in the generated and ground truth images
        img = cv.imread(os.path.join(gen_path, gen))
        gt_img = cv.imread(os.path.join(gt_path, gt))

        # Calculate psnr
        psnr_score = peak_signal_noise_ratio(img, gt)
        temp_psnr.append(psnr_score)

    return temp_psnr


def calculate_mse(gen_imgs: list, gt_imgs: list, gen_path: str, gt_path: str) -> list:
    """
    This function computes mean squared error of the list of generated and ground truth images.
    :param gen_imgs: List of generated image names.
    :param gt_imgs: List of ground truth image names.
    :param gt_path: String of the generated image path.
    :param gen_path: String of the ground truth image path.
    :return: List of mse scores.
    """
    temp_mse = []

    for gen, gt in tqdm.tqdm(zip(gen_imgs, gt_imgs), total=len(gen_imgs)):
        # Reading in the generated and ground truth images
        img = cv.imread(os.path.join(gen_path, gen))
        gt_img = cv.imread(os.path.join(gt_path, gt))

        # Calculate mse
        mse_score = mean_squared_error(img, gt_img)
        temp_mse.append(mse_score)

    return temp_mse


if __name__ == "__main__":
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--gen_path", required=True, type=str, help="Path to generated image directory.")
    ap.add_argument("-g", "--ground_path", required=True, type=str, help="Path to ground truth image directory")
    ap.add_argument("-n", "--name", required=False, type=str, default=None, help="Name of the model.")
    opts = vars(ap.parse_args())

    # Path Variables
    generated_path = opts['gen_path']
    ground_truth_path = opts['ground_path']
    name = opts['name']
    output_file_path = f"./runs/metrics/inpaint_metric_{name}_output_{datetime.now()}.txt" if name is not None else f"./runs/metrics/inpaint_metric_output_{datetime.now()}.txt"

    # Score Variables
    ssim_scores = []
    psnr_scores = []
    mse_scores = []

    ave_ssim_score = 0
    ave_psnr_score = 0
    ave_mse_score = 0

    # Checking if generated image path is valid
    if not os.path.exists(generated_path):
        print(f"ERROR: '{generated_path}' does not exist!")
        exit()

    # Checking if ground truth image path is valid
    if not os.path.exists(ground_truth_path):
        print(f"ERROR: '{ground_truth_path}' does not exist!")
        exit()

    # Creating the output folder if it doesn't exist
    if not os.path.exists("./runs/metrics/"):
        os.makedirs("./runs/metrics/")

    # Getting images
    gen_imgs = os.listdir(generated_path)
    gt_imgs = os.listdir(ground_truth_path)

    gen_imgs.sort()
    gt_imgs.sort()

    # Make sure we have the same number of generated and ground truth images
    assert len(gen_imgs) == len(gt_imgs)

    print(f"Number of generated images: {len(gen_imgs)}")
    print(f"Number of ground truth images: {len(gt_imgs)}")

    # Compute the metrics
    ssim_scores = calculate_ssim(gen_imgs, gt_imgs, generated_path, ground_truth_path)
    psnr_scores = calculate_psnr(gen_imgs, gt_imgs, generated_path, ground_truth_path)
    mse_scores = calculate_mse(gen_imgs, gt_imgs, generated_path, ground_truth_path)

    ave_ssim_score = np.mean(ssim_scores)
    ave_psnr_score = np.mean(psrn_scores)
    ave_mse_score = np.mean(mse_scores)

    print("=" * 80)
    print(f"Average SSIM: {ave_ssim_score:.3f}")
    print(f"Average PSNR: {ave_psnr_score:.3f}")
    print(f"Average MSE: {ave_mse_score:.3f}")
    print("=" * 80)

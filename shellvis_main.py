"""
This module is the main file to run the entire ShellVis Pipeline. Some code was pulled from the Latent-Diffusion
project by CompVis.

Author: Deja S.
Version: 1.0.3
Created: 22-04-2024
Last Edited: 27-04-2024
"""

import os
import sys
sys.path.append("/home/th/Vault-Lab/CSCE-768-Final-Project/latent-diffusion")
import glob
import torch
import logging
import argparse
import cv2 as cv
import numpy as np
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO
from untils import add_class
from datetime import datetime
from untils import load_config
from omegaconf import OmegaConf
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, device):
    """
    This function is originally written by ablattmann from CompVis as part of the Latent-Diffusion project.
    This function creates batches from the given masks and images.
    :param image: List of input image paths.
    :param mask: List of input mask paths.
    :param device: The computing device used. (GPU or CPU)
    :return: A dictionary of image and mask pairs.
    """
    image = np.array(Image.open(image).convert("RGB"))
    image = image.astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1 - mask) * image

    batch = {"image": image, "mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k] * 2.0 - 1.0
    return batch


def create_result_img(img_path, pred_img, class_name, conf):
    """
    This function creates the result image.
    :param img_path: The string path to the original image.
    :param pred_img: The generated image.
    :param class_name: The string class name predicted by YOLO8.
    :param conf: The model's prediction confidence value.
    :return: A new image. (OpenCV Image)
    """
    img = cv.imread(img_path)
    text = f"{class_name}: {conf:.2f}"

    # Concatenate the original and the predicted image
    temp_img = cv2.hconcat(img, pred_img)

    border_size = 100
    h, w = temp_img.shape[:2]
    new_h = h + border_size

    # Add border to image
    new_img = np.zeros((new_h, w, 3), dtype=np.uint8)
    new_img[:h, :] = temp_img

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
    # Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', "--input_img", required=True, type=str, help="The path to the input image.")
    ap.add_argument('-m', "--input_mask", required=True, type=str, help="The path to the input mask.")
    ap.add_argument('-o', "--output_dir", required=False, type=str, default="../shellvis_output")
    opts = vars(ap.parse_args())

    # Argparse variables
    input_img = opts['input_img']
    input_mask = opts['input_mask']
    output_dir = opts['output_dir']

    # Model Variables
    cls_weights_path = "weights/cls_best.pt"
    inpainting_weights_path = "latent-diffusion/models/ldm/inpainting_big/last.ckpt"
    inp_config_path = "latent-diffusion/models/ldm/inpainting_big/config.yaml"
    results = None
    prompt = None
    cls_conf = 0
    results_img = None
    inpainted_img = None

    # Select a device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"--- GPU Available: {torch.cuda.is_available()}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

    config = OmegaConf.load(inp_config_path)
    inp_model = instantiate_from_config(config.model)
    inp_model.load_state_dict(torch.load(inpainting_weights_path)["state_dict"],
                          strict=False)
    inp_model = inp_model.to(device)
    sampler = DDIMSampler(inp_model)

    # Classify ================================================================
    results = cls_model.predict(input_img, conf=0.5)
    for r in results:
        index = results[r].probs.top1
        cls_conf = results[r].probs.top1conf
        prompt = results[r].names[index]

    # Restore =================================================================
    with torch.no_grad():
        with inp_model.ema_scope():
            batch = make_batch(input_img, input_mask, device=device)

            # Encode masked image and concat down sampled mask
            c = inp_model.cond_stage_model.encode(batch["masked_image"])
            cc = torch.nn.functional.interpolate(batch["mask"],
                                                 size=c.shape[-2:])
            c = torch.cat((c, cc), dim=1)

            shape = (c.shape[1]-1,)+c.shape[2:]
            samples_ddim, _ = sampler.sample(S=opt.steps,
                                             conditioning=c,
                                             batch_size=c.shape[0],
                                             shape=shape,
                                             verbose=False)
            x_samples_ddim = inp_model.decode_first_stage(samples_ddim)

            img = torch.clamp((batch["image"]+1.0)/2.0, min=0.0, max=1.0)
            mask = torch.clamp((batch["mask"]+1.0)/2.0, min=0.0, max=1.0)
            predicted_img = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

            inpainted_img = (1 - mask) * img + mask * predicted_img
            inpainted_img = inpainted_img.cpu().numpy().transpose(0, 2, 3, 1)[0] * 255

    # Save Results ============================================================
    save_path = os.path.join(output_dir, f"ShellVis_output_{datetime.now()}.png")
    results_img = create_result_img(input_img, inpainted_img, prompt, cls_conf)
    cv.imwrite(save_path, results_img)

    print(f"Done: Output image can be found at the following path: {save_path}")

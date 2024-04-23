import os
import torch
import argparse
import cv2 as cv
import numpy as np
from datetime import datetime
from diffusers.utils import load_image
from diffusers import AutoPipelineForInpainting

if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_img", required=True, type=str, help="The input image path.")
    ap.add_argument("-m", "--input_mask", required=True, type=str, help="The input mask path.")
    ap.add_argument("-p", "--prompt", required=False, type=str, help="The text prompt.")
    opts = vars(ap.parse_args())

    input_img_path = opts['input_img']
    input_mask_path = opts['input_mask']
    prompt = opts['prompt']
    output_dir = "./runs/inpaint/test"

    # Check if image path is valid
    if not os.path.exists(input_img_path):
        print(f"ERROR: Input image '{input_img_path}' does not exist!")
        exit()

    # Check if mask path is valid
    if not os.path.exists(input_mask_path):
        print(f"ERROR: Input mask '{input_mask_path}' does not exist!")
        exit()

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Test values
    # input_img_path = "./Example_Image/inpainting_results/Glycymeris_rotunda_5_A_missing_20p.png"
    # input_mask_path = "./Example_Image/inpainting_sample/Glycymeris_rotunda_5_A_missing_20p_mask.png"

    input_img = load_image(input_img_path)
    input_mask = load_image(input_mask_path)

    # Model definition
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()

    generator = torch.Generator("cuda").manual_seed(92)

    # Small Experiment -> Testing if prompt makes a difference in generation.
    runs = 5

    for i in range(runs):
        # Test prompts
        prompt = "glycymeris rotunda"

        img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask).images[0]
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Save Image
        filename = f"output_{datetime.now()}.png"
        cv.imwrite(os.path.join(output_dir, filename), img)

    for i in range(runs):
        # Test prompts
        prompt = "alvania weinkauffi"

        img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask).images[0]
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        # Save Image
        filename = f"output_{datetime.now()}.png"
        cv.imwrite(os.path.join(output_dir, filename), img)

    # cv.imshow("Results", img)
    # cv.waitKey()

    # # Save Image
    # filename = f"output_{datetime.now()}.png"
    # cv.imwrite(os.path.join(output_dir, filename), img)

    print("Done.")

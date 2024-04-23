import os
import torch
import argparse
import cv2 as cv
import numpy as np
from datetime import datetime
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_img", required=True, type=str, help="The input image path.")
    ap.add_argument("-m", "--input_mask", required=True, type=str, help="The input mask path.")
    ap.add_argument("-p", "--prompt", required=False, type=str, help="The text prompt.")
    ap.add_argument("-g", "--gen_num", required=False, type=int, default=0, help="The seed number for the manual generator. Default: 0")
    opts = vars(ap.parse_args())

    input_img_path = opts['input_img']
    input_mask_path = opts['input_mask']
    prompt = opts['prompt']
    gen_num = opts['gen_num']
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

    input_img = load_image(input_img_path)
    input_mask = load_image(input_mask_path)

    # Model definition
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16"
    )
    pipeline.enable_model_cpu_offload()

    img = None

    if gen_num > 0:
        # Inpaints with
        generator = torch.Generator("cuda").manual_seed(gen_num)
        img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask, generator=generator).images[0]
    else:
        img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask).images[0]

    grid_img = make_image_grid([input_img, img], rows=1, cols=2)
    img = np.array(img)
    grid_img = np.array(grid_img)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    grid_img = cv.cvtColor(grid_img, cv.COLOR_BGR2RGB)

    # Display the results
    cv.imshow("Results", grid_img)
    cv.waitKey()

    # Save Image
    filename = f"output_{datetime.now()}.png"
    save_path = os.path.join(output_dir, filename)
    cv.imwrite(save_path, img)

    print(f"Results are saved at the following path: {save_path}")
    print("Done.")

    # Small Experiment -> Testing if prompt makes a difference in generation.
    # Test values
    # input_img_path = "./Example_Image/inpainting_results/Glycymeris_rotunda_5_A_missing_20p.png"
    # input_mask_path = "./Example_Image/inpainting_sample/Glycymeris_rotunda_5_A_missing_20p_mask.png"

    # runs = 5

    # for i in range(runs):
    #     # Test prompts
    #     prompt = "glycymeris rotunda"
    #
    #     img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask).images[0]
    #     img = np.array(img)
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    #     # Save Image
    #     filename = f"output_{datetime.now()}.png"
    #     cv.imwrite(os.path.join(output_dir, filename), img)
    #
    # for i in range(runs):
    #     # Test prompts
    #     prompt = "alvania weinkauffi"
    #
    #     img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask).images[0]
    #     img = np.array(img)
    #     img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    #     # Save Image
    #     filename = f"output_{datetime.now()}.png"
    #     cv.imwrite(os.path.join(output_dir, filename), img)

import os
import tqdm
import torch
import argparse
import cv2 as cv
import numpy as np
from datetime import datetime, date
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid

if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_img", required=True, type=str, help="The input image path.")
    ap.add_argument("-m", "--input_mask", required=False, type=str, default=None, help="The input mask path.")
    ap.add_argument("-p", "--prompt", required=False, type=str, default=None, help="The text prompt.")
    ap.add_argument("-g", "--gen_num", required=False, type=int, default=0,
                    help="The seed number for the manual generator. Default: 0")
    ap.add_argument("-b", "--batch", required=False, action="store_true",
                    help="If you want to run a directory instead of a single image.")
    opts = vars(ap.parse_args())

    input_img_path = opts['input_img']
    input_mask_path = opts['input_mask']
    prompt = opts['prompt']
    gen_num = opts['gen_num']
    output_dir = "./runs/inpaint/stable_diffusion_1"
    is_batch = opts['batch']

    # Check if image path is valid
    if not os.path.exists(input_img_path):
        print(f"ERROR: Input image '{input_img_path}' does not exist!")
        exit()

    # Check if mask path is valid
    if not is_batch:
        if not os.path.exists(input_mask_path):
            print(f"ERROR: Input mask '{input_mask_path}' does not exist!")
            exit()

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Model definition
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "runwayml/stable-diffusion-inpainting", torch_dtype=torch.float16, variant="fp16", safety_checker=None,
        requires_safety_checker=False
    )
    pipeline.enable_model_cpu_offload()

    if is_batch:
        imgs = [x for x in os.listdir(input_img_path) if "_mask" not in x]
        masks = [os.path.join(input_img_path, y) for y in os.listdir(input_img_path) if "_mask" in y]

        print(f"--- Number of Images: {len(imgs)}")
        print(f"--- Number of Masks: {len(masks)}")

        assert len(imgs) == len(masks)

        for in_img_path in imgs:
            # Load in image and mask
            in_mask_path = os.path.join(input_img_path, f"{in_img_path.split('.')[0]}_mask.png")
            input_img = load_image(os.path.join(input_img_path, in_img_path))
            input_mask = load_image(in_mask_path)

            img = None

            # Load in the Class prompt
            temp_name = in_img_path.split("_")
            name = f"{temp_name[0]}_{temp_name[1]}"
            prompt = name.lower()
            save_num = 1

            print(f"Prompt: {prompt}")

            if gen_num > 0:
                # Inpaints with manual generation seed
                generator = torch.Generator("cuda").manual_seed(gen_num)
                img = pipeline(image=input_img, mask_image=input_mask, generator=generator).images[0]
            else:
                # Inpaint without generation seed
                img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask).images[0]

            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # Save Image
            filename = f"{name}_{save_num}.png"
            save_path = os.path.join(output_dir, filename)
            cv.imwrite(save_path, img)

    else:
        # Run for Single image
        input_img = load_image(input_img_path)
        input_mask = load_image(input_mask_path)

        img = None

        if gen_num > 0:
            # Inpaints with manual generation seed
            generator = torch.Generator("cuda").manual_seed(gen_num)
            img = pipeline(prompt=prompt, image=input_img, mask_image=input_mask, generator=generator).images[0]
        else:
            # Inpaint without generation seed
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

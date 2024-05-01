"""
This code is from Openai's GLIDE Inpainting Notebook Tutorial
https://github.com/openai/glide-text2im/blob/main/notebooks/inpaint.ipynb
"""

import argparse
import numpy as np
import torch as th
from PIL import Image
from typing import Tuple
import torch.nn.functional as F
from IPython.display import display

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)


def show_images(batch: th.Tensor):
    """ Display a batch of images inline. """
    scaled = ((batch + 1) * 127.5).round().clamp(0, 255).to(th.uint8).cpu()
    reshaped = scaled.permute(2, 0, 3, 1).reshape([batch.shape[2], -1, 3])
    display(Image.fromarray(reshaped.numpy()))


def read_image(path: str, size: int = 256) -> Tuple[th.Tensor, th.Tensor]:
    pil_img = Image.open(path).convert('RGB')
    pil_img = pil_img.resize((size, size), resample=Image.BICUBIC)
    img = np.array(pil_img)
    return th.from_numpy(img)[None].permute(0, 3, 1, 2).float() / 127.5 - 1


if __name__ == "__main__":
    # Argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_img", required=True, type=str, help="The input image path.")
    ap.add_argument("-m", "--input_mask", required=True, type=str, help="The input mask path.")
    ap.add_argument("-p", "--prompt", required=False, type=str, help="The text prompt.")

    input_img_path = opts['input_img']
    input_mask_path = opts['input_mask']
    prompt = opts['prompt']

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    # Create base model.
    options = model_and_diffusion_defaults()
    options['inpaint'] = True
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100'  # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()

    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base-inpaint', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    # Sampling parameters
    prompt = "a corgi in a field"
    batch_size = 1
    guidance_scale = 5.0

    # Tune this parameter to control the sharpness of 256x256 images.
    # A value of 1.0 is sharper, but sometimes results in grainy artifacts.
    upsample_temp = 0.997

    # Source image we are inpainting
    source_image_256 = read_image('grass.png', size=256)
    source_image_64 = read_image('grass.png', size=64)

    # The mask should always be a boolean 64x64 mask, and then we
    # can upsample it for the second stage.
    source_mask_64 = th.ones_like(source_image_64)[:, :1]
    source_mask_64[:, :, 20:] = 0
    source_mask_256 = F.interpolate(source_mask_64, (256, 256), mode='nearest')

    # Visualize the image we are inpainting
    show_images(source_image_256 * source_mask_256)

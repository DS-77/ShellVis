ShellVis: Seashell Classification and Restoration Model
---

![Pipeline Overview]{Example_Image/Repo_images/pipeline_overview.png}

*Abstract:*

Seashell image reconstruction is a valuable technique that
can provide insights into seashell species, subspecies, and
their environmental and ecological contexts. In this paper,
we propose a novel two-stage network to aid in classify-
ing seashells by species and image inpainting to visualise
the possible complete shell structure. In the first stage, the
image reconstruction phase, latent diffusion is used to re-
construct the missing parts of a shell using the binary mask
as a guide. The second stage, Latent Stable Diffusion, is
applied to the reconstruction of the missing part of the shell
using a binary mask. In our comparative analysis, we em-
ployed the Vision Transformer (ViT) model to benchmark
classification performance across tasks. The results demon-
strate the superior performance of YOLOv8 over ViT on the
testing subset of our dataset, achieving 75% accuracy for
classification tasks and a Structural Similarity Index (SSIM)
score of 96% (0.968), the highest Peak Signal-to-Noise Ratio
(PSNR) at 30.360, and the lowest Mean Squared Error (MSE)
at 117.193 for shell reconstruction. Moreover, we present
a comparison between our model and the Latent and Sta-
ble diffusion models, demonstrating the effectiveness of our
framework within the broader context of existing methods.
Overall, the proposed model presents a promising approach
to seashell image reconstruction and classification that has
significant implications for marine biology and conservation.

Preliminary Results:

![Qualitative-Gallery]{Example_Image/Repo_images/inpainting_gallery.png}

## How to install ShellVis

1) Clone this repository in your chosen directory.
2) Create a conda environment for ShellVis: `conda create --name <my-env>`
3) Install the dependencies for ShellVis: `pip install -e .`
4) Install the YOLO8 library from the platform Ultralytics using pip: `pip install ultralytics` or use pip to install it from their repository: `pip install git+https://github.com/ultralytics/ultralytics.git@main` 
5) Inside the ShellVis, download the Latent Diffusion Repository from CompVis/latent-diffusion: `git clone https://github.com/CompVis/latent-diffusion.git`
6) Install the needed dependencies for Latent Diffusion. Follow the directions provided by the official repo.

## How to run ShellVis
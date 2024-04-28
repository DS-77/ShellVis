ShellVis: Seashell Classification and Restoration Model
---

*Abstract:*

The topic of image reconstruction and object classification, particularly in the context of seashells, is a valuable and 
intriguing area of research. Through the use of a deep learning model, a shell can be identified by species and missing 
portions of a shell can be restored, providing users with a complete panel of the shell's information. This framework has 
numerous applications, including aiding marine biologists and researchers in gathering data on the ocean's ecosystems and 
marine life, as well as providing seashell collectors and educational institutions with species-specific information. In 
this project, We propose an image-processing pipeline that involves the preprocessing of a single 512x512 RGB image of a 
seashell and utilises a YOLO 8 object classification model and a latent diffusion model to classify and generate missing 
portions of a seashell image. We will utilise the "All Shell Images" dataset and a small testing dataset to train and validate 
my pipeline. The results of this project have the potential to provide valuable insights into seashell classification and 
distribution, as well as the importance of wildlife conservation and protection.


Goal: This project attempts to classify a seashell based on a given RGB image and restore the shell with a given mask of the missing region.

TODO LIST:

- [x] Find a dataset
- [x] Extract class names from the image names
- [ ] Augment the dataset to have missing portions and generate masks for them (generation model)
- [x] Define preprocessing steps
- [x] Train classification model
- [x] Find the best metrics for evaluating the model
- [x] Find comparative models
- [ ] Create figures
- [ ] Write up the report
- [ ] Gather all references
- [ ] (Optional) Make a nice interface for the pipline

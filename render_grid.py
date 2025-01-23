import os
import torch
from tqdm.auto import tqdm
from glob import glob
from torchvision.utils import save_image
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


device = "cuda"
pipeline = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    safety_checker=None,
    feature_extractor=None,
    image_encoder=None,
    requires_safety_checker=False,
    ).to(device)

prompt = "Yann LeCun in a blue shirt and glasses giving a speech."
images = []
cols = 8

for i in range(cols):
    images.append(pipeline(prompt, guidance_scale=5, output_type="pt")[0])

checkpoints = glob("./experiments/samples-10000/checkpoint-*/unet")
for ckpt in tqdm(checkpoints):
    pipeline.unet = UNet2DConditionModel.from_pretrained(ckpt).to(device)
    for i in range(cols):
        images.append(pipeline(prompt, guidance_scale=5, output_type="pt")[0])

images = torch.cat(images, dim=0)
save_image(images, f"./experiments/samples-10000/checkpoint_grid_{cols}.png", nrow=cols)
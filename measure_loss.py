import os
import random
from glob import glob
from tqdm.auto import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from torchvision.io import read_image
from torchvision.utils import save_image

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import compute_snr


class T2iDataset(Dataset):
    def __init__(self, root_folder, resolution=512):
        self.root_folder = root_folder
        self.resolution = resolution
        self.images = []
        self.latents = []
        self.captions = []
        self.encoder_hidden_states = []
        
        # for image_path in glob(os.path.join(root_folder, "*.png")):
        for image_path in glob(os.path.join(root_folder, "*.png"))[:2]:
            self.images.append(Image.open(image_path).convert('RGB'))
            with open(os.path.splitext(image_path)[0] + ".txt", "r") as capfile:
                self.captions.append(capfile.read())
        
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=self.resolution),
            v2.CenterCrop(size=self.resolution),
        ])
    
    def encode_images(self, vae):
        for i in range(len(self.images)):
            pixels = self.images[i]
            pixels = self.transforms(pixels) * 2 - 1
            pixels = torch.clamp(torch.nan_to_num(pixels), min=-1, max=1)
            pixels = pixels.unsqueeze(0).to(vae.device)
            latents = vae.encode(pixels).latent_dist.sample() * 0.18215
            self.latents.append(latents[0].to("cpu"))
    
    def encode_captions(self, tokenizer, text_encoder):
        for caption in self.captions:
            ids = tokenizer(
                caption,
                max_length=tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt",
                ).input_ids.to(text_encoder.device)
            enc = text_encoder(ids, return_dict=False)[0]
            self.encoder_hidden_states.append(enc[0].to("cpu"))
    
    def __len__(self):
        return len(self.latents)
    
    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "encoder_hidden_states": self.encoder_hidden_states[idx],
        }


def train(
    output_path = "./experiments/",
    dataset_path = None,
    seed = 1234,
    batch_size = 1,
    device = "cuda",
    unet_path = None,
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    os.makedirs(output_path, exist_ok=True)
    t_writer = SummaryWriter(log_dir=output_path, flush_secs=60)
    
    hf_identifier = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    noise_scheduler = DDPMScheduler.from_pretrained(hf_identifier, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(hf_identifier, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(hf_identifier, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(hf_identifier, subfolder="vae").to(device)
    if unet_path is not None:
        unet = UNet2DConditionModel.from_pretrained(unet_path).to(device)
    else:
        unet = UNet2DConditionModel.from_pretrained(hf_identifier, subfolder="unet").to(device)
    
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    
    def collate_batch(batch):
        latents = []
        encoder_hidden_states = []
        for sample in batch:
            latents.append(sample["latents"])
            encoder_hidden_states.append(sample["encoder_hidden_states"])
        latents = torch.stack(latents, dim=0)
        encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)
        return latents, encoder_hidden_states
    
    val_dataset = T2iDataset(os.path.join(dataset_path, "val"))
    # val_dataset = T2iDataset(os.path.join(dataset_path, "train"))
    val_dataset.encode_images(vae)
    val_dataset.encode_captions(tokenizer, text_encoder)
    
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_batch,
        num_workers = 0,
        pin_memory = True,
        drop_last = False,
    )
    
    global_step = 0
    progress_bar = tqdm(range(0, noise_scheduler.config.num_train_timesteps))
    while global_step < noise_scheduler.config.num_train_timesteps:
        timesteps = torch.tensor([global_step] * batch_size).long().to(device)
        snr = compute_snr(noise_scheduler, timesteps).detach().item()
        loss = 0
        for step, batch in enumerate(val_dataloader):
            latents = batch[0].to(device)
            encoder_hidden_states = batch[1].to(device)
            noise = torch.randn_like(latents)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss += F.mse_loss(model_pred.float(), noise.float(), reduction="mean").detach().item()
        
        loss = loss / len(val_dataloader)
        t_writer.add_scalar("loss", loss, global_step)
        t_writer.add_scalar("snr", snr, global_step)
        
        global_step += 1
        progress_bar.update(1)

if __name__ == "__main__":
    experiment_name = "./experiments/loss_snr/optimal_val"
    dataset_path = "./datasets/example"
    unet_path = "./experiments/long_lr_sweep/1.0e-07/unet"
    
    train(output_path=experiment_name, dataset_path=dataset_path, unet_path=unet_path)
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
        self.captions = []
        
        for image_path in glob(os.path.join(root_folder, "*.png")):
            self.images.append(Image.open(image_path).convert('RGB'))
            with open(os.path.splitext(image_path)[0] + ".txt", "r") as capfile:
                self.captions.append(capfile.read())
        
        self.transforms = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=self.resolution),
            v2.CenterCrop(size=self.resolution),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        pixels = self.images[idx]
        pixels = self.transforms(pixels) * 2 - 1
        pixels = torch.clamp(torch.nan_to_num(pixels), min=-1, max=1)
        caption = self.captions[idx]
        return {"pixels": pixels, "caption": caption}


def train(
    output_path = "./experiments/",
    dataset_path = None,
    lr = 1e-4,
    train_steps = 1000,
    val_steps = 100,
    seed = None,
    batch_size = 1,
    device = "cuda",
):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
    
    os.makedirs(output_path, exist_ok=True)
    t_writer = SummaryWriter(log_dir=output_path, flush_secs=60)
    
    def collate_batch(batch):
        pixels = []
        captions = []
        for sample in batch:
            pixels.append(sample["pixels"])
            captions.append(sample["caption"])
        pixels = torch.stack(pixels, dim=0)
        return pixels, captions
    
    train_dataset = T2iDataset(os.path.join(dataset_path, "train"))
    train_dataloader = DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        collate_fn = collate_batch,
        num_workers = 0,
        pin_memory = True,
        drop_last = False,
    )
    
    val_dataset = T2iDataset(os.path.join(dataset_path, "val"))
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = batch_size,
        shuffle = False,
        collate_fn = collate_batch,
        num_workers = 0,
        pin_memory = True,
        drop_last = False,
    )
    
    hf_identifier = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    noise_scheduler = DDPMScheduler.from_pretrained(hf_identifier, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(hf_identifier, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(hf_identifier, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(hf_identifier, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(hf_identifier, subfolder="unet").to(device)
    
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(True)
    unet.train()
    
    train_lr = lr * batch_size
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr = train_lr,
        weight_decay = 1e-4,
    )
    
    def encode_captions(captions, dropout=0, generator=None):
        input_ids = []
        for caption in captions:
            if dropout > 0:
                if torch.rand(1, generator=generator) < dropout:
                    caption = "" # caption dropout for better CFG
            ids = tokenizer(
                caption,
                max_length=tokenizer.model_max_length, 
                padding="max_length", 
                truncation=True, 
                return_tensors="pt",
                ).input_ids
            input_ids.append(ids)
        input_ids = torch.stack(input_ids, dim=0).to(device)
        return text_encoder(input_ids, return_dict=False)[0]
    
    def vae_encode(pixels):
        latents = vae.encode(pixels.to(device)).latent_dist.sample()
        return latents * 0.18215
    
    def sample_timesteps(latents, generator=None):
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device = latents.device,
            generator=generator,
            ).long()
        return timesteps
    
    def randn_like_g(x, generator=None):
        return torch.randn(x.size(), generator=generator, dtype=x.dtype, layout=x.layout, device=x.device)
    
    def sample_noise(latents, offset=0, generator=None):
        noise = randn_like_g(latents, generator=generator)
        if offset > 0:
            noise += offset * randn_like_g(latents[..., 0, 0], generator=generator)[..., None, None]
        return noise
    
    def prepare_inputs(batch, dropout=0, offset=0, generator=None):
        pixels, captions = batch
        encoder_hidden_states = encode_captions(captions, dropout=dropout, generator=generator)
        latents = vae_encode(pixels)
        timesteps = sample_timesteps(latents, generator=generator)
        noise = sample_noise(latents, offset=offset, generator=generator)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        return encoder_hidden_states, timesteps, noise, noisy_latents
    
    global_step = 0
    progress_bar = tqdm(range(0, train_steps))
    while global_step < train_steps:
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            encoder_hidden_states, timesteps, noise, noisy_latents = prepare_inputs(batch)
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step)
            
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            global_step += 1
            
            if global_step % val_steps == 0:
                with torch.inference_mode():
                    all_loss = 0.0
                    repeat = 4
                    for i in range(repeat):
                        for step, batch in enumerate(val_dataloader):
                            gen_cuda = torch.Generator(device=device)
                            gen_cuda.manual_seed(seed + step + i * 1000)
                            encoder_hidden_states, timesteps, noise, noisy_latents = prepare_inputs(batch, generator=gen_cuda)
                            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                            val_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                            all_loss += val_loss.detach().item()
                    all_loss = all_loss / (len(val_dataloader) * repeat)
                    t_writer.add_scalar("test/val", all_loss, global_step)
            
            if global_step >= train_steps:
                break
    
    unet.save_pretrained(os.path.join(output_path, "unet"), safe_serialization=True)


if __name__ == "__main__":
    experiment_name = "./experiments/long_lr_sweep/"
    lr_sweep = [1e-7, 2e-7, 3e-7, 5e-7, 1e-6, 5e-6]
    
    dataset_path = "E:/datasets/yann/v1"
    train_steps = 10_000
    val_steps = 100
    seed = 1234
    batch_size = 1
    device = "cuda"
    
    for lr in lr_sweep:
        output_path = experiment_name + f"{lr:.1e}/"
        train(
            output_path = output_path,
            dataset_path = dataset_path,
            lr = lr,
            train_steps = train_steps,
            val_steps = val_steps,
            seed = seed,
            batch_size = batch_size,
            device = device,
        )
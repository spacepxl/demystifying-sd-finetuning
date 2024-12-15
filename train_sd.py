import os
import random
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from contextlib import contextmanager
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


@contextmanager
def temp_rng(new_seed=None):
	"""
    https://github.com/fpgaminer/bigasp-training/blob/main/utils.py#L73
	Context manager that saves and restores the RNG state of PyTorch, NumPy and Python.
	If new_seed is not None, the RNG state is set to this value before the context is entered.
	"""

	# Save RNG state
	old_torch_rng_state = torch.get_rng_state()
	old_torch_cuda_rng_state = torch.cuda.get_rng_state()
	old_numpy_rng_state = np.random.get_state()
	old_python_rng_state = random.getstate()

	# Set new seed
	if new_seed is not None:
		torch.manual_seed(new_seed)
		torch.cuda.manual_seed(new_seed)
		np.random.seed(new_seed)
		random.seed(new_seed)

	yield

	# Restore RNG state
	torch.set_rng_state(old_torch_rng_state)
	torch.cuda.set_rng_state(old_torch_cuda_rng_state)
	np.random.set_state(old_numpy_rng_state)
	random.setstate(old_python_rng_state)


def train(
    output_path = "./experiments/",
    dataset_path = None,
    lr = 1e-4,
    train_steps = 1000,
    val_steps = 100,
    seed = None,
    batch_size = 1,
    val_repeats = 4,
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
    
    test_dataset = T2iDataset(os.path.join(dataset_path, "test"))
    test_dataloader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        collate_fn = collate_batch,
        num_workers = 0,
        pin_memory = True,
        drop_last = False,
    )
    
    val_dataset = T2iDataset(os.path.join(dataset_path, "val"))
    val_dataloader = DataLoader(
        dataset = val_dataset,
        batch_size = 1,
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
    
    train_lr = lr * (batch_size ** 0.5)
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr = train_lr,
        weight_decay = 1e-4,
    )
    
    def encode_captions(captions, dropout=0):
        input_ids = []
        for caption in captions:
            if torch.rand(1) < dropout:
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
        return latents * vae.config.scaling_factor
    
    def sample_timesteps(latents, timestep_range=None):
        min_timestep = timestep_range[0] if timestep_range is not None else 0
        max_timestep = timestep_range[1] if timestep_range is not None else noise_scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            min_timestep,
            max_timestep,
            (latents.shape[0],),
            device = latents.device,
            ).long()
        return timesteps
    
    def sample_noise(latents, offset=0):
        noise = torch.randn_like(latents)
        if offset > 0:
            noise += offset * torch.randn_like(latents[..., 0, 0])[..., None, None]
        return noise
    
    def mse_loss(pred, target, timesteps):
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) # reduce over all dimensions except batch
        debiased_loss = loss / (0.7365 * torch.exp(-0.0052 * timesteps)) # debias by loss/timestep fit function
        return loss.mean(), debiased_loss.mean()
    
    def get_pred(batch, dropout=0, offset=0, timestep_range=None):
        pixels, captions = batch
        encoder_hidden_states = encode_captions(captions, dropout=dropout)
        latents = vae_encode(pixels)
        timesteps = sample_timesteps(latents, timestep_range=timestep_range)
        noise = sample_noise(latents, offset=offset)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        return mse_loss(model_pred, noise, timesteps)
    
    global_step = 0
    progress_bar = tqdm(range(0, train_steps))
    while global_step < train_steps:
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, debiased_loss = get_pred(batch)
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step * batch_size)
            t_writer.add_scalar("loss/debiased", debiased_loss.detach().item(), global_step * batch_size)
            
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            global_step += 1
            
            if global_step == 1 or global_step % val_steps == 0:
                with torch.inference_mode(), temp_rng(seed):
                    test_loss = 0.0
                    val_loss = 0.0
                    for i in range(val_repeats):
                        min_timestep = int(i * noise_scheduler.config.num_train_timesteps / val_repeats)
                        max_timestep = int((i + 1) * noise_scheduler.config.num_train_timesteps / val_repeats)
                        for step, batch in enumerate(test_dataloader):
                            _, debiased_loss = get_pred(batch, timestep_range=(min_timestep, max_timestep))
                            test_loss += debiased_loss.detach().item()
                        
                        for step, batch in enumerate(val_dataloader):
                            _, debiased_loss = get_pred(batch, timestep_range=(min_timestep, max_timestep))
                            val_loss += debiased_loss.detach().item()
                    
                    t_writer.add_scalar("test/test", test_loss / (len(test_dataloader) * val_repeats), global_step * batch_size)
                    t_writer.add_scalar("test/val", val_loss / (len(val_dataloader) * val_repeats), global_step * batch_size)
            
            if global_step >= train_steps:
                break
    
    unet.save_pretrained(os.path.join(output_path, "unet"), safe_serialization=True)


if __name__ == "__main__":
    experiment_name = "./experiments/seed_new/"
    # lr_sweep = [1e-7, 2e-7, 3e-7, 5e-7, 1e-6, 5e-6]
    lr_sweep = [5e-7,]
    
    dataset_path = "E:/datasets/yann/v1"
    train_steps = 1250
    val_steps = 25
    seed = 8675309
    batch_size = 4
    val_repeats = 8
    device = "cuda"
    
    for lr in lr_sweep:
        # output_path = experiment_name + f"{lr:.1e}/"
        output_path = experiment_name + f"{seed}/"
        train(
            output_path = output_path,
            dataset_path = dataset_path,
            lr = lr,
            train_steps = train_steps,
            val_steps = val_steps,
            seed = seed,
            batch_size = batch_size,
            val_repeats = val_repeats,
            device = device,
        )
import os
import random
import numpy as np
from glob import glob
from tqdm.auto import tqdm
from contextlib import contextmanager
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2, InterpolationMode
from torchvision.io import read_image
from torchvision.utils import save_image

from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.training_utils import compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.loaders import StableDiffusionLoraLoaderMixin

class T2iDataset(Dataset):
    def __init__(self, root_folder, resolution=512, random_crop=False):
        self.root_folder = root_folder
        self.resolution = resolution
        self.images = []
        self.captions = []
        
        for image_path in glob(os.path.join(root_folder, "*.png")):
            self.images.append(Image.open(image_path).convert('RGB'))
            with open(os.path.splitext(image_path)[0] + ".txt", "r") as capfile:
                self.captions.append(capfile.read())
        
        if random_crop:
            self.transforms = v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomResizedCrop(
                    size = self.resolution,
                    scale = (0.25, 1.0),
                    ratio = (0.9, 1.1),
                ),
            ])
        else:
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
    rank = 8,
    alpha = 1,
    train_steps = 1000,
    save_steps = 1000,
    val_steps = 100,
    stable_train_loss = True,
    seed = None,
    val_seed = 1234,
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
    
    train_dataset = T2iDataset(os.path.join(dataset_path, "train"), random_crop=False)
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
    unet.requires_grad_(False)
    
    # lora target options: attn, attn+ff, attn+ff+resnet
    # target_modules = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj"]
    target_modules = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "ff.net.0.proj", "ff.net.2"]
    # target_modules = ["to_k", "to_q", "to_v", "to_out.0", "add_k_proj", "add_v_proj", "ff.net.0.proj", "ff.net.2", "conv1", "conv2"]
    
    unet_lora_config = LoraConfig(
        r = rank,
        lora_alpha = alpha,
        init_lora_weights = "gaussian",
        target_modules = target_modules,
    )
    unet.add_adapter(unet_lora_config)
    unet.train()
    
    train_lr = lr * (batch_size ** 0.5)
    optimizer = torch.optim.AdamW(
        params = list(filter(lambda p: p.requires_grad, unet.parameters())),
        lr = train_lr,
        weight_decay = 1e-4,
    )
    
    global_step = 0
    train_logs = {"train_step": [], "train_loss": [], "train_timestep": []}
    test_logs = {"train_step": [], "train_loss": [], "train_timestep": []}
    val_logs = {"train_step": [], "train_loss": [], "train_timestep": []}
    
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
    
    def mse_loss(pred, target, timesteps, log_to=None):
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) # reduce over all dimensions except batch
        if log_to is not None:
            for i in range(timesteps.shape[0]):
                log_to["train_step"].append(global_step)
                log_to["train_loss"].append(loss[i].item())
                log_to["train_timestep"].append(timesteps[i].item())
        debiased_loss = loss / (0.7365 * torch.exp(-0.0052 * timesteps)) # debias by loss/timestep fit function
        return loss.mean(), debiased_loss.mean()
    
    def get_pred(batch, dropout=0, offset=0, timestep_range=None, log_to=None):
        pixels, captions = batch
        encoder_hidden_states = encode_captions(captions, dropout=dropout)
        latents = vae_encode(pixels)
        timesteps = sample_timesteps(latents, timestep_range=timestep_range)
        noise = sample_noise(latents, offset=offset)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
        return mse_loss(model_pred, noise, timesteps, log_to=log_to)
    
    def plot_logs(log_dict):
        plt.scatter(log_dict["train_timestep"], log_dict["train_loss"], s=3, c=log_dict["train_step"], marker=".", cmap='cool')
        plt.xlabel("timestep")
        plt.ylabel("loss")
        plt.yscale("log")
    
    def save_lora(global_step):
        save_path = os.path.join(output_path, f"checkpoint-{global_step:08}")
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        lora_sd_to_save = {}
        for key in unet_lora_layers.keys():
            lora_sd_to_save[key] = unet_lora_layers[key]
            for pattern in ["lora.down.weight", "lora_A.weight"]:
                if pattern in key:
                    alpha_key = key.replace(pattern, "alpha")
                    lora_sd_to_save[alpha_key] = torch.tensor([alpha])
        StableDiffusionLoraLoaderMixin.save_lora_weights(save_path, unet_lora_layers=lora_sd_to_save)
    
    progress_bar = tqdm(range(0, train_steps))
    while global_step < train_steps:
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, debiased_loss = get_pred(batch, log_to=train_logs)
            t_writer.add_scalar("loss/train", loss.detach().item(), global_step * batch_size)
            t_writer.add_scalar("loss/debiased", debiased_loss.detach().item(), global_step * batch_size)
            
            loss.backward()
            optimizer.step()
            progress_bar.update(1)
            global_step += 1
            
            if global_step == 1 or global_step % val_steps == 0:
                with torch.inference_mode(), temp_rng(val_seed):
                    inference_steps = len(val_dataloader) * val_repeats
                    if stable_train_loss:
                        inference_steps += len(test_dataloader) * val_repeats
                    temp_pbar = tqdm(range(inference_steps), desc="validation", leave=False)
                    test_loss = 0.0
                    val_loss = 0.0
                    for i in range(val_repeats):
                        min_timestep = int(i * noise_scheduler.config.num_train_timesteps / val_repeats)
                        max_timestep = int((i + 1) * noise_scheduler.config.num_train_timesteps / val_repeats)
                        if stable_train_loss:
                            for step, batch in enumerate(test_dataloader):
                                loss, _ = get_pred(batch, timestep_range=(min_timestep, max_timestep), log_to=test_logs)
                                test_loss += loss.detach().item()
                                temp_pbar.update(1)
                        
                        for step, batch in enumerate(val_dataloader):
                            loss, _ = get_pred(batch, timestep_range=(min_timestep, max_timestep), log_to=val_logs)
                            val_loss += loss.detach().item()
                            temp_pbar.update(1)
                    del temp_pbar
                    
                    plot_logs(train_logs)
                    t_writer.add_figure("train_loss", plt.gcf(), global_step * batch_size)
                    
                    plot_logs(test_logs)
                    t_writer.add_figure("test_loss", plt.gcf(), global_step * batch_size)
                    t_writer.add_scalar("test/test", test_loss / (len(test_dataloader) * val_repeats), global_step * batch_size)
                    
                    plot_logs(val_logs)
                    t_writer.add_figure("val_loss", plt.gcf(), global_step * batch_size)
                    t_writer.add_scalar("test/val", val_loss / (len(val_dataloader) * val_repeats), global_step * batch_size)
            
            if global_step >= train_steps or global_step % save_steps == 0:
                save_lora(global_step)
            
            if global_step >= train_steps:
                break


if __name__ == "__main__":
    experiment_name = "./experiments/example_lora"
    
    # dataset subfolders: example/train, example/test, example/val
    dataset_path = "./datasets/example"
    
    train(
        output_path = experiment_name,
        dataset_path = dataset_path,
        lr = 1.5e-4,
        rank = 128,
        alpha = 1,
        train_steps = 5000,
        save_steps = 500,
        val_steps = 500,
        seed = 1234,
        val_seed = 1234,
        batch_size = 1,
        val_repeats = 4,
        device = "cuda",
    )
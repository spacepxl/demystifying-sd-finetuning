import os
from tqdm.auto import tqdm
from glob import glob
import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
from unittest.mock import patch


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Workaround for FlashAttention"""
    if os.path.basename(filename) != "modeling_florence2.py":
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def caption(base_folder, model, processor, device, dtype):
    images = glob(os.path.join(base_folder, "*.png"))
    for img_path in tqdm(images):
        img_id = os.path.split(os.path.splitext(img_path)[0])[-1]
        img_px = Image.open(img_path).convert('RGB')
        
        prompt = "<CAPTION>"
        inputs = processor(text=prompt, images=img_px, return_tensors="pt").to(device, dtype)
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(img_px.width, img_px.height)
            )
        caption = parsed_answer[prompt]
        
        caption_file = os.path.join(os.path.split(img_path)[0], f"{img_id}.txt")
        with open(caption_file, 'w') as file:
            file.write(caption)


if __name__ == "__main__":
    dataset_dir = "./datasets/example"
    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "val")
    device = "cuda"
    dtype = torch.float16
    
    identifier = "microsoft/Florence-2-base"
    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        processor = AutoProcessor.from_pretrained(
            identifier,
            trust_remote_code=True
            )
        
        model = AutoModelForCausalLM.from_pretrained(
            identifier,
            torch_dtype=dtype,
            trust_remote_code=True
            ).to(device)
    
    caption(train_path, model, processor, device, dtype)
    caption(val_path, model, processor, device, dtype)
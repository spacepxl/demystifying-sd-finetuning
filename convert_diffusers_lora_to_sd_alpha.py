import os
import argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Convert diffusers lora to sd key names",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input lora .safetensors file",
        )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .safetensors file to save",
        )
    parser.add_argument(
        "--alpha",
        type=int,
        default=None,
        help="Network alpha, defaults to rank",
        )
    
    args = parser.parse_args()
    return args

# KEY_MAP = [
    # ("unet.", ""),
    # ("lora_A", "lora.down"),
    # ("lora_B", "lora.up"),
    # ("proj_in.lora", "proj_in_lora"),
    # ("proj_out.lora", "proj_out_lora"),
    # ("to_k.lora", "processor.to_k_lora"),
    # ("to_q.lora", "processor.to_q_lora"),
    # ("to_v.lora", "processor.to_v_lora"),
    # ("to_out.0.lora", "processor.to_out_lora"),
    # ("net.0.proj.lora", "net.0.proj_lora"),
    # ("net.2.lora", "net.2_lora"),
    # ]

KEY_MAP = [
    ("unet.", ""),
    ("lora_A", "lora.down"),
    ("lora_B", "lora.up"),
    # (".lora.", "_lora."),
    ("to_", "processor.to_"),
    ("to_out.0", "to_out"),
    ]

def main(args):
    sd = {}
    with safe_open(args.input, framework="pt", device="cpu") as f:
        for key in f.keys():
            # new_key = key.replace("unet.", "")
            # new_key = new_key.replace("lora_A", "down")
            # new_key = new_key.replace("lora_B", "up")
            # new_key = new_key.replace(".to_", ".processor.to_")
            
            new_key = key
            for (orig, new) in KEY_MAP:
                new_key = new_key.replace(orig, new)
            
            sd[new_key] = f.get_tensor(key)
            
            if "lora.down" in new_key:
                rank = sd[new_key].shape[0]
                alpha = args.alpha if args.alpha is not None else rank
                alpha_key = new_key.replace("lora.down.weight", "alpha")
                if alpha_key not in sd.keys():
                    sd[alpha_key] = torch.tensor([alpha])
    
    output_path = args.output
    if output_path is None:
        name, ext = os.path.splitext(os.path.basename(args.input))
        name = name + "_sd" + ext
        output_path = os.path.join(os.path.dirname(args.input), name)
    
    print(f"saving lora to {output_path}")
    save_file(sd, output_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)
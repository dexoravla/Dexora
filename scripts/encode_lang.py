import os
import glob
import argparse

import torch
import yaml
import sys
sys.path.append("/baai-cwm-vepfs/cwm/zongzheng.zhang/Dex-RDT")
from models.multimodal_encoder.t5_encoder import T5Embedder


# Defaults
DEFAULT_GPU = 0
DEFAULT_MODEL_PATH = "google/t5-v1_1-xxl"
DEFAULT_CONFIG_PATH = "configs/base.yaml"
DEFAULT_SAVE_DIR = "outs/"
DEFAULT_ROOT_DIR = "data/ours/final"


def parse_args():
    parser = argparse.ArgumentParser(description="Encode language instructions for all actions under data/ours/action*/action.txt")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT_DIR, help="Root directory that contains action*/action.txt")
    parser.add_argument("--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="Directory to save .pt files")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Config file path for tokenizer length")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="HF model id or local path")
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU, help="GPU index")
    parser.add_argument("--offload-dir", type=str, default=None, help="Offload directory for low-VRAM usage")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    os.makedirs(args.save_dir, exist_ok=True)

    # Build file list: data/ours/action*/action.txt
    pattern = os.path.join(args.root, "action*", "instruction1.txt")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No action.txt found under: {pattern}")
        return

    device = torch.device(f"cuda:{args.gpu}")
    text_embedder = T5Embedder(
        from_pretrained=args.model,
        model_max_length=config["dataset"]["tokenizer_max_length"],
        device=device,
        use_offload_folder=args.offload_dir,
    )
    tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    for path in files:
        task_name = os.path.basename(os.path.dirname(path))  # e.g., action10
        with open(path, "r", encoding="utf-8") as f:
            instruction = f.read().strip()
        if not instruction:
            print(f"[WARN] Empty instruction in {path}, skip.")
            continue

        tokens = tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            truncation=True,
        )["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = text_encoder(tokens).last_hidden_state.detach().cpu()

        save_path = os.path.join(args.save_dir, f"{task_name}.pt")
        torch.save({
            "name": task_name,
            "instruction": instruction,
            "embeddings": pred,
        }, save_path)

        print(f'"{instruction}" from "{task_name}" -> encoded by "{args.model}" into {tuple(pred.shape)} saved to "{save_path}"')


if __name__ == "__main__":
    main()

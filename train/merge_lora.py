"""
Merge LoRA adapter weights into the base model and save the merged model.

Usage:
    python qwen/merge.py \
        --base-model Qwen/Qwen2.5-VL-7B-Instruct \
        --adapter-path /data/wada/misc/checkpoints/qwen_lora/epoch_1 \
        --output-dir /data/wada/misc/checkpoints/qwen_merged/epoch_1

The output directory can then be passed to eval.py via --adapter-path
(auto-detected as a merged model since it has no adapter_config.json).
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def merge(args):
    print(f"Base model  : {args.base_model}")
    print(f"Adapter     : {args.adapter_path}")
    print(f"Output      : {args.output_dir}")
    print("")

    print("Loading base model...")
    base = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir} ...")
    model.save_pretrained(args.output_dir)

    print("Saving processor...")
    processor = AutoProcessor.from_pretrained(args.adapter_path, use_fast=False)
    processor.save_pretrained(args.output_dir)

    print("Done.")


def parse_args():
    p = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    p.add_argument("--base-model",    default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--adapter-path",  required=True,
                   help="Path to LoRA adapter directory (e.g. .../epoch_1)")
    p.add_argument("--output-dir",    required=True,
                   help="Where to save the merged model")
    return p.parse_args()


if __name__ == "__main__":
    merge(parse_args())
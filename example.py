"""
Kaus — quick-start example.

Scores 5 sample (image, caption) pairs from data/sample/ using the
2-turn SpecReason + ExtendedScoreHead pipeline.

Usage:
    python example.py \
        --base-model-path  Qwen/Qwen2.5-VL-7B-Instruct \
        --spec-model-path  JLKang/ViSpec-Qwen2.5-VL-7B-Instruct \
        --score-head-path  <path/to/score_head.pt>
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor

from kaus.model.spec_model import SpecModel
from evaluate.kaus import (
    ExtendedScoreHead,
    build_turn1_inputs,
    build_turn2_inputs,
    expected_score_extended,
    run_specgen,
    run_score_generation_kaus,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "data", "sample")


def load_sample_data():
    with open(os.path.join(SAMPLE_DIR, "sample.json"), encoding="utf-8") as f:
        records = json.load(f)
    for r in records:
        r["image"] = Image.open(os.path.join(SAMPLE_DIR, r["image"])).convert("RGB")
    return records


def score(model, score_head, processor, image, caption, args, device):
    # Turn 1: speculative reasoning generation
    # run_specgen returns (inputs_dev, generated_ids_cpu, step_time, acceptance_len_list)
    inputs1 = build_turn1_inputs(processor, image, caption)
    _, generated_ids, _, _ = run_specgen(model, inputs1, args.max_new_tokens_reason, args, device)
    reason = processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # Turn 2: score via ExtendedScoreHead
    # run_score_generation_kaus returns (generated_ids, step_time, t2_text, score, digit_probs_list, sh_raw_text)
    inputs2 = build_turn2_inputs(processor, image, caption, reason)
    _, _, _, kaus_score, _, _ = run_score_generation_kaus(
        model, score_head, processor, inputs2, args.max_new_tokens_score, device
    )
    return kaus_score, reason


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path",  required=True)
    parser.add_argument("--spec-model-path",  required=True)
    parser.add_argument("--score-head-path",  required=True)
    parser.add_argument("--device",           default="cuda:0")
    parser.add_argument("--max-new-tokens-reason", type=int, default=256)
    parser.add_argument("--max-new-tokens-score",  type=int, default=16)
    parser.add_argument("--total-token",  type=int, default=30)
    parser.add_argument("--depth",        type=int, default=3)
    parser.add_argument("--top-k",        type=int, default=8)
    parser.add_argument("--num-q",        type=int, default=2)
    parser.add_argument("--temperature",  type=float, default=0.0)
    args = parser.parse_args()

    device = args.device

    print("Loading model...")
    model = SpecModel.from_pretrained(
        base_model_path=args.base_model_path,
        spec_model_path=args.spec_model_path,
        total_token=args.total_token,
        depth=args.depth,
        top_k=args.top_k,
        num_q=args.num_q,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    score_head = ExtendedScoreHead.load(args.score_head_path, device=device).to(device)
    processor  = AutoProcessor.from_pretrained(args.base_model_path)

    records = load_sample_data()

    print(f"\n{'caption':<45} {'gold':>5}  {'kaus':>5}")
    print("-" * 60)
    for r in records:
        kaus_score, _ = score(model, score_head, processor, r["image"], r["caption"], args, device)
        print(f"{r['caption']:<45} {r['score']:>5.2f}  {kaus_score:>5.2f}")


if __name__ == "__main__":
    main()

"""
Non-speculative Qwen evaluation using 2-turn RS format + ExtendedScoreHead.

ExtendedScoreHead: 13-class linear layer over virtual digits -1..+11.
Trained with "Score: " forced prefix; inference applies the head at the trailing
space position (last token of the prefix).

Turn 1 (reason): reason prompt → model.generate()
Turn 2 (score):  scoring prompt with reason in history → model.generate() with "Score: " forced prefix
                 → ExtendedScoreHead applied to hidden state at last prefix position (prefill, last token)
                 → expected_score_extended(probs)  =  0.1 * E[virtual_digit], clipped to [0, 1]

G-VEval-style prompts are used for both turns.

ExtendedScoreHead definition mirrors qwen/train_gveval_prompt_nonscore_prefill_extended_score_head.py.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluate.dataset_utils import get_expert_dataset

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "datasets"))
TARGET_H = 336


# ─── ExtendedScoreHead (mirrors qwen/train_gveval_prompt_extended_score_head.py) ─

class ExtendedScoreHead(nn.Module):
    """Linear projection: hidden_size → 13 virtual digit classes (-1 to +11)."""

    NUM_CLASSES = 13  # virtual digits: -1, 0, 1, ..., 10, 11

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, self.NUM_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "ExtendedScoreHead":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        head = cls(ckpt["hidden_size"])
        head.load_state_dict(ckpt["state_dict"])
        return head


def expected_score_extended(probs: torch.Tensor) -> float:
    """
    Expected score from ExtendedScoreHead output.
    probs: (13,) distribution over virtual digits -1..+11 (tensor indices 0..12).
    Returns 0.1 * E[virtual_digit], clipped to [0.0, 1.0].
    """
    virtual_digits = torch.arange(-1, 12, dtype=probs.dtype, device=probs.device)
    e = (probs * virtual_digits).sum()
    return float(torch.clamp(0.1 * e, 0.0, 1.0).detach())


# ─── Prompts ──────────────────────────────────────────────────────────────────

_SCORING_PROMPT = (
    "Based on your analysis, assign a score from 0.0 to 1.0.\n\n"
    "Generated caption:\n"
    "{caption}\n\n"
    "Score (0.0~1.0):"
)

_REASON_PROMPT = (
    "You will be given one sentence of visual caption generated from one image.\n\n"
    "Your task is to analyze the generated caption.\n\n"
    "Evaluation Criteria:\n\n"
    "The generated caption should accurately describe the important aspects of the image. "
    "Annotators were instructed to penalize captions which contained redundancies and excess information.\n\n"
    "Evaluation Steps:\n\n"
    "1. Carefully observe the image provided.\n"
    "2. Identify the main points of the visual content in the image.\n"
    "3. Assess how well the generated caption covers the main points of the visual content, "
    "and how much irrelevant or redundant information it contains.\n\n"
    "Generated caption:\n"
    "{caption}"
)


def load_reason_cache(jsonl_paths: list) -> dict:
    """Load cached reasons from JSONL files.

    Key: (dataset, image_id, caption) → turn1_text.
    Multiple files are merged; later files overwrite earlier ones on conflict.
    """
    cache: dict = {}
    for path in jsonl_paths:
        if not path or not os.path.exists(path):
            print(f"[warn] cached-reason-jsonl not found: {path}")
            continue
        n_before = len(cache)
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                key = (r.get("dataset", ""), r.get("image_id", ""), r.get("caption", ""))
                reason = r.get("turn1_text") or r.get("turn2_text")
                if reason:
                    cache[key] = reason
        print(f"[reason cache] +{len(cache) - n_before} entries from {path}  (total {len(cache)})")
    return cache


def _resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    return transforms.Resize((TARGET_H, max(1, int(w * TARGET_H / h))))(image)


# ─── Input builders ───────────────────────────────────────────────────────────

def build_turn1_inputs(processor, image, caption):
    """Turn 1: Reason generation prompt (image + reason prompt)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": _REASON_PROMPT.format(caption=caption)},
        ],
    }]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    return processor(images=[image], text=chat, return_tensors="pt")


def build_turn2_inputs(processor, image, caption, reason_text: str):
    """Turn 2: Score generation with reason in history + 'Score: ' forced prefix.

    Uses continue_final_message=True to avoid rope-index mismatch caused by
    manually appending tokens after processor() call.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _REASON_PROMPT.format(caption=caption)},
            ],
        },
        {
            "role": "assistant",
            "content": reason_text,
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": _SCORING_PROMPT.format(caption=caption)},
            ],
        },
        {
            "role": "assistant",
            "content": "Score: ",
        },
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=False,
                                         continue_final_message=True)
    return processor(images=[image], text=chat, return_tensors="pt")


# ─── Generation functions ─────────────────────────────────────────────────────

def run_reason_generation(model, processor, inputs, max_new_tokens, temperature, device):
    """Turn 1: reason generation, returns (gen_ids_cpu, step_time, t1_text)."""
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs_dev["input_ids"].shape[1]

    gen_kwargs = {"max_new_tokens": max_new_tokens, "return_dict_in_generate": True}
    if temperature > 0.0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = 1.0

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = model.generate(**inputs_dev, **gen_kwargs)
        torch.cuda.synchronize()
        step_time = time.time() - t0

    gen_ids = outputs.sequences[0, input_len:].cpu()
    t1_text = processor.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    return gen_ids, step_time, t1_text


def run_score_generation_extended_score_head(
    model, score_head, processor, inputs, max_new_tokens, device
):
    """Turn 2: greedy decode + ExtendedScoreHead at last prefix position.

    The input already ends with 'Score: ' tokens (forced prefix, same as training).
    ExtendedScoreHead is applied to the hidden state at the last input position
    (last token of 'Score: ') from the prefill step of model.generate().

    Hidden state indexing:
      outputs.hidden_states[0][-1]        last layer at prefill step
      outputs.hidden_states[0][-1][0, -1, :]   batch 0, last token of prefix

    Returns (gen_ids_cpu, step_time, t2_text, score, digit_probs_list, sh_raw_text).
      digit_probs_list : list of 13 floats (virtual digit distribution -1..+11)
      sh_raw_text      : "vd={argmax_virtual_digit}" for logging
    """
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs_dev["input_ids"].shape[1]

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.time()
        outputs = model.generate(
            **inputs_dev,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=max_new_tokens,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        torch.cuda.synchronize()
        step_time = time.time() - t0

    gen_ids = outputs.sequences[0, input_len:].cpu()
    gen_text = processor.tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
    t2_text = "Score: " + gen_text

    # Apply ExtendedScoreHead at '.' position (prefill step, last token of input).
    h_dot = outputs.hidden_states[0][-1][0, -1, :].float()
    logits = score_head(h_dot.unsqueeze(0))   # (1, 13)
    probs  = F.softmax(logits[0], dim=-1)     # (13,)

    score = expected_score_extended(probs)

    # Argmax virtual digit for logging: tensor_index - 1 = virtual_digit
    argmax_virtual = int(probs.argmax().item()) - 1
    sh_raw_text = f"vd={argmax_virtual}"

    digit_probs_list = probs.cpu().tolist()
    return gen_ids, step_time, t2_text, score, digit_probs_list, sh_raw_text


# ─── Evaluation loop ──────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map={"": device},
        attn_implementation=args.attn_implementation,
    ).eval()

    score_head = ExtendedScoreHead.load(args.score_head_path, device=device)
    score_head = score_head.to(device).eval()
    print(f"Loaded ExtendedScoreHead from {args.score_head_path}")
    print(f"  classes={ExtendedScoreHead.NUM_CLASSES}  (virtual digits -1..+11)")

    reason_cache = load_reason_cache(args.cached_reason_jsonl or [])

    all_results = {}
    all_records = []

    output_base = Path(args.output_prefix)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    out_path   = output_base.with_suffix(".json")
    jsonl_path = output_base.with_suffix(".jsonl")

    # Resume from existing JSON
    if out_path.exists():
        try:
            with out_path.open() as f:
                existing = json.load(f)
            all_results = existing.get("results", {})
            all_records = existing.get("outputs", [])
            print(f"Resuming from {out_path}: "
                  f"{len(all_results)} datasets done, {len(all_records)} records loaded")
        except Exception as e:
            print(f"[warn] Could not load existing JSON: {e}")

    # Build sample-level resume index from JSONL
    processed_keys: dict = {}
    processed_records: dict = {}
    if jsonl_path.exists():
        try:
            with jsonl_path.open(encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    ds  = r.get("dataset", "")
                    key = (r.get("image_id", ""), r.get("caption", ""))
                    processed_keys.setdefault(ds, set()).add(key)
                    processed_records.setdefault(ds, []).append(r)
            total = sum(len(v) for v in processed_keys.values())
            if total:
                print(f"  Sample-level resume: {total} records found in JSONL")
        except Exception as e:
            print(f"[warn] Could not load JSONL for sample-level resume: {e}")

    def save_checkpoint():
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"results": all_results, "outputs": all_records}, f,
                      ensure_ascii=False, indent=2)

    def append_to_jsonl(record):
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f" Dataset: {dataset_name}")
        print(f"{'='*60}")

        if dataset_name in all_results and "kendall_tau_b" in all_results[dataset_name]:
            v = all_results[dataset_name]
            print(f"  [skip] tau-b={100*v['kendall_tau_b']:.3f} (n={v['n']})")
            continue

        records = get_expert_dataset(dataset_name, args)

        if args.debug:
            records = records[:2]
        elif args.max_samples:
            records = records[:args.max_samples]

        already_done = processed_records.get(dataset_name, [])
        preds  = [r["pred"]  for r in already_done]
        golds  = [r["gold"]  for r in already_done]
        times  = [r["time"]  for r in already_done]
        done_keys = processed_keys.get(dataset_name, set())
        if done_keys:
            print(f"  Resuming {dataset_name}: {len(done_keys)} samples already done")
        existing_in_all = {
            (r["dataset"], r["image_id"], r["caption"]) for r in all_records
        }
        for r in already_done:
            if (r["dataset"], r["image_id"], r["caption"]) not in existing_in_all:
                all_records.append(r)

        prev_image_id = prev_caption = None
        prev_pred = prev_t1_text = prev_t2_text = None
        prev_digit_probs = prev_sh_raw = None

        for item in tqdm(records, desc=dataset_name):
            image_input = item["image"]
            caption     = item["caption"]
            human_score = item["gold"]
            if isinstance(image_input, str):
                image_id = os.path.relpath(image_input, DATA_ROOT)
            else:
                image_id = item.get("id", "none")

            if (image_id, caption) in done_keys:
                continue

            if image_id == prev_image_id and caption == prev_caption and prev_pred is not None:
                pred        = prev_pred
                t1_text     = prev_t1_text
                t2_text     = prev_t2_text
                digit_probs = prev_digit_probs
                sh_raw      = prev_sh_raw
                t1_time = t2_time = step_time = 0.0
            else:
                if isinstance(image_input, str):
                    try:
                        image = Image.open(image_input).convert("RGB")
                    except FileNotFoundError:
                        print(f"[WARN] Image not found: {image_input}, skipping.")
                        continue
                else:
                    image = image_input.convert("RGB")

                image = _resize(image)

                # ── Turn 1: Reason generation ──────────────────────────────
                cached_reason = reason_cache.get((dataset_name, image_id, caption))
                if cached_reason is not None:
                    t1_text = cached_reason
                    t1_time = 0.0
                else:
                    t1_inputs = build_turn1_inputs(processor, image, caption)
                    if t1_inputs["input_ids"].shape[1] > args.max_input_len:
                        for key in ("input_ids", "attention_mask"):
                            if key in t1_inputs:
                                t1_inputs[key] = t1_inputs[key][:, -args.max_input_len:]

                    _, t1_time, t1_text = run_reason_generation(
                        model, processor, t1_inputs, args.max_new_tokens_reason,
                        args.temperature, device)

                # ── Turn 2: Score with ExtendedScoreHead at '.' position ───
                t2_inputs = build_turn2_inputs(processor, image, caption, t1_text)
                if t2_inputs["input_ids"].shape[1] > args.max_input_len:
                    for key in ("input_ids", "attention_mask"):
                        if key in t2_inputs:
                            t2_inputs[key] = t2_inputs[key][:, -args.max_input_len:]

                _, t2_time, t2_text, pred, digit_probs, sh_raw = \
                    run_score_generation_extended_score_head(
                        model, score_head, processor, t2_inputs,
                        args.max_new_tokens_score, device)

                step_time = t1_time + t2_time

                prev_image_id    = image_id
                prev_caption     = caption
                prev_pred        = pred
                prev_t1_text     = t1_text
                prev_t2_text     = t2_text
                prev_digit_probs = digit_probs
                prev_sh_raw      = sh_raw

            preds.append(pred)
            golds.append(human_score)
            times.append(step_time)

            record = {
                "dataset":        dataset_name,
                "image_id":       image_id,
                "caption":        caption,
                "pred":           pred,
                "gold":           human_score,
                "time":           step_time,
                "t1_time":        t1_time,
                "t2_time":        t2_time,
                "turn1_text":     t1_text,
                "turn2_text":     t2_text,
                "score_head_raw": sh_raw,
                "digit_probs":    digit_probs,   # list of 13 floats (virtual digits -1..+11)
            }
            all_records.append(record)
            append_to_jsonl(record)

        if not preds:
            print(f"[WARN] No predictions for {dataset_name}, skipping metrics.")
            continue

        tau_b, _ = scipy.stats.kendalltau(preds, golds, variant="b")
        tau_c, _ = scipy.stats.kendalltau(preds, golds, variant="c")

        nonzero   = [r for r in all_records if r["dataset"] == dataset_name and r["time"] > 0]
        mean_t1   = float(np.mean([r["t1_time"] for r in nonzero])) if nonzero else float("nan")
        mean_t2   = float(np.mean([r["t2_time"] for r in nonzero])) if nonzero else float("nan")
        mean_time = float(np.mean(times)) if times else float("nan")

        all_results[dataset_name] = {
            "kendall_tau_b": tau_b,
            "kendall_tau_c": tau_c,
            "mean_t1":       mean_t1,
            "mean_t2":       mean_t2,
            "mean_time":     mean_time,
            "n":             len(preds),
        }

        print(f"{dataset_name}: tau-b={100*tau_b:.3f}  tau-c={100*tau_c:.3f} "
              f"(n={len(preds)}), t1={mean_t1:.2f}s t2={mean_t2:.2f}s total={mean_time:.2f}s")
        save_checkpoint()

    print("\n=== Summary ===")
    print(f"{'Dataset':<30} {'tau-b (×100)':>12} {'tau-c (×100)':>12}  n")
    print("-" * 60)
    for ds_name, v in all_results.items():
        print(f"{ds_name:<30} {100*v['kendall_tau_b']:>12.3f} {100*v['kendall_tau_c']:>12.3f}  {v['n']}")

    tau_bs = [v["kendall_tau_b"] for v in all_results.values() if not np.isnan(v["kendall_tau_b"])]
    tau_cs = [v["kendall_tau_c"] for v in all_results.values() if not np.isnan(v["kendall_tau_c"])]
    if tau_bs:
        print(f"{'Mean':<30} {100*np.mean(tau_bs):>12.3f} {100*np.mean(tau_cs):>12.3f}")

    return all_results


def parse_args():
    p = argparse.ArgumentParser(
        description="Non-speculative Qwen evaluation — 2-turn RS with G-VEval prompts + ExtendedScoreHead")
    p.add_argument("--model-path",            type=str, required=True)
    p.add_argument("--score-head-path",       type=str, required=True,
                   help="Path to trained ExtendedScoreHead .pt file (extended_score_head_epoch_N.pt)")
    p.add_argument(
        "--datasets", nargs="+",
        default=["nebula"],
        choices=["nebula", "flickr8k-ex", "flickr8k-cf", "composite",
                 "polaris-exp-train", "nebula-exp-train"],
    )
    p.add_argument("--device",                default="cuda:0")
    p.add_argument("--max-samples",           type=int,   default=None)
    p.add_argument("--max-new-tokens-score",  type=int,   default=16)
    p.add_argument("--max-new-tokens-reason", type=int,   default=256)
    p.add_argument("--max-input-len",         type=int,   default=4000)
    p.add_argument("--temperature",           type=float, default=0.0)
    p.add_argument("--attn-implementation",   default="sdpa", choices=["sdpa", "eager"])
    p.add_argument("--output-prefix",
                   default="out_qwen_vanilla_rs_gveval_prompt_extended_score_head")
    p.add_argument("--verbose",               action="store_true")
    p.add_argument("--debug",                 action="store_true",
                   help="2 samples per dataset only")
    p.add_argument("--cached-reason-jsonl",   nargs="*", default=[],
                   help="JSONL file(s) with pre-computed turn1_text to skip Turn 1 generation")
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())

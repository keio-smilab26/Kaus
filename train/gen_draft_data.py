"""
Generate ViSpec draft model training ckpts from pre-generated reasoning texts.

Instead of AR generation, does a single teacher-forcing forward pass on
(prompt + known turn1_text) to extract target model hidden states.
Much faster than ge_data_longcap.py.

Usage:
    python train/gen_draft_data.py \
        --model <target_model_path> \
        --reason-jsonl <path/eval_results.jsonl> \
        --dataset nebula-exp-train \
        --outdir <output_dir> \
        [--gpu-index 0] [--index 0] [--start 0] [--end -1] \
        [--max-len 4096]
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--reason-jsonl", type=str, required=True,
                    help="eval_results.jsonl with image_id / caption / turn1_text fields")
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset alias for get_expert_dataset (e.g. nebula-exp-train)")
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--gpu-index", type=int, default=0)
parser.add_argument("--index", type=int, default=0,
                    help="Process index — used as output sub-directory name")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=-1)
parser.add_argument("--max-len", type=int, default=4096)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

import json
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


from evaluate.dataset_utils import get_expert_dataset

TARGET_H = 336

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


def _resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    return transforms.Resize((TARGET_H, max(1, int(w * TARGET_H / h))))(image)


# ── Load reasoning texts ───────────────────────────────────────────────────────
print(f"Loading reasoning texts from {args.reason_jsonl} ...")
all_reasons = []
with open(args.reason_jsonl, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_reasons.append(json.loads(line))
print(f"  {len(all_reasons)} entries loaded")

# ── Load dataset records (images) ─────────────────────────────────────────────
print(f"Loading dataset '{args.dataset}' ...")
records = get_expert_dataset(args.dataset)

# Build lookup: (basename_image_id, caption) → record
# basename handles polaris ("polaris/polaris/images/<hash>") and nebula ("<hash>")
key_to_record: dict = {}
for r in records:
    rid = str(r["id"])
    cap = r.get("caption") or r.get("mt", "")
    key_to_record[(rid, cap)] = r
    key_to_record[(os.path.basename(rid), cap)] = r
print(f"  {len(records)} records loaded")


def _find_record(image_id: str, caption: str):
    r = key_to_record.get((image_id, caption))
    if r is None:
        r = key_to_record.get((os.path.basename(image_id), caption))
    return r


# ── Build matched list ─────────────────────────────────────────────────────────
valid = []
for entry in all_reasons:
    r = _find_record(entry["image_id"], entry["caption"])
    if r is not None and entry.get("turn1_text"):
        valid.append((entry, r))

print(f"  {len(valid)} matched (of {len(all_reasons)} reasoning entries)")

end = args.end if args.end >= 0 else len(valid)
local_items = valid[args.start:end]
local_indices = list(range(args.start, end))
print(f"[index {args.index}] {len(local_items)} samples assigned  (range {args.start}:{end})")

# ── Model + processor ─────────────────────────────────────────────────────────
processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
bigmodel = AutoModelForImageTextToText.from_pretrained(
    args.model,
    device_map={"": 0},
    torch_dtype="auto",
)
bigmodel.eval()

image_token_id = None
try:
    image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
except AttributeError:
    pass
if image_token_id is None:
    try:
        image_token_id = int(bigmodel.config.image_token_id)
    except AttributeError:
        pass

device = next(bigmodel.parameters()).device


# ── Teacher-forcing forward pass ───────────────────────────────────────────────
@torch.no_grad()
def ge(image, caption: str, turn1_text: str):
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    image = _resize(image)

    reason_text = _REASON_PROMPT.format(caption=caption)

    # Prompt-only (add_generation_prompt=True) to get input_len
    prompt_messages = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": reason_text}],
    }]
    prompt_chat = processor.apply_chat_template(prompt_messages, add_generation_prompt=True)
    prompt_inputs = processor(images=[image], text=prompt_chat, return_tensors="pt")
    input_len = prompt_inputs["input_ids"].shape[1]

    # Full sequence: prompt + turn1_text as assistant response
    full_messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": reason_text}],
        },
        {
            "role": "assistant",
            "content": turn1_text,
        },
    ]
    full_chat = processor.apply_chat_template(full_messages, add_generation_prompt=False)
    full_inputs = processor(images=[image], text=full_chat, return_tensors="pt")

    full_len = full_inputs["input_ids"].shape[1]
    if full_len <= input_len:
        return None  # turn1_text added no tokens

    # Truncate to max_len (keep from the start so prompt is always intact)
    if full_len > args.max_len:
        for key in ["input_ids", "attention_mask"]:
            if key in full_inputs:
                full_inputs[key] = full_inputs[key][:, :args.max_len]
        full_len = args.max_len
        if full_len <= input_len:
            return None

    inputs_dev = {k: v.to(device) for k, v in full_inputs.items()}
    outputs = bigmodel(**inputs_dev, output_hidden_states=True, return_dict=True)

    T = full_len
    # hidden_states[i]: [1, T, D] for each layer i
    # We take positions 0..T-2 (shift: h[i] predicts token i+1)
    inputs_embeds = outputs.hidden_states[0][:, :-1, :].squeeze(0).cpu()   # [T-1, D]
    hidden_state  = outputs.hidden_states[-1][:, :-1, :].squeeze(0).cpu()  # [T-1, D]

    # loss_mask: True from position (input_len-1) onward — the generated tokens
    loss_mask = torch.zeros(T - 1, dtype=torch.bool)
    loss_mask[input_len - 1:] = True

    if image_token_id is not None:
        image_mask = (full_inputs["input_ids"][0, :-1] == image_token_id)
    else:
        image_mask = torch.zeros(T - 1, dtype=torch.bool)

    return {
        "inputs_embeds": inputs_embeds,
        "hidden_state":  hidden_state,
        "loss_mask":     loss_mask,
        "image_mask":    image_mask,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
outdir = os.path.join(args.outdir, str(args.index))
os.makedirs(outdir, exist_ok=True)

skipped = 0
for (entry, record), global_idx in tqdm(
    zip(local_items, local_indices),
    total=len(local_items),
):
    td = ge(record["image"], entry["caption"], entry["turn1_text"])
    if td is None:
        skipped += 1
        continue
    torch.save(td, os.path.join(outdir, f"data_{global_idx}.ckpt"))

print(
    f"[index {args.index}] Done. "
    f"Saved {len(local_items) - skipped} files to {outdir}  (skipped {skipped})"
)

"""
LoRA fine-tuning of Qwen2.5-VL-7B-Instruct with self-generated reasoning.

Key differences from train_reason_score_gveval_prompt_exsh.py:
  - Qwen generates its own Turn 1 reasoning at each training step (AR, no grad)
  - Score head loss uses the self-generated reasoning as Turn 2 context
    → aligns training hidden states with test-time conditions
  - Reasoning regularization loss (CE, teacher-forcing on pre-collected reason)
    prevents the model's reasoning from drifting as score head loss updates LoRA
  - Option B backward: sh_loss.backward() first, then reason_loss.backward()
    to avoid holding two computation graphs in VRAM simultaneously

Training per step:
  1. model.generate() on Turn 1 inputs  (no grad) → generated_reason
  2. Build Turn 2 inputs with generated_reason dynamically
     → forward (output_hidden_states=True) → hidden state at last "Score: " prefix token
     → frozen score head → CE + KL + MSE loss → backward (free Turn 2 graph)
  3. Teacher-forcing forward on Turn 1 with pre-collected reason as labels
     → reasoning regularization loss → backward (free Turn 1 graph)
  4. optimizer.step()

Single-GPU:
    CUDA_VISIBLE_DEVICES=0 python qwen/train_reason_score_gveval_prompt_self_exsh.py \\
        --score-head-path /path/to/extended_score_head_epoch_3.pt \\
        --out-dir /path/to/lora_ckpt

Multi-GPU (DDP):
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \\
        qwen/train_reason_score_gveval_prompt_self_exsh.py \\
        --score-head-path /path/to/extended_score_head_epoch_3.pt \\
        --out-dir /path/to/lora_ckpt
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup


DATA_ROOT = "/home/initial/Documents/smilab/lab26/03B4progress26/dataset"
_TARGET_H = 336

# Must match qwen_vanilla_rs_gveval_prompt_extended_score_head.py
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

_SCORING_PROMPT = (
    "Based on your analysis, assign a score from 0.0 to 1.0.\n\n"
    "Generated caption:\n"
    "{caption}\n\n"
    "Score (0.0~1.0):"
)


# ── ExtendedScoreHead ──────────────────────────────────────────────────────────

class ExtendedScoreHead(nn.Module):
    NUM_CLASSES = 13  # virtual digits: -1, 0, 1, ..., 10, 11

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, self.NUM_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    @classmethod
    def load(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        head = cls(ckpt["hidden_size"])
        head.load_state_dict(ckpt["state_dict"])
        return head


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_score_space_offset(tokenizer) -> int:
    """Return offset of the last token within the tokenization of 'Score: '."""
    tokens = tokenizer("Score: ", add_special_tokens=False)["input_ids"]
    return len(tokens) - 1


def score_to_ce_label(score: float) -> int:
    vd = int(round(score * 10))
    return max(0, min(10, vd)) + 1


def gaussian_prior_extended(center: float, sigma2: float) -> torch.Tensor:
    vd = torch.arange(-1, 12, dtype=torch.float32)
    return torch.softmax(-0.5 * (vd - center) ** 2 / sigma2, dim=0)


# ── Distributed helpers ───────────────────────────────────────────────────────

def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank, world_size


# ── Dataset ───────────────────────────────────────────────────────────────────
# Identical to train_reason_score_gveval_prompt_exsh.py.
# pre-collected reason is used as the teacher-forcing target (reasoning reg. loss).

class RSDataset(Dataset):
    def __init__(
        self,
        polaris_json: str,
        nebula_json: str,
        polaris_image_dir: str,
        nebula_hf: str = "Ka2ukiMatsuda/Nebula",
        bin_size: float = 0.0,
        max_samples_polaris: int = 0,
        max_samples_nebula: int = 0,
        reason_jsonl_polaris: str = "",
        reason_jsonl_nebula: str = "",
    ):
        self.samples = []
        self.nebula_hf_ds = None
        seen = set()

        def _clean(s: str) -> str:
            return s.strip().rstrip(".")

        reason_lookup: dict[tuple[str, str], str] = {}
        for jsonl_path in [reason_jsonl_polaris, reason_jsonl_nebula]:
            if jsonl_path and os.path.exists(jsonl_path):
                with open(jsonl_path) as f:
                    for line in f:
                        d = json.loads(line)
                        img_key = os.path.basename(d["image_id"])
                        reason_lookup[(img_key, d["caption"])] = d["turn1_text"]
                print(f"[reason cache] loaded {len(reason_lookup)} entries from {jsonl_path}")
        use_reason_cache = len(reason_lookup) > 0

        if os.path.exists(polaris_json):
            with open(polaris_json) as f:
                entries = json.load(f)
            if max_samples_polaris > 0:
                entries = entries[:max_samples_polaris]
            n_skip = 0
            for e in entries:
                key = (e["imgid"], e["mt"])
                if key in seen:
                    n_skip += 1
                    continue
                seen.add(key)
                score = float(e["score"])
                if bin_size > 0:
                    score = max(0.0, min(1.0, round(round(score / bin_size) * bin_size, 10)))
                if use_reason_cache:
                    reason = reason_lookup.get((e["imgid"], e["mt"]))
                    if reason is None:
                        n_skip += 1
                        continue
                else:
                    exp = e.get("explanation", {})
                    reason = (
                        f"Fluency: {_clean(exp.get('fluency', ''))}.\n"
                        f"Relevance: {_clean(exp.get('relevance', ''))}.\n"
                        f"Descriptiveness: {_clean(exp.get('descriptiveness', ''))}."
                    ).strip()
                self.samples.append({
                    "source":     "polaris",
                    "image_path": os.path.join(polaris_image_dir, e["imgid"]),
                    "caption":    e["mt"],
                    "score":      score,
                    "reason":     reason,
                })
            if n_skip:
                print(f"[Polaris] skipped {n_skip}")
            print(f"[Polaris] loaded {len(self.samples)} samples")
        else:
            print(f"[warn] polaris json not found: {polaris_json}")

        if os.path.exists(nebula_json):
            print(f"Loading Nebula HF dataset ({nebula_hf})...")
            self.nebula_hf_ds = load_dataset(nebula_hf, split="train")
            nebula_imgid_to_idx = {item["file_name"]: i for i, item in enumerate(self.nebula_hf_ds)}
            print(f"  HF index built: {len(nebula_imgid_to_idx)} images")

            with open(nebula_json) as f:
                entries = json.load(f)
            if max_samples_nebula > 0:
                entries = entries[:max_samples_nebula]
            n_skip = n_missing = 0
            n_before = len(self.samples)
            seen_nebula: set = set()
            for e in entries:
                key = (e["imgid"], e["mt"])
                if key in seen_nebula:
                    n_skip += 1
                    continue
                if e["imgid"] not in nebula_imgid_to_idx:
                    n_missing += 1
                    continue
                seen_nebula.add(key)
                score = float(e["score"])
                if bin_size > 0:
                    score = max(0.0, min(1.0, round(round(score / bin_size) * bin_size, 10)))
                if use_reason_cache:
                    reason = reason_lookup.get((e["imgid"], e["mt"]))
                    if reason is None:
                        n_missing += 1
                        continue
                else:
                    exp = e.get("explanation", {})
                    reason = (
                        f"Fluency: {_clean(exp.get('fluency', ''))}.\n"
                        f"Relevance: {_clean(exp.get('relevance', ''))}.\n"
                        f"Descriptiveness: {_clean(exp.get('descriptiveness', ''))}."
                    ).strip()
                self.samples.append({
                    "source":  "nebula",
                    "hf_idx":  nebula_imgid_to_idx[e["imgid"]],
                    "caption": e["mt"],
                    "score":   score,
                    "reason":  reason,
                })
            if n_skip:
                print(f"[Nebula] skipped {n_skip} duplicates")
            if n_missing:
                print(f"[Nebula] skipped {n_missing} entries with no HF image / reason")
            print(f"[Nebula] loaded {len(self.samples) - n_before} samples")
        else:
            print(f"[warn] nebula json not found: {nebula_json}")

        print(f"Total: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if s["source"] == "polaris":
            image = Image.open(s["image_path"])
        else:
            image = self.nebula_hf_ds[s["hf_idx"]]["image"]
        return {**s, "image": image}


# ── Collate ───────────────────────────────────────────────────────────────────

def make_collate(processor, max_len: int, max_reason_len: int):
    """
    Returns a collate function for batch_size=1.

    Each returned batch dict contains:
      t1_*        : Turn 1 inputs for AR reason generation (torch.no_grad)
      tf_*        : Teacher-forcing inputs for reasoning regularization loss
                    (pre-collected reason as labels; prompt masked with -100)
      caption     : str  — for dynamic Turn 2 construction inside training loop
      image       : PIL.Image — for dynamic Turn 2 construction (num_workers=0)
      ce_label    : int
      gold        : float
    """

    def collate(batch):
        s = batch[0]  # batch_size=1

        try:
            image = s["image"].convert("RGB")
        except Exception as e:
            print(f"[warn] image load failed ({s.get('source', '?')}): {e}")
            return None

        w, h = image.size
        image = transforms.Resize((_TARGET_H, max(1, int(w * _TARGET_H / h))))(image)

        caption = s["caption"]
        pre_collected_reason = s["reason"]

        user_content = [
            {"type": "image"},
            {"type": "text", "text": _REASON_PROMPT.format(caption=caption)},
        ]

        # ── Turn 1: inputs for AR generation ──────────────────────────────────
        t1_messages = [{"role": "user", "content": user_content}]
        t1_chat = processor.apply_chat_template(t1_messages, add_generation_prompt=True)
        t1_enc = processor(images=[image], text=t1_chat, return_tensors="pt")

        t1_prompt_len = t1_enc["input_ids"].shape[1]
        if t1_prompt_len > max_len:
            t1_enc["input_ids"] = t1_enc["input_ids"][:, :max_len]
            if "attention_mask" in t1_enc:
                t1_enc["attention_mask"] = t1_enc["attention_mask"][:, :max_len]
            t1_prompt_len = max_len

        # ── Turn 1: teacher-forcing inputs for reasoning regularization loss ───
        # Input : [image | user turn | <|im_start|>assistant\n | pre_collected_reason | <|im_end|>]
        # Labels: -100 for prompt (up to and including <|im_start|>assistant\n),
        #         token IDs for the reason + EOS
        # t1_prompt_len from above marks the boundary between prompt and reason.
        tf_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": pre_collected_reason},
        ]
        tf_chat = processor.apply_chat_template(tf_messages, add_generation_prompt=False)
        tf_enc = processor(images=[image], text=tf_chat, return_tensors="pt")

        tf_ids = tf_enc["input_ids"][0]                                    # (TF,)
        tf_attn = tf_enc.get("attention_mask", torch.ones(1, tf_ids.shape[0]))[0]  # (TF,)

        if tf_ids.shape[0] > max_reason_len:
            tf_ids  = tf_ids[:max_reason_len]
            tf_attn = tf_attn[:max_reason_len]

        tf_labels = tf_ids.clone()
        mask_end = min(t1_prompt_len, tf_ids.shape[0])
        tf_labels[:mask_end] = -100   # mask prompt; supervise only on reason tokens

        if (tf_labels != -100).sum() == 0:
            print("[warn] pre-collected reason fully truncated; skipping sample")
            return None

        result = {
            # Turn 1 — AR generation
            "t1_input_ids":      t1_enc["input_ids"],          # (1, T1)
            "t1_attention_mask": t1_enc.get("attention_mask"), # (1, T1) or None
            "t1_pixel_values":   t1_enc["pixel_values"],
            "t1_prompt_len":     t1_prompt_len,
            # Turn 1 — teacher-forcing (reasoning regularization)
            "tf_input_ids":      tf_ids.unsqueeze(0),          # (1, TF)
            "tf_attention_mask": tf_attn.unsqueeze(0),         # (1, TF)
            "tf_pixel_values":   tf_enc["pixel_values"],
            "tf_labels":         tf_labels.unsqueeze(0),       # (1, TF)
            # Meta — for dynamic Turn 2 construction in training loop
            "caption":           caption,
            "image":             image,
            # Score labels
            "ce_label":          score_to_ce_label(s["score"]),
            "gold":              s["score"],
        }
        if "image_grid_thw" in t1_enc:
            result["t1_image_grid_thw"] = t1_enc["image_grid_thw"]
        if "image_grid_thw" in tf_enc:
            result["tf_image_grid_thw"] = tf_enc["image_grid_thw"]

        return result

    return collate


# ── Dynamic Turn 2 construction ───────────────────────────────────────────────

def build_turn2_inputs(processor, image, caption: str, reason_text: str, max_len: int):
    """
    Build Turn 2 forward-pass inputs (with 'Score: ' forced prefix) using the
    model's self-generated reason_text as conversational context.

    Returns a dict with input_ids, attention_mask, pixel_values, (image_grid_thw),
    and t2_prompt_len (length before 'Score: ' prefix, used to locate the last prefix position).
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": _REASON_PROMPT.format(caption=caption)},
            ],
        },
        {"role": "assistant", "content": reason_text},
        {
            "role": "user",
            "content": [{"type": "text", "text": _SCORING_PROMPT.format(caption=caption)}],
        },
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)
    enc  = processor(images=[image], text=chat, return_tensors="pt")

    t2_prompt_len = enc["input_ids"].shape[1]

    # Append "Score: " prefix (same as inference time)
    prefix_ids = processor.tokenizer.encode("Score: ", add_special_tokens=False)
    prefix_t   = torch.tensor([prefix_ids], dtype=enc["input_ids"].dtype)
    input_ids  = torch.cat([enc["input_ids"], prefix_t], dim=1)

    if input_ids.shape[1] > max_len:
        input_ids = input_ids[:, :max_len]

    result = {
        "input_ids":      input_ids,
        "attention_mask": torch.ones(1, input_ids.shape[1], dtype=torch.long),
        "pixel_values":   enc["pixel_values"],
        "t2_prompt_len":  t2_prompt_len,
    }
    if "image_grid_thw" in enc:
        result["image_grid_thw"] = enc["image_grid_thw"]
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    local_rank, world_size = setup_dist()
    main_process = local_rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)

    # ── Frozen score head ─────────────────────────────────────────────────────
    score_head = ExtendedScoreHead.load(args.score_head_path, device=device).to(device).eval()
    for p in score_head.parameters():
        p.requires_grad = False
    if main_process:
        print(f"Loaded frozen score head from {args.score_head_path}")

    # ── Qwen + LoRA ───────────────────────────────────────────────────────────
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=args.attn_implementation,
    )

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    if main_process:
        model.print_trainable_parameters()
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # model.module gives the underlying PeftModel (needed for .generate())
    raw_model = model.module if isinstance(model, DDP) else model

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    dataset = RSDataset(
        polaris_json=args.polaris_json,
        nebula_json=args.nebula_json,
        polaris_image_dir=args.polaris_image_dir,
        nebula_hf=args.nebula_hf,
        bin_size=args.bin_size,
        max_samples_polaris=args.max_samples_polaris,
        max_samples_nebula=args.max_samples_nebula,
        reason_jsonl_polaris=args.reason_jsonl_polaris,
        reason_jsonl_nebula=args.reason_jsonl_nebula,
    )

    if args.bin_size > 0:
        bin_counts = Counter(s["score"] for s in dataset.samples)
        weights    = torch.DoubleTensor([1.0 / bin_counts[s["score"]] for s in dataset.samples])
        sampler    = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=1,   # fixed: AR generation per step requires bs=1
        shuffle=(sampler is None and world_size == 1),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=make_collate(processor, args.max_len, args.max_reason_len),
        pin_memory=False,
    )

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(loader) * args.epochs
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.05 * total_steps),
        num_training_steps=total_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        stats = {"loss": 0.0, "sh": 0.0, "rsn": 0.0, "ce": 0.0, "kl": 0.0, "mse": 0.0}
        n = 0

        pbar = tqdm(loader, disable=not main_process, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch in pbar:
            if batch is None:
                continue

            image   = batch["image"]     # PIL image (num_workers=0)
            caption = batch["caption"]   # str
            ce_label = batch["ce_label"] # int
            gold     = batch["gold"]     # float

            # ── Step 1: AR generation of Turn 1 reason (no grad) ──────────────
            # Use raw_model.generate() to bypass DDP wrapper (no backward needed).
            raw_model.eval()
            with torch.no_grad():
                t1_gen_kwargs = {
                    "input_ids":    batch["t1_input_ids"].to(device),
                    "pixel_values": batch["t1_pixel_values"].to(device),
                }
                if batch.get("t1_attention_mask") is not None:
                    t1_gen_kwargs["attention_mask"] = batch["t1_attention_mask"].to(device)
                if "t1_image_grid_thw" in batch:
                    t1_gen_kwargs["image_grid_thw"] = batch["t1_image_grid_thw"].to(device)

                gen_ids = raw_model.generate(
                    **t1_gen_kwargs,
                    do_sample=False,
                    temperature=1.0,
                    max_new_tokens=args.max_new_tokens_reason,
                )

            t1_prompt_len = batch["t1_prompt_len"]
            generated_reason = processor.tokenizer.decode(
                gen_ids[0, t1_prompt_len:].tolist(), skip_special_tokens=True
            ).strip()
            del gen_ids   # free VRAM before gradient forward passes

            raw_model.train()

            # ── Step 2: Build Turn 2 inputs dynamically ────────────────────────
            # Turn 2 uses the self-generated reason as context, matching test time.
            t2_dict = build_turn2_inputs(processor, image, caption, generated_reason, args.max_len)

            # space_pos: last token of prefill = last token of "Score: " prefix
            # Matches eval code: outputs.hidden_states[0][-1][0, -1, :]
            space_pos = t2_dict["input_ids"].shape[1] - 1

            t2_inputs = {
                "input_ids":      t2_dict["input_ids"].to(device),
                "attention_mask": t2_dict["attention_mask"].to(device),
                "pixel_values":   t2_dict["pixel_values"].to(device),
            }
            if "image_grid_thw" in t2_dict:
                t2_inputs["image_grid_thw"] = t2_dict["image_grid_thw"].to(device)

            # ── Step 3: Score head forward + backward ──────────────────────────
            outputs    = model(**t2_inputs, output_hidden_states=True, return_dict=True)
            h_dot      = outputs.hidden_states[-1][0, space_pos, :]
            h_batch    = h_dot.unsqueeze(0).to(dtype=next(score_head.parameters()).dtype)
            logits     = score_head(h_batch).float()   # (1, 13)

            labels_t = torch.tensor([ce_label], device=device)
            ce_loss  = F.cross_entropy(logits, labels_t)
            sh_loss  = ce_loss

            kl_loss = torch.tensor(0.0)
            if args.kl_weight > 0.0:
                prior   = gaussian_prior_extended(gold * 10, args.sigma2).to(device)
                kl_loss = F.kl_div(
                    F.log_softmax(logits, dim=-1), prior.unsqueeze(0), reduction="batchmean"
                )
                sh_loss = sh_loss + args.kl_weight * kl_loss

            mse_loss = torch.tensor(0.0)
            if args.mse_weight > 0.0:
                vd        = torch.arange(-1, 12, dtype=torch.float32, device=device)
                probs     = F.softmax(logits, dim=-1)
                exp_score = torch.clamp(0.1 * (probs * vd).sum(dim=-1), 0.0, 1.0)
                gold_t    = torch.tensor([gold], dtype=torch.float32, device=device)
                mse_loss  = F.mse_loss(exp_score, gold_t)
                sh_loss   = sh_loss + args.mse_weight * mse_loss

            # Option B: backward Turn 2 graph first to free VRAM before Turn 1 TF forward
            sh_loss.backward()

            # ── Step 4: Reasoning regularization loss + backward ───────────────
            tf_inputs = {
                "input_ids":      batch["tf_input_ids"].to(device),
                "attention_mask": batch["tf_attention_mask"].to(device),
                "pixel_values":   batch["tf_pixel_values"].to(device),
                "labels":         batch["tf_labels"].to(device),
            }
            if "tf_image_grid_thw" in batch:
                tf_inputs["image_grid_thw"] = batch["tf_image_grid_thw"].to(device)

            tf_outputs   = model(**tf_inputs, return_dict=True)
            reason_loss  = tf_outputs.loss

            (args.reason_weight * reason_loss).backward()

            # ── Step 5: Optimizer step ─────────────────────────────────────────
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            total_val = sh_loss.item() + args.reason_weight * reason_loss.item()
            stats["loss"] += total_val
            stats["sh"]   += sh_loss.item()
            stats["rsn"]  += reason_loss.item()
            stats["ce"]   += ce_loss.item()
            stats["kl"]   += kl_loss.item()
            stats["mse"]  += mse_loss.item()
            n += 1
            global_step += 1

            if main_process:
                pbar.set_postfix({"sh": f"{sh_loss.item():.3f}", "rsn": f"{reason_loss.item():.3f}"})
                wandb.log({
                    "train/loss":        total_val,
                    "train/sh_loss":     sh_loss.item(),
                    "train/reason_loss": reason_loss.item(),
                    "train/ce_loss":     ce_loss.item(),
                    "train/kl_loss":     kl_loss.item(),
                    "train/mse_loss":    mse_loss.item(),
                    "train/lr":          scheduler.get_last_lr()[0],
                }, step=global_step)

        if main_process:
            d = max(1, n)
            print(
                f"Epoch {epoch + 1}: loss={stats['loss']/d:.4f}  "
                f"sh={stats['sh']/d:.4f}  reason={stats['rsn']/d:.4f}  "
                f"ce={stats['ce']/d:.4f}  kl={stats['kl']/d:.4f}  mse={stats['mse']/d:.4f}"
            )
            wandb.log({"train/epoch_loss": stats["loss"] / d, "epoch": epoch + 1}, step=global_step)

            save_path = out_dir / f"epoch_{epoch + 1}"
            raw_model.save_pretrained(save_path)
            processor.save_pretrained(save_path)
            print(f"Saved LoRA adapter → {save_path}")

    if world_size > 1:
        dist.destroy_process_group()
    if main_process:
        wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(
        description="LoRA fine-tuning of Qwen2.5-VL with self-generated reasoning + frozen ExtendedScoreHead"
    )
    p.add_argument("--model",                  default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--score-head-path",        required=True)
    p.add_argument("--polaris-json",           default="EXPERT/exp_datasets/polaris_exp_train.json")
    p.add_argument("--nebula-json",            default="EXPERT/exp_datasets/nebula_exp_train.json")
    p.add_argument("--polaris-image-dir",      default="data/images/Polaris")
    p.add_argument("--nebula-hf",              default="Ka2ukiMatsuda/Nebula")
    p.add_argument("--max-samples-polaris",    type=int,   default=0)
    p.add_argument("--max-samples-nebula",     type=int,   default=0)
    p.add_argument("--reason-jsonl-polaris",   default="")
    p.add_argument("--reason-jsonl-nebula",    default="")
    p.add_argument("--out-dir",                required=True)
    p.add_argument("--epochs",                 type=int,   default=1)
    p.add_argument("--lr",                     type=float, default=2e-5)
    p.add_argument("--lora-r",                 type=int,   default=128)
    p.add_argument("--lora-alpha",             type=int,   default=256)
    p.add_argument("--lora-dropout",           type=float, default=0.05)
    p.add_argument("--max-len",                type=int,   default=1200,
                   help="Max token length for Turn 2 (score head) inputs")
    p.add_argument("--max-reason-len",         type=int,   default=1200,
                   help="Max token length for Turn 1 teacher-forcing inputs")
    p.add_argument("--max-new-tokens-reason",  type=int,   default=256,
                   help="Max new tokens for AR Turn 1 generation")
    p.add_argument("--bin-size",               type=float, default=0.1)
    p.add_argument("--kl-weight",              type=float, default=1.0)
    p.add_argument("--mse-weight",             type=float, default=10.0)
    p.add_argument("--sigma2",                 type=float, default=1.0)
    p.add_argument("--reason-weight",          type=float, default=1.0,
                   help="Weight of reasoning regularization loss (CE on pre-collected reason)")
    p.add_argument("--attn-implementation",    default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--num-workers",            type=int,   default=0)
    p.add_argument("--wandb-project",          default="qwen-lora-self-exsh")
    p.add_argument("--wandb-run",              default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

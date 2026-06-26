"""
Extended Score-Head training: limited MAE variant.

Loss: MAE(expected_score, gold) only for samples where |argmax_score - gold| < 0.05.
      All other samples are skipped (no gradient).

Intended as a "final adjustment" after CE/KL pre-training (--init-head).
The argmax is already close to gold for most samples; this loss nudges the
soft expectation further toward gold without noise from mismatched samples.

  expected_score = clamp(0.1 * Σ probs[i] * vd[i], 0.0, 1.0)
  argmax_score   = (argmax_idx - 1) / 10   (buffer classes vd=-1,+11 はmatch不可)
  match          = |argmax_score - gold| < 0.05

Format: reason → score  (RS, same as mae_integral variant)
  Turn 2 target : score prompt → "Score: " (forced prefix, trailing space)
  Hidden state at " " position → 13-class ExtendedScoreHead
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup


# ─── Extended Score Head ──────────────────────────────────────────────────────

class ExtendedScoreHead(nn.Module):
    """Linear projection: hidden_size → 13 virtual digit classes (-1 to +11)."""

    NUM_CLASSES = 13  # virtual digits: -1, 0, 1, 2, ..., 10, 11

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, self.NUM_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., hidden_size) → (..., 13) logits"""
        return self.linear(x)

    def save(self, path: str | Path) -> None:
        torch.save({
            "hidden_size": self.linear.in_features,
            "num_classes": self.NUM_CLASSES,
            "state_dict":  self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "ExtendedScoreHead":
        ckpt = torch.load(path, map_location=device, weights_only=True)
        head = cls(ckpt["hidden_size"])
        head.load_state_dict(ckpt["state_dict"])
        return head


# ─── Helper functions ─────────────────────────────────────────────────────────

def find_score_space_offset(tokenizer) -> int:
    """
    Tokenize "Score: " and return the index of the last token (the trailing space).
    Used to locate the " " position in the appended "Score: " prefix during collation.
    The trailing space ensures the model cannot insert an extra space before the score.
    """
    tokens = tokenizer("Score: ", add_special_tokens=False)["input_ids"]
    offset = len(tokens) - 1
    decoded_last = tokenizer.decode([tokens[offset]], skip_special_tokens=False)
    print(f"[prefix] 'Score: ' tokenizes to {len(tokens)} tokens; "
          f"last token = {tokens[offset]!r} ({decoded_last!r})")
    return offset


def score_to_ce_label(score: float) -> int:
    """
    Convert gold score to CE label (tensor index into 13-class ExtendedScoreHead).
      virtual_digit = round(score * 10), clamped to [0, 10]
      tensor_index  = virtual_digit + 1
    Examples:
      gold=0.0 → virtual 0 → index 1
      gold=0.9 → virtual 9 → index 10
      gold=1.0 → virtual 10 → index 11
    """
    virtual_digit = int(round(score * 10))
    virtual_digit = max(0, min(10, virtual_digit))
    return virtual_digit + 1


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

_TARGET_H = 336


# ─── Dataset ──────────────────────────────────────────────────────────────────

class ScoreHeadDataset(Dataset):
    def __init__(self, polaris_json: str, nebula_json: str,
                 polaris_image_dir: str,
                 nebula_hf: str = "Ka2ukiMatsuda/Nebula",
                 bin_size: float = 0.0,
                 max_samples_polaris: int = 0,
                 max_samples_nebula: int = 0,
                 reason_jsonl_polaris: str = "",
                 reason_jsonl_nebula: str = ""):
        self.samples = []
        seen = set()
        self.nebula_hf_ds = None
        self._nebula_imgid_to_idx: dict[str, int] = {}

        def _clean(s: str) -> str:
            return s.strip().rstrip(".")

        reason_lookup: dict[tuple[str, str], str] = {}
        for jsonl_path in [reason_jsonl_polaris, reason_jsonl_nebula]:
            if jsonl_path and os.path.exists(jsonl_path):
                with open(jsonl_path) as f:
                    for line in f:
                        d = json.loads(line)
                        img_key = os.path.basename(d["image_id"])
                        key = (img_key, d["caption"])
                        reason_lookup[key] = d["turn1_text"]
                print(f"[reason cache] loaded {len(reason_lookup)} entries from {jsonl_path}")
        use_reason_cache = len(reason_lookup) > 0

        if os.path.exists(polaris_json):
            with open(polaris_json) as f:
                entries = json.load(f)
            if max_samples_polaris > 0:
                entries = entries[:max_samples_polaris]
                print(f"[Polaris] limiting to {max_samples_polaris} entries")
            n_skip = 0
            for e in entries:
                key = (e["imgid"], e["mt"])
                if key in seen:
                    n_skip += 1
                    continue
                seen.add(key)
                score = float(e["score"])
                if bin_size > 0:
                    score = round(round(score / bin_size) * bin_size, 10)
                    score = max(0.0, min(1.0, score))
                if use_reason_cache:
                    reason_text = reason_lookup.get((e["imgid"], e["mt"]))
                    if reason_text is None:
                        n_skip += 1
                        continue
                else:
                    exp = e.get("explanation", {})
                    reason_text = (
                        f"Fluency: {_clean(exp.get('fluency', ''))}.\n"
                        f"Relevance: {_clean(exp.get('relevance', ''))}.\n"
                        f"Descriptiveness: {_clean(exp.get('descriptiveness', ''))}."
                    ).strip()
                self.samples.append({
                    "source":     "polaris",
                    "image_path": os.path.join(polaris_image_dir, e["imgid"]),
                    "caption":    e["mt"],
                    "score":      score,
                    "reason":     reason_text,
                })
            if n_skip:
                print(f"[Polaris] skipped {n_skip} entries")
            print(f"[Polaris] loaded {len(self.samples)} samples")
        else:
            print(f"[warn] polaris json not found: {polaris_json}")

        seen = set()

        if os.path.exists(nebula_json):
            print(f"Loading Nebula HF dataset ({nebula_hf})...")
            self.nebula_hf_ds = load_dataset(nebula_hf, split="train")
            self._nebula_imgid_to_idx = {
                item["file_name"]: i for i, item in enumerate(self.nebula_hf_ds)
            }
            print(f"  HF index built: {len(self._nebula_imgid_to_idx)} images")

            with open(nebula_json) as f:
                entries = json.load(f)
            if max_samples_nebula > 0:
                entries = entries[:max_samples_nebula]
                print(f"[Nebula] limiting to {max_samples_nebula} entries")
            n_skip = n_missing = 0
            n_before = len(self.samples)
            for e in entries:
                key = (e["imgid"], e["mt"])
                if key in seen:
                    n_skip += 1
                    continue
                if e["imgid"] not in self._nebula_imgid_to_idx:
                    n_missing += 1
                    continue
                seen.add(key)
                score = float(e["score"])
                if bin_size > 0:
                    score = round(round(score / bin_size) * bin_size, 10)
                    score = max(0.0, min(1.0, score))
                if use_reason_cache:
                    reason_text = reason_lookup.get((e["imgid"], e["mt"]))
                    if reason_text is None:
                        n_missing += 1
                        continue
                else:
                    exp = e.get("explanation", {})
                    reason_text = (
                        f"Fluency: {_clean(exp.get('fluency', ''))}.\n"
                        f"Relevance: {_clean(exp.get('relevance', ''))}.\n"
                        f"Descriptiveness: {_clean(exp.get('descriptiveness', ''))}."
                    ).strip()
                self.samples.append({
                    "source":   "nebula",
                    "hf_idx":   self._nebula_imgid_to_idx[e["imgid"]],
                    "caption":  e["mt"],
                    "score":    score,
                    "reason":   reason_text,
                })
            if n_skip:
                print(f"[Nebula] skipped {n_skip} duplicates")
            if n_missing:
                print(f"[Nebula] skipped {n_missing} entries with no HF image")
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


# ─── Collate ──────────────────────────────────────────────────────────────────

def make_collate(processor, max_len: int):
    tokenizer = processor.tokenizer

    # Pre-compute the "Score: " prefix tokens and the offset of " " (trailing space).
    # These are constant for a given tokenizer.
    score_space_token_ids = tokenizer("Score: ", add_special_tokens=False)["input_ids"]
    eos_token_ids = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    prefix_tensor = torch.tensor(
        score_space_token_ids + eos_token_ids, dtype=torch.long
    ).unsqueeze(0)  # (1, prefix_len)

    space_offset = find_score_space_offset(tokenizer)

    def collate(batch):
        results = []
        for s in batch:
            try:
                image = s["image"].convert("RGB")
            except Exception as e:
                print(f"[warn] image load failed ({s.get('source', '?')}): {e}")
                continue

            w, h = image.size
            image = transforms.Resize((_TARGET_H, max(1, int(w * _TARGET_H / h))))(image)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": _REASON_PROMPT.format(caption=s["caption"])},
                    ],
                },
                {"role": "assistant", "content": s["reason"]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _SCORING_PROMPT.format(caption=s["caption"])},
                    ],
                },
            ]
            chat = processor.apply_chat_template(messages, add_generation_prompt=True)
            p_enc = processor(images=[image], text=chat, return_tensors="pt")
            prompt_len = p_enc["input_ids"].shape[1]

            # Always append "Score: " + eos as forced prefix (gold score not encoded).
            input_ids = torch.cat([p_enc["input_ids"], prefix_tensor], dim=1)
            if input_ids.shape[1] > max_len:
                input_ids = input_ids[:, :max_len]
                prompt_len = min(prompt_len, max_len)

            space_pos = prompt_len + space_offset
            if space_pos >= input_ids.shape[1]:
                print(f"[warn] space_pos {space_pos} >= seq_len {input_ids.shape[1]}, skipping")
                continue

            ce_label = score_to_ce_label(s["score"])
            attention_mask = torch.ones_like(input_ids)

            item = {
                "input_ids":      input_ids[0],
                "attention_mask": attention_mask[0],
                "pixel_values":   p_enc["pixel_values"],
                "space_pos":      space_pos,
                "ce_label":       ce_label,
                "gold":           s["score"],
            }
            if "image_grid_thw" in p_enc:
                item["image_grid_thw"] = p_enc["image_grid_thw"]
            results.append(item)

        if not results:
            return None

        max_seq = max(x["input_ids"].shape[0] for x in results)
        pad_id = tokenizer.pad_token_id or 0

        def pad1d(t, val):
            return F.pad(t, (0, max_seq - t.shape[0]), value=val)

        out = {
            "input_ids":      torch.stack([pad1d(x["input_ids"], pad_id) for x in results]),
            "attention_mask": torch.stack([pad1d(x["attention_mask"], 0) for x in results]),
            "pixel_values":   torch.cat([x["pixel_values"] for x in results], dim=0),
            "space_positions": [x["space_pos"] for x in results],
            "ce_labels":      [x["ce_label"] for x in results],
            "golds":          [x["gold"] for x in results],
        }
        if "image_grid_thw" in results[0]:
            out["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in results], dim=0)
        return out

    return collate


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation=args.attn_implementation,
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    hidden_size = model.lm_head.in_features
    if args.init_head:
        print(f"Loading ExtendedScoreHead from: {args.init_head}")
        score_head = ExtendedScoreHead.load(args.init_head, device=device).to(device).to(torch.bfloat16)
    else:
        score_head = ExtendedScoreHead(hidden_size).to(device).to(torch.bfloat16)
    n_params = sum(p.numel() for p in score_head.parameters())
    print(f"ExtendedScoreHead: hidden_size={hidden_size}, classes={ExtendedScoreHead.NUM_CLASSES}, params={n_params:,}")
    print(f"Loss: limited MAE (mae_weight={args.mae_weight})")
    print(f"      match condition: |argmax_score - gold| < 0.05")

    wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))

    dataset = ScoreHeadDataset(
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

    # CE label distribution (virtual digit -1..+11, label 0..12)
    ce_label_counts = Counter(score_to_ce_label(s["score"]) for s in dataset.samples)
    max_count = max(ce_label_counts.values(), default=1)
    print(f"\nCE label distribution (n={len(dataset.samples)}):")
    for label in range(13):
        vd = label - 1
        count = ce_label_counts.get(label, 0)
        bar = "█" * (count * 40 // max_count)
        print(f"  label={label:2d} (vd={vd:+3d}, score={vd/10:+.1f})  {count:6d}  {bar}")
    print()

    if args.bin_size > 0:
        bin_counts = Counter(s["score"] for s in dataset.samples)
        weights = torch.DoubleTensor([1.0 / bin_counts[s["score"]] for s in dataset.samples])
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=make_collate(processor, args.max_len),
        pin_memory=False,
    )

    optimizer = torch.optim.AdamW(
        score_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    virtual_digits = None  # initialized on first use

    for epoch in range(args.epochs):
        score_head.train()
        total_loss, total_mae = 0.0, 0.0
        total_matched, total_samples, n_batches = 0, 0, 0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            if batch is None:
                continue

            space_positions = batch.pop("space_positions")
            batch.pop("ce_labels")
            golds_list      = batch.pop("golds")
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model(**batch, output_hidden_states=True, return_dict=True)
            last_hidden = outputs.hidden_states[-1]

            head_inputs, gold_values = [], []
            for b_idx, (space_pos, gold) in enumerate(zip(space_positions, golds_list)):
                head_inputs.append(last_hidden[b_idx, space_pos, :])
                gold_values.append(gold)

            if not head_inputs:
                continue

            h_batch     = torch.stack(head_inputs)
            logits      = score_head(h_batch).float()           # (N, 13)
            gold_tensor = torch.tensor(gold_values, dtype=torch.float32, device=device)

            if virtual_digits is None:
                virtual_digits = torch.arange(-1, 12, dtype=torch.float32, device=device)

            probs = F.softmax(logits, dim=-1)                   # (N, 13)
            expected_score = torch.clamp(
                0.1 * (probs * virtual_digits).sum(dim=-1), 0.0, 1.0
            )                                                   # (N,)

            argmax_score = (logits.argmax(dim=-1).float() - 1) / 10.0  # (N,)
            match   = (argmax_score - gold_tensor).abs() < 0.05 # (N,) bool
            n_match = match.sum()

            total_matched += n_match.item()
            total_samples += len(gold_values)

            if n_match == 0:
                continue

            mae_per  = (expected_score - gold_tensor).abs()     # (N,)
            mae_loss = (mae_per * match.float()).sum() / n_match
            loss     = args.mae_weight * mae_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(score_head.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            total_mae  += mae_loss.item()
            n_batches  += 1
            wandb.log({
                "train/loss":         loss.item(),
                "train/mae_loss":     mae_loss.item(),
                "train/n_matched":    n_match.item(),
                "train/frac_matched": n_match.item() / len(gold_values),
                "train/step":         epoch * len(loader) + n_batches,
            })

        mean_loss    = total_loss / max(1, n_batches)
        mean_mae     = total_mae  / max(1, n_batches)
        frac_matched = total_matched / max(1, total_samples)
        print(
            f"Epoch {epoch + 1}: loss={mean_loss:.4f}  mae={mean_mae:.4f}  "
            f"matched={total_matched}/{total_samples} ({100*frac_matched:.1f}%)"
        )
        wandb.log({"train/epoch_loss": mean_loss, "epoch": epoch + 1})

        ckpt_path = out_dir / f"extended_score_head_epoch_{epoch + 1}.pt"
        score_head.save(ckpt_path)
        print(f"Saved → {ckpt_path}")

    print("Training complete.")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser(
        description="Train extended score head with limited MAE (matched samples only)"
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--polaris-json",      default="EXPERT/exp_datasets/polaris_exp_train.json")
    p.add_argument("--nebula-json",       default="EXPERT/exp_datasets/nebula_exp_train.json")
    p.add_argument("--polaris-image-dir", default="data/images/Polaris")
    p.add_argument("--nebula-hf",         default="Ka2ukiMatsuda/Nebula")
    p.add_argument("--max-samples-polaris", type=int, default=0)
    p.add_argument("--max-samples-nebula",  type=int, default=0)
    p.add_argument("--reason-jsonl-polaris", default="")
    p.add_argument("--reason-jsonl-nebula",  default="")
    p.add_argument("--out-dir",      required=True)
    p.add_argument("--init-head",    default="",
                   help="warm-start from existing ExtendedScoreHead checkpoint (.pt)")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch-size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-len",      type=int,   default=1200)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--bin-size",     type=float, default=0.1)
    p.add_argument("--mae-weight",   type=float, default=1.0,
                   help="Weight of MAE term (default: 1.0)")
    p.add_argument("--attn-implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--wandb-project", default="qwen-extended-score-head-gveval")
    p.add_argument("--wandb-run",     default=None)
    p.add_argument("--local-rank",    type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

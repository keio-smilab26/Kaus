"""
ViSpec draft model fine-tuning on pre-computed hidden-state ckpts.

Based on the original ViSpec train script; .

Usage (via run_draft_train.sh):
    accelerate launch scripts/train/train_draft.py \
        --base-model-path <target_model_dir> \
        --pretrained-spec-path JLKang/ViSpec-Qwen2.5-VL-7B-Instruct \
        --tmpdir draft_train \
        --cp-dir ../models/draft/checkpoints/<run_name>
"""

import argparse
import os
import sys

# Make kaus package importable from repo root
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_VISPEC_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _VISPEC_ROOT)

parser = argparse.ArgumentParser(description="ViSpec draft model training")
parser.add_argument("--base-model-path",      type=str, required=True)
parser.add_argument("--pretrained-spec-path", type=str,
                    default="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct")
parser.add_argument("--config-path",          type=str, default=None)
parser.add_argument("--load-path",            type=str, default=None)
parser.add_argument("--tmpdir",               type=str, required=True)
parser.add_argument("--cp-dir",               type=str, default="checkpoints/draft")
parser.add_argument("--lr",                   type=float, default=3e-6)
parser.add_argument("--bs",                   type=int,   default=1)
parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
parser.add_argument("--num-workers",          type=int,   default=4)
parser.add_argument("--max-len",              type=int,   default=4096)
parser.add_argument("--num-epochs",           type=int,   default=8)
parser.add_argument("--num-q",                type=int,   default=2)
parser.add_argument("--mtp-steps",            type=int,   default=1)
parser.add_argument("--pw",                   type=float, default=0.1)
parser.add_argument("--save-freq",            type=int,   default=1)
parser.add_argument("--begin-epoch",          type=int,   default=0)
parser.add_argument("--no-resume",            action="store_true")
parser.add_argument("--check-ckpt",           action="store_true")
parser.add_argument("--wandb-project",        type=str,   default="kaus-draft-train")
parser.add_argument("--wandb-name",           type=str,   default=None)
args = parser.parse_args()

train_config = {
    "lr":                          args.lr,
    "bs":                          args.bs,
    "gradient_accumulation_steps": args.gradient_accumulation_steps,
    "is_warmup":                   True,
    "num_epochs":                  args.num_epochs,
    "p_w":                         args.pw,
    "v_w":                         1.0,
    "head_w":                      0.1,
    "num_workers":                 args.num_workers,
    "data_noise":                  True,
    "noise":                       "uniform",
    "std":                         0.2,
    "max_len":                     args.max_len,
    "b1":                          0.9,
    "b2":                          0.95,
    "grad_clip":                   0.5,
    "save_freq":                   args.save_freq,
}

import json
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load as safetensors_load
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    get_linear_schedule_with_warmup,
)

from kaus.model.draft_network import Model
from kaus.model.config import EConfig

set_seed(0)
accelerator = Accelerator(
    gradient_accumulation_steps=train_config["gradient_accumulation_steps"]
)

if accelerator.is_main_process:
    import wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config={
            **train_config,
            "pretrained_spec_path": args.pretrained_spec_path,
            "base_model_path":      args.base_model_path,
            "mtp_steps":            args.mtp_steps,
            "num_q":                args.num_q,
        },
        dir=args.cp_dir,
    )
    wandb.define_metric("val/epoch*",   step_metric="epoch")
    wandb.define_metric("train/epoch*", step_metric="epoch")

# ---------------------------------------------------------------------------
# lm_head (frozen)
# ---------------------------------------------------------------------------
try:
    baseconfig = AutoConfig.from_pretrained(args.base_model_path)
    try:
        head = nn.Linear(baseconfig.hidden_size, baseconfig.vocab_size, bias=False)
    except AttributeError:
        head = nn.Linear(
            baseconfig.text_config.hidden_size,
            baseconfig.text_config.vocab_size,
            bias=False,
        )
    try:
        try:
            with open(os.path.join(args.base_model_path,
                                   "model.safetensors.index.json")) as f:
                index_json = json.loads(f.read())
                head_path  = index_json["weight_map"]["lm_head.weight"]
            with safe_open(os.path.join(args.base_model_path, head_path),
                           framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice("lm_head.weight")
                vocab_size, hidden_dim = tensor_slice.get_shape()
                tensor = tensor_slice[:, :hidden_dim].float()
        except Exception:
            with open(os.path.join(args.base_model_path,
                                   "pytorch_model.bin.index.json")) as f:
                index_json = json.loads(f.read())
                head_path  = index_json["weight_map"]["lm_head.weight"]
            weights = torch.load(os.path.join(args.base_model_path, head_path))
            tensor  = weights["lm_head.weight"].float()
    except Exception:
        m = AutoModelForImageTextToText.from_pretrained(
            args.base_model_path, torch_dtype="auto"
        )
        try:
            tensor = m.language_model.lm_head.weight.float()
        except AttributeError:
            tensor = m.lm_head.weight.float()
        del m
except Exception:
    tensor = torch.load(args.base_model_path)["lm_head.weight"].float()
    head   = nn.Linear(tensor.shape[1], tensor.shape[0], bias=False)

head.weight.data = tensor
head.eval()
for param in head.parameters():
    param.requires_grad = False

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class AddUniformNoise:
    def __init__(self, std=0.0):
        self.std = std

    def __call__(self, data):
        tensor = data["hidden_state_big"]
        noise  = (torch.rand_like(tensor) - 0.5) * self.std * 512 / tensor.shape[1]
        data["hidden_state_big"] = tensor + noise
        return data


def list_files(path, check=False):
    datapath = []
    n_skipped = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".ckpt"):
                fp = os.path.join(root, file)
                if check:
                    try:
                        torch.load(fp)
                        datapath.append(fp)
                    except Exception:
                        print(f"[warn] skipping corrupted ckpt: {fp}")
                        n_skipped += 1
                else:
                    datapath.append(fp)
    if n_skipped:
        print(f"[warn] {n_skipped} corrupted file(s) excluded")
    return datapath


class CustomDataset(Dataset):
    def __init__(self, datapath, transform=None):
        self.data      = datapath
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.load(self.data[index])
        new_data = {}

        hidden_state  = data["hidden_state"][: train_config["max_len"]][None, :]
        inputs_embeds = data.get("inputs_embeds")
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds[: train_config["max_len"]][None, :]
        loss_mask  = data["loss_mask"][: train_config["max_len"]][None, :]
        image_mask = data.get("image_mask")
        if image_mask is not None:
            image_mask = image_mask[: train_config["max_len"]][None, :]
            new_data["image_mask"] = image_mask[0].tolist()

        length         = hidden_state.shape[1]
        attention_mask = [1] * length
        loss_mask      = loss_mask[0].tolist()

        if inputs_embeds is not None:
            loss_mask = loss_mask[1:] + [0]
        else:
            loss_mask[-1] = 0

        if inputs_embeds is not None:
            emb_target = inputs_embeds[:, 1:]
            zero_pad   = torch.zeros_like(emb_target[:, :1])
            new_data["inputs_embeds"] = torch.cat((emb_target, zero_pad), dim=1)

        target   = hidden_state[:, 1:]
        zero_pad = torch.zeros(1, 1, target.shape[2])
        target   = torch.cat((target, zero_pad), dim=1)

        new_data["attention_mask"]   = attention_mask
        new_data["loss_mask"]        = loss_mask
        new_data["target"]           = target
        new_data["hidden_state_big"] = hidden_state

        if self.transform:
            new_data = self.transform(new_data)
        return new_data


class DataCollatorWithPadding:
    def _pad3d(self, t, N):
        B, n, S = t.shape
        return torch.cat((t, torch.zeros(B, N - n, S)), dim=1)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(features) == 1, "Only bs=1 is supported"
        max_length = max(item["hidden_state_big"].shape[1] for item in features)

        batch_inputs_embeds = features[0].get("inputs_embeds")
        if batch_inputs_embeds is not None:
            batch_inputs_embeds = torch.cat(
                [self._pad3d(item["inputs_embeds"], max_length) for item in features]
            )

        batch_hidden_states  = torch.cat(
            [self._pad3d(item["hidden_state_big"], max_length) for item in features]
        )
        batch_target = torch.cat(
            [self._pad3d(item["target"], max_length) for item in features]
        )
        batch_loss_mask = torch.tensor(
            [item["loss_mask"] + [0] * (max_length - len(item["loss_mask"]))
             for item in features],
            dtype=torch.bool,
        )
        batch_image_mask = features[0].get("image_mask")
        if batch_image_mask is not None:
            batch_image_mask = torch.tensor(
                [item["image_mask"] + [0] * (max_length - len(item["image_mask"]))
                 for item in features],
                dtype=torch.bool,
            )
        batch_attention_mask = torch.tensor(
            [item["attention_mask"] + [0] * (max_length - len(item["attention_mask"]))
             for item in features]
        )
        return {
            "inputs_embeds":  batch_inputs_embeds,
            "hidden_states":  batch_hidden_states,
            "target":         batch_target,
            "attention_mask": batch_attention_mask,
            "loss_mask":      batch_loss_mask,
            "image_mask":     batch_image_mask,
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def compute_loss(target_p_masked, predict_masked, topk=10):
    out_head  = head(predict_masked)
    predict_p = F.softmax(out_head, dim=-1)
    ploss     = torch.mean(torch.abs(predict_p - target_p_masked).sum(dim=-1))

    _, topk_idx  = torch.topk(target_p_masked, k=topk, dim=-1)
    student_topk = out_head.gather(-1, topk_idx)
    rev          = torch.flip(student_topk, dims=[-1])
    log_denom    = torch.flip(torch.logcumsumexp(rev, dim=-1), dims=[-1])
    rloss        = -torch.mean((student_topk - log_denom).sum(-1))

    return 10 * ploss + 0.1 * rloss, out_head


def top_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in topk]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
datapath = list_files(args.tmpdir, check=args.check_ckpt)
if accelerator.is_main_process:
    print(f"Total ckpt files: {len(datapath)}")

train_datapath = datapath[:int(len(datapath) * 0.95)]
test_datapath  = datapath[int(len(datapath) * 0.95):]

aug = AddUniformNoise(std=train_config["std"]) if train_config["data_noise"] else None

train_dataset = CustomDataset(train_datapath, transform=aug)
test_dataset  = CustomDataset(test_datapath)

train_loader = DataLoader(
    train_dataset, batch_size=train_config["bs"], shuffle=True,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"], pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=train_config["bs"], shuffle=False,
    collate_fn=DataCollatorWithPadding(),
    num_workers=train_config["num_workers"], pin_memory=True,
)

# ---------------------------------------------------------------------------
# Draft model
# ---------------------------------------------------------------------------
_config_source = args.config_path if args.config_path else args.pretrained_spec_path
config      = EConfig.from_pretrained(_config_source)
draft_model = Model(config, load_emb=True, path=args.base_model_path, num_q=args.num_q)
draft_model.gradient_checkpointing = False


def _load_spec_weights(source: str) -> dict:
    for filename, loader in [
        ("pytorch_model.bin",  lambda p: torch.load(p, map_location="cpu")),
        ("model.safetensors",  lambda p: safetensors_load(open(p, "rb").read())),
    ]:
        local_path = os.path.join(source, filename)
        if not os.path.exists(local_path):
            try:
                local_path = hf_hub_download(source, filename)
            except Exception:
                continue
        try:
            return loader(local_path)
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find weights in '{source}'")


def _apply_weights(model, state_dict, label):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing and accelerator.is_main_process:
        print(f"[{label}] missing: {missing}")
    if unexpected and accelerator.is_main_process:
        print(f"[{label}] unexpected: {unexpected}")


if accelerator.is_main_process:
    print(f"Loading pretrained spec: {args.pretrained_spec_path}")
_apply_weights(draft_model, _load_spec_weights(args.pretrained_spec_path), "pretrained")

if args.load_path:
    if accelerator.is_main_process:
        print(f"Overriding with: {args.load_path}")
    with open(args.load_path, "rb") as f:
        _apply_weights(draft_model, safetensors_load(f.read()), "checkpoint")

# ---------------------------------------------------------------------------
# Optimizer / scheduler / auto-resume
# ---------------------------------------------------------------------------
os.makedirs(args.cp_dir, exist_ok=True)

if not args.no_resume:
    ckpts = [c for c in os.listdir(args.cp_dir) if c.startswith("state")]
    if ckpts:
        begin_epoch  = max(int(c.split("_")[1]) + 1 for c in ckpts)
        resume_path  = os.path.join(args.cp_dir, f"state_{begin_epoch - 1}",
                                    "model.safetensors")
        if os.path.exists(resume_path):
            if accelerator.is_main_process:
                print(f"Auto-resuming from {resume_path}")
            with open(resume_path, "rb") as f:
                _apply_weights(draft_model, safetensors_load(f.read()), "resume")
            args.begin_epoch = begin_epoch

optimizer = optim.AdamW(
    draft_model.parameters(),
    lr=train_config["lr"],
    betas=(train_config["b1"], train_config["b2"]),
)
num_warmup_steps = len(train_loader) * 1
total_steps      = len(train_loader) * train_config["num_epochs"]
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)

draft_model, head, optimizer, train_loader, test_loader, scheduler = accelerator.prepare(
    draft_model, head, optimizer, train_loader, test_loader, scheduler
)

for _ in range(args.begin_epoch * len(train_loader)):
    scheduler.step()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for epoch in range(args.begin_epoch, train_config["num_epochs"] + 1):
    top_3acc    = [0, 0, 0]
    correct     = 0
    total       = 0
    epoch_loss  = 0.0
    num_batches = 0
    draft_model.train()

    for batch_idx, data in enumerate(
        tqdm(train_loader, disable=not accelerator.is_local_main_process)
    ):
        with accelerator.accumulate(draft_model):
            optimizer.zero_grad()

            predict = draft_model(
                data["hidden_states"],
                inputs_embeds=data["inputs_embeds"],
                attention_mask=data["attention_mask"],
                image_mask=data["image_mask"],
            )

            mtp_predicts = [predict]
            mtp_predict  = predict
            for _ in range(args.mtp_steps):
                mtp_predict = torch.cat(
                    (data["hidden_states"][:, :1, ...], mtp_predict[:, :-1, ...]), dim=1
                )
                mtp_predict = draft_model(
                    mtp_predict,
                    inputs_embeds=data["inputs_embeds"],
                    attention_mask=data["attention_mask"],
                    image_mask=data["image_mask"],
                )
                mtp_predicts.append(mtp_predict)
            mtp_predicts = torch.cat(mtp_predicts, dim=0)

            loss_mask = data["loss_mask"][:, :, None].expand(
                [args.mtp_steps + 1] + list(data["loss_mask"].shape) + [1]
            ).flatten(0, 1)
            mask_flat = loss_mask[..., 0]

            with torch.no_grad():
                base_mask       = data["loss_mask"][0]
                target_p_masked = F.softmax(
                    head(data["target"][0][base_mask]), dim=-1
                ).detach()
                target_p_masked = target_p_masked.repeat(args.mtp_steps + 1, 1)

            predict_masked = mtp_predicts[mask_flat]
            loss, out_head = compute_loss(target_p_masked, predict_masked, topk=10)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_value_(draft_model.parameters(),
                                             train_config["grad_clip"])
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            _, predicted = torch.max(out_head, dim=-1)
            _, tgt       = torch.max(target_p_masked, dim=-1)
            ct           = mask_flat.sum().item()
            cc           = (predicted == tgt).sum().item()
            topkacc      = top_accuracy(out_head, tgt, (1, 2, 3))
            for i in range(len(topkacc)):
                top_3acc[i] += topkacc[i]
            total   += ct
            correct += cc

        if accelerator.is_main_process and ct != 0:
            logdict = {
                "train/lr":   optimizer.optimizer.param_groups[0]["lr"],
                "train/loss": loss.item(),
                "train/acc":  cc / ct,
            }
            for i, v in enumerate(topkacc):
                logdict[f"train/top_{i+1}_acc"] = v.item() / ct
            if batch_idx % 50 == 0:
                alloc_gb  = torch.cuda.memory_allocated() / 1e9
                reserv_gb = torch.cuda.memory_reserved()  / 1e9
                logdict["cuda/allocated_gb"] = alloc_gb
                logdict["cuda/reserved_gb"]  = reserv_gb
            wandb.log(logdict, step=epoch * len(train_loader) + batch_idx)

        epoch_loss  += loss.item() if not loss.isnan() else 0.0
        num_batches += 1

    correct_t = torch.tensor(correct).to(accelerator.device)
    total_t   = torch.tensor(total).to(accelerator.device)
    correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
    correct_t, total_t = correct_t.sum().item(), total_t.sum().item()
    epoch_loss /= max(num_batches, 1)
    top_3acc    = accelerator.gather_for_metrics(top_3acc)

    if accelerator.is_main_process and total_t > 0:
        epoch_log = {
            "train/epochacc":  correct_t / total_t,
            "train/epochloss": epoch_loss,
            "epoch":           epoch,
        }
        for i, v in enumerate(top_3acc):
            epoch_log[f"train/epochtop_{i+1}_acc"] = v.sum().item() / total_t
        wandb.log(epoch_log, step=(epoch + 1) * len(train_loader))
        print(f"Epoch [{epoch+1}/{train_config['num_epochs']}]  "
              f"loss={epoch_loss:.4f}  acc={100*correct_t/total_t:.2f}%")

    # ── Validation ────────────────────────────────────────────────────────────
    top_3acc    = [0, 0, 0]
    correct     = 0
    total       = 0
    epoch_loss  = 0.0
    num_batches = 0
    draft_model.eval()

    for data in tqdm(test_loader, disable=not accelerator.is_local_main_process):
        with torch.no_grad():
            predict = draft_model(
                data["hidden_states"],
                inputs_embeds=data["inputs_embeds"],
                attention_mask=data["attention_mask"],
                image_mask=data["image_mask"],
            )
            base_mask       = data["loss_mask"][0]
            target_p_masked = F.softmax(
                head(data["target"][0][base_mask]), dim=-1
            ).detach()
            predict_masked  = predict[0][base_mask]
            loss, out_head  = compute_loss(target_p_masked, predict_masked)

            _, predicted = torch.max(out_head, dim=-1)
            _, tgt       = torch.max(target_p_masked, dim=-1)
            ct           = base_mask.sum().item()
            cc           = (predicted == tgt).sum().item()
            topkacc      = top_accuracy(out_head, tgt, (1, 2, 3))
            for i in range(len(topkacc)):
                top_3acc[i] += topkacc[i]
            total   += ct
            correct += cc
        epoch_loss  += loss.item() if not loss.isnan() else 0.0
        num_batches += 1

    epoch_loss /= max(num_batches, 1)
    correct_t = torch.tensor(correct).to(accelerator.device)
    total_t   = torch.tensor(total).to(accelerator.device)
    correct_t, total_t = accelerator.gather_for_metrics((correct_t, total_t))
    correct_t, total_t = correct_t.sum().item(), total_t.sum().item()
    top_3acc    = accelerator.gather_for_metrics(top_3acc)

    if accelerator.is_main_process and total_t > 0:
        val_log = {
            "val/epochacc":  correct_t / total_t,
            "val/epochloss": epoch_loss,
            "epoch":         epoch,
        }
        for i, v in enumerate(top_3acc):
            val_log[f"val/epochtop_{i+1}_acc"] = v.sum().item() / total_t
        wandb.log(val_log, step=(epoch + 1) * len(train_loader))
        print(f"Val   [{epoch+1}/{train_config['num_epochs']}]  "
              f"loss={epoch_loss:.4f}  acc={100*correct_t/total_t:.2f}%")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if accelerator.is_main_process:
        state_dir = os.path.join(args.cp_dir, f"state_{epoch}")
        accelerator.save_state(output_dir=state_dir)
        config.to_json_file(os.path.join(state_dir, "config.json"))

if accelerator.is_main_process:
    wandb.finish()

#!/bin/bash
# ViSpec draft model fine-tuning from pre-computed hidden-state ckpts.
# Calls train/train_draft.py

set -euo pipefail
cd "$(dirname "$0")/.." 

# ── Configuration ──────────────────────────────────────────────────────────────
# ckpt生成に使ったモデルと合わせること (ge_data_from_reason.sh の MODEL と同じ)
BASE_MODEL="../models/target/score:_prefill_rewritten/ce_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
PRETRAINED_SPEC="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"

# ge_data_from_reason.sh の出力先
TMPDIR="draft_train"
# 訓練結果の保存先
CPDIR="../models/draft/checkpoints/selfgvl-rs"

LR=3e-6
NUM_EPOCHS=8
MTP_STEPS=1
NUM_Q=2
MAX_LEN=4096
BS=1
GRAD_ACCUM=1
NUM_WORKERS=4

WANDB_PROJECT="kaus-draft-train"
WANDB_NAME="selfgvl-rs"
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$CPDIR"

echo "========================================"
echo " ViSpec Draft Model Training"
echo " Base model : $BASE_MODEL"
echo " Spec init  : $PRETRAINED_SPEC"
echo " Data dir   : $TMPDIR"
echo " Checkpoint : $CPDIR"
echo "========================================"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch \
    --num_processes 1 \
    --mixed_precision bf16 \
    train_draft.py \
        --base-model-path        "$BASE_MODEL" \
        --pretrained-spec-path   "$PRETRAINED_SPEC" \
        --tmpdir                 "$TMPDIR" \
        --cp-dir                 "$CPDIR" \
        --lr                     $LR \
        --bs                     $BS \
        --gradient-accumulation-steps $GRAD_ACCUM \
        --num-workers            $NUM_WORKERS \
        --max-len                $MAX_LEN \
        --num-epochs             $NUM_EPOCHS \
        --num-q                  $NUM_Q \
        --mtp-steps              $MTP_STEPS \
        --wandb-project          "$WANDB_PROJECT" \
        --wandb-name             "$WANDB_NAME"

echo ""
echo "========================================"
echo " Training complete"
echo " Checkpoints saved to: $CPDIR"
echo "========================================"
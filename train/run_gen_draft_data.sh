#!/bin/bash
# Generate ViSpec draft model training ckpts from pre-existing reasoning texts.
# Teacher-forcing forward pass — much faster than AR generation.
#
# nebula-exp-train (23,251 samples) → outdir/0/
# polaris-exp-train (11,777 samples) → outdir/1/
# Two datasets run in parallel on GPU 0 and GPU 1.

set -euo pipefail
cd "$(dirname "$0")/.." 

# ── Configuration ──────────────────────────────────────────────────────────────
MODEL="../models/target/score:_prefill_rewritten/ce_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
REASON_ROOT="results/nonspec/self_gvl_score:_rewritten"
OUTDIR="draft_train"
MAX_LEN=4096
# ──────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUTDIR"

echo "========================================"
echo " ge_data_from_reason"
echo " Model       : $MODEL"
echo " Reason root : $REASON_ROOT"
echo " Outdir      : $OUTDIR"
echo "========================================"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# nebula-exp-train → outdir/0/
echo "[1/2] nebula-exp-train ..."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/gen_draft_data.py \
    --model        "$MODEL" \
    --reason-jsonl "$REASON_ROOT/nebula-exp-train/eval_results.jsonl" \
    --dataset      nebula-exp-train \
    --outdir       "$OUTDIR" \
    --gpu-index    0 \
    --index        0 \
    --max-len      $MAX_LEN

# polaris-exp-train → outdir/1/
echo "[2/2] polaris-exp-train ..."
CUDA_VISIBLE_DEVICES=0 .venv/bin/python train/gen_draft_data.py \
    --model        "$MODEL" \
    --reason-jsonl "$REASON_ROOT/polaris-exp-train/eval_results.jsonl" \
    --dataset      polaris-exp-train \
    --outdir       "$OUTDIR" \
    --gpu-index    0 \
    --index        1 \
    --max-len      $MAX_LEN
echo ""
echo "========================================"
echo " All done. Ckpts saved to $OUTDIR"
echo "========================================"

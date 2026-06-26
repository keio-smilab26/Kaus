#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# LoRA fine-tuning of Qwen2.5-VL with self-generated reasoning + frozen ExtendedScoreHead
# RS format, G-VEval prompts
#
# Key difference from run_qwen_gveval_prompt_exsh_train.sh:
#   Qwen generates its own Turn 1 reasoning at each step (AR, no grad).
#   Score head loss uses the self-generated reasoning → aligns with test time.
#   Reasoning regularization loss (teacher-forcing on pre-collected reason)
#   prevents reasoning drift.  batch_size is fixed to 1 (AR generation per step).
#
# 使用例:
#   bash run_target_train.sh
#   bash run_target_train.sh --reason-weight 0.5
#   bash run_target_train.sh --kl-weight 0.0
#   bash run_target_train.sh --epochs 3
#   bash run_target_train.sh --gpu 1
#   bash run_target_train.sh --no-merge
#   bash run_target_train.sh --debug
# ==============================================================================

# ── デフォルト値 ───────────────────────────────────────────────────────────────
MODEL_ALIAS="vanilla"
EPOCHS=1
GPU=0
LR=2e-5
MAX_LEN=1200
MAX_REASON_LEN=1200
MAX_NEW_TOKENS_REASON=256
BIN_SIZE=0.1
SIGMA2=1.0
KL_WEIGHT=1.0
MSE_WEIGHT=10.0
REASON_WEIGHT=1.0
LORA_R=128
LORA_ALPHA=256
DO_MERGE=1
DEBUG=0

MAX_SAMPLES_POLARIS=0
MAX_SAMPLES_NEBULA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)               EPOCHS="$2";               shift 2 ;;
        --gpu)                  GPU="$2";                  shift 2 ;;
        --lr)                   LR="$2";                   shift 2 ;;
        --max-len)              MAX_LEN="$2";              shift 2 ;;
        --max-reason-len)       MAX_REASON_LEN="$2";       shift 2 ;;
        --max-new-tokens-reason) MAX_NEW_TOKENS_REASON="$2"; shift 2 ;;
        --sigma2)               SIGMA2="$2";               shift 2 ;;
        --kl-weight)            KL_WEIGHT="$2";            shift 2 ;;
        --mse-weight)           MSE_WEIGHT="$2";           shift 2 ;;
        --reason-weight)        REASON_WEIGHT="$2";        shift 2 ;;
        --lora-r)               LORA_R="$2";               shift 2 ;;
        --lora-alpha)           LORA_ALPHA="$2";           shift 2 ;;
        --no-merge)             DO_MERGE=0;                shift ;;
        --debug)                DEBUG=1;                   shift ;;
        *)                      shift ;;
    esac
done

# ── パス設定 ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="../datasets"

POLARIS_JSON="${DATASET_ROOT}/polaris-exp/polaris_exp_train.json"
NEBULA_JSON="${DATASET_ROOT}/nebula-exp/nebula_exp_train.json"
POLARIS_IMAGE_DIR="${DATASET_ROOT}/polaris/polaris/images"

REASON_JSONL_POLARIS="results/nonspec/vanilla/nebula-exp-train/eval_results.jsonl"
REASON_JSONL_NEBULA="results/nonspec/vanilla/nebula-exp-train/eval_results.jsonl"

# ── スコアヘッド（固定）────────────────────────────────────────────────────────
SCORE_HEAD_PATH="../models/head/exsh/exsh_pretrain_epoch_3.pt"

# ── ベースモデル ───────────────────────────────────────────────────────────────
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"

# ── 出力先 ─────────────────────────────────────────────────────────────────────
CKPT_ROOT="${REPO_ROOT}/../models/target"
OUT_DIR="${CKPT_ROOT}/lora/${MODEL_ALIAS}/score:_prefill_rewritten/ce_kl${KL_WEIGHT}_mse${MSE_WEIGHT}_rsn${REASON_WEIGHT}"
MERGE_OUT_DIR="${CKPT_ROOT}/merged/${MODEL_ALIAS}/score:_prefill_rewritten/ce_kl${KL_WEIGHT}_mse${MSE_WEIGHT}_rsn${REASON_WEIGHT}/qwen_vl_7b"

# ── デバッグ時はサンプル数を縮小 ─────────────────────────────────────────────
if [[ $DEBUG -eq 1 ]]; then
    EPOCHS=1
    MAX_SAMPLES_POLARIS=10
    MAX_SAMPLES_NEBULA=10
    echo "[debug] debug mode: epochs=1, samples=10+10"
fi

# ── 実験条件ログ ───────────────────────────────────────────────────────────────
echo "============================================================"
echo " LoRA Training — Qwen + Self-Generated Reasoning + frozen ExtendedScoreHead"
echo "============================================================"
echo " Model              : ${MODEL_PATH}"
echo " Score head         : ${SCORE_HEAD_PATH}"
echo " Polaris JSON       : ${POLARIS_JSON} (max: ${MAX_SAMPLES_POLARIS:-all})"
echo " Nebula JSON        : ${NEBULA_JSON} (max: ${MAX_SAMPLES_NEBULA:-all})"
echo " JSONL Polaris      : ${REASON_JSONL_POLARIS}"
echo " JSONL Nebula       : ${REASON_JSONL_NEBULA}"
echo " KL weight          : ${KL_WEIGHT}  (sigma2=${SIGMA2})"
echo " MSE weight         : ${MSE_WEIGHT}"
echo " Reason weight      : ${REASON_WEIGHT}"
echo " Max reason tokens  : ${MAX_NEW_TOKENS_REASON}"
echo " Max len (Turn 2)   : ${MAX_LEN}"
echo " Max len (Turn 1 TF): ${MAX_REASON_LEN}"
echo " LoRA r             : ${LORA_R}"
echo " LoRA alpha         : ${LORA_ALPHA}"
echo " Epochs             : ${EPOCHS}"
echo " LR                 : ${LR}"
echo " GPU                : ${GPU}"
echo " LoRA out           : ${OUT_DIR}"
echo " Merged out         : ${MERGE_OUT_DIR}"
echo " Do merge           : ${DO_MERGE}"
echo "============================================================"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"
mkdir -p "${UV_CACHE_DIR}"

cd "${REPO_ROOT}"

# ── LoRAトレーニング ───────────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES="${GPU}" \
    .venv/bin/python train_target.py \
        --model                    "${MODEL_PATH}" \
        --score-head-path          "${SCORE_HEAD_PATH}" \
        --polaris-json             "${POLARIS_JSON}" \
        --nebula-json              "${NEBULA_JSON}" \
        --polaris-image-dir        "${POLARIS_IMAGE_DIR}" \
        --reason-jsonl-polaris     "${REASON_JSONL_POLARIS}" \
        --reason-jsonl-nebula      "${REASON_JSONL_NEBULA}" \
        --out-dir                  "${OUT_DIR}" \
        --epochs                   "${EPOCHS}" \
        --lr                       "${LR}" \
        --max-len                  "${MAX_LEN}" \
        --max-reason-len           "${MAX_REASON_LEN}" \
        --max-new-tokens-reason    "${MAX_NEW_TOKENS_REASON}" \
        --max-samples-polaris      "${MAX_SAMPLES_POLARIS}" \
        --max-samples-nebula       "${MAX_SAMPLES_NEBULA}" \
        --kl-weight                "${KL_WEIGHT}" \
        --mse-weight               "${MSE_WEIGHT}" \
        --reason-weight            "${REASON_WEIGHT}" \
        --sigma2                   "${SIGMA2}" \
        --lora-r                   "${LORA_R}" \
        --lora-alpha               "${LORA_ALPHA}" \
        --bin-size                 "${BIN_SIZE}" \
        --num-workers              0 \
        --attn-implementation      sdpa \
        --wandb-project            "qwen-lora-self-exsh" \
        --wandb-run                "self-exsh-ce_kl${KL_WEIGHT}_mse${MSE_WEIGHT}_rsn${REASON_WEIGHT}-${MODEL_ALIAS}-ep${EPOCHS}" \
    2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "============================================================"
echo " Training done — adapters saved to ${OUT_DIR}"
echo "============================================================"

# ── マージ ─────────────────────────────────────────────────────────────────────
if [[ $DO_MERGE -eq 1 ]]; then
    LAST_EPOCH_ADAPTER="${OUT_DIR}/epoch_${EPOCHS}"
    echo ""
    echo "============================================================"
    echo " Merging LoRA adapter (epoch ${EPOCHS}) into base model..."
    echo " Adapter : ${LAST_EPOCH_ADAPTER}"
    echo " Output  : ${MERGE_OUT_DIR}"
    echo "============================================================"

    mkdir -p "${MERGE_OUT_DIR}"

    CUDA_VISIBLE_DEVICES="${GPU}" \
        .venv/bin/python merge.py \
            --base-model   "${MODEL_PATH}" \
            --adapter-path "${LAST_EPOCH_ADAPTER}" \
            --output-dir   "${MERGE_OUT_DIR}" \
        2>&1 | tee "${MERGE_OUT_DIR}/merge.log"

    echo ""
    echo "============================================================"
    echo " Done — merged model saved to ${MERGE_OUT_DIR}"
    echo "============================================================"
fi

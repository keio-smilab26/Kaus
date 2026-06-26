#!/usr/bin/env bash
set -euo pipefail
# ==============================================================================
# Extended Score-Head Training — virtual digits -1..+11 (13 classes)
# frozen Qwen2.5-VL (RS format, G-VEval prompts)
#
# 使用例:
#   bash run_exsh_pretrain.sh                             # CE only
#   bash run_exsh_pretrain.sh --kl-weight 1.0             # CE + KLdiv
#   bash run_exsh_pretrain.sh --mse-weight 1.0            # CE + MSE
#   bash run_exsh_pretrain.sh --kl-weight 1.0 --mse-weight 1.0  # CE + KL + MSE
#   bash run_exsh_pretrain.sh --model sr        # SR merged model
#   bash run_exsh_pretrain.sh --model rs        # RS merged model
#   bash run_exsh_pretrain.sh --epochs 3
#   bash run_exsh_pretrain.sh --gpu 1
#   bash run_exsh_pretrain.sh --debug           # 小バッチで動作確認
#   bash run_exsh_pretrain.sh --jsonl nonspec-vanilla-rs-gveval-prompt  (デフォルト)
# ==============================================================================

# ── デフォルト値 ───────────────────────────────────────────────────────────────
MODEL_ALIAS="vanilla"
EPOCHS=3
GPU=0
BATCH_SIZE=4
LR=1e-3
MAX_LEN=1200
BIN_SIZE=0.1
DEBUG=0
JSONL_ALIAS="nonspec_vanilla"
SIGMA2=1.0
CE_WEIGHT=1.0
KL_WEIGHT=1.0
MSE_WEIGHT=10
INIT_HEAD=""

# ── サンプル数制限（0 = 全件使用）─────────────────────────────────────────────
MAX_SAMPLES_POLARIS=0
MAX_SAMPLES_NEBULA=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)      MODEL_ALIAS="$2";  shift 2 ;;
        --epochs)     EPOCHS="$2";       shift 2 ;;
        --gpu)        GPU="$2";          shift 2 ;;
        --batch-size) BATCH_SIZE="$2";   shift 2 ;;
        --lr)         LR="$2";           shift 2 ;;
        --max-len)    MAX_LEN="$2";      shift 2 ;;
        --jsonl)      JSONL_ALIAS="$2";  shift 2 ;;
        --sigma2)     SIGMA2="$2";       shift 2 ;;
        --ce-weight)  CE_WEIGHT="$2";   shift 2 ;;
        --kl-weight)  KL_WEIGHT="$2";   shift 2 ;;
        --mse-weight) MSE_WEIGHT="$2";  shift 2 ;;
        --init-head)  INIT_HEAD="$2";   shift 2 ;;
        --debug)      DEBUG=1;           shift ;;
        *)            shift ;;
    esac
done

# ── パス設定 ───────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="../datasets"
MODEL_ROOT="../models"

POLARIS_JSON="${DATASET_ROOT}/polaris-exp/polaris_exp_train.json"
NEBULA_JSON="${DATASET_ROOT}/nebula-exp/nebula_exp_train.json"
POLARIS_IMAGE_DIR="${DATASET_ROOT}/polaris/polaris/images"

case "$JSONL_ALIAS" in
    "default-exp")
        REASON_JSONL_POLARIS=""
        REASON_JSONL_NEBULA=""
        ;;
    "nonspec-vanilla")
        REASON_JSONL_POLARIS="results/nonspec/vanilla/polaris-exp-train/eval_results.jsonl"
        REASON_JSONL_NEBULA="results/nonspec/vanilla/nebula-exp-train/eval_results.jsonl"
        ;;
    *)
        echo "Error: Unknown --jsonl alias '${JSONL_ALIAS}'" >&2
        echo "Choices: default-exp | nonspec-vanilla-rs-gveval-prompt" >&2
        exit 1
        ;;
esac

# ── モデルパスの解決 ───────────────────────────────────────────────────────────
case "$MODEL_ALIAS" in
    "vanilla")
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
        ;;
    "rs")
        MODEL_PATH="${MODEL_ROOT}/rs_checkpoints/qwen_vl_7b"
        ;;
    "sr")
        MODEL_PATH="${MODEL_ROOT}/sr_checkpoints/qwen_vl_7b"
        ;;
    "selfgvlexsh")
        MODEL_PATH="/home/initial/Documents/smilab/lab26/03B4progress26/models/qwen/target_model/self_exsh_checkpoints/merged/vanilla/ce_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
        ;;
    "self_token_pred")
        MODEL_PATH="/home/initial/Documents/smilab/lab26/03B4progress26/models/qwen/target_model/token_pred_checkpoints/merged/vanilla/score1.0_rsn0.1/qwen_vl_7b"
        ;;
    *)
        MODEL_PATH="$MODEL_ALIAS"
        ;;
esac

# ── 出力先 ─────────────────────────────────────────────────────────────────────
CKPT_ROOT="${REPO_ROOT}/../models/head"
OUT_DIR="${CKPT_ROOT}/${MODEL_ALIAS}/${JSONL_ALIAS}/nonscore_prefill/ce${CE_WEIGHT}_kl${KL_WEIGHT}_mse${MSE_WEIGHT}"

# ── デバッグ時はエポック・バッチ・サンプル数を縮小 ───────────────────────────
if [[ $DEBUG -eq 1 ]]; then
    EPOCHS=1
    BATCH_SIZE=1
    MAX_SAMPLES_POLARIS=10
    MAX_SAMPLES_NEBULA=10
    echo "[debug] debug mode: epochs=1, batch=1, samples=10+10"
fi

# ── 実験条件ログ ───────────────────────────────────────────────────────────────
echo "============================================================"
echo " Extended Score-Head Training (virtual digits -1..+11)"
echo "============================================================"
echo " Model alias  : ${MODEL_ALIAS}"
echo " Model path   : ${MODEL_PATH}"
echo " Polaris JSON : ${POLARIS_JSON} (max: ${MAX_SAMPLES_POLARIS:-all})"
echo " Nebula JSON  : ${NEBULA_JSON} (max: ${MAX_SAMPLES_NEBULA:-all})"
echo " JSONL alias  : ${JSONL_ALIAS}"
echo " JSONL Polaris: ${REASON_JSONL_POLARIS:-(none, use EXPERT explanation)}"
echo " JSONL Nebula : ${REASON_JSONL_NEBULA:-(none, use EXPERT explanation)}"
echo " Image dir    : ${POLARIS_IMAGE_DIR}"
echo " CE weight    : ${CE_WEIGHT}"
echo " KL weight    : ${KL_WEIGHT}  (sigma2=${SIGMA2})"
echo " MSE weight   : ${MSE_WEIGHT}"
echo " Epochs       : ${EPOCHS}"
echo " Batch size   : ${BATCH_SIZE}"
echo " LR           : ${LR}"
echo " GPU          : ${GPU}"
echo " Output       : ${OUT_DIR}"
echo " Init head    : ${INIT_HEAD:-(random init)}"
echo "============================================================"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uvcache}"
mkdir -p "${UV_CACHE_DIR}"

cd "${REPO_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
    .venv/bin/python train_exsh_pretrain.py \
        --model                "${MODEL_PATH}" \
        --polaris-json         "${POLARIS_JSON}" \
        --nebula-json          "${NEBULA_JSON}" \
        --polaris-image-dir    "${POLARIS_IMAGE_DIR}" \
        --out-dir              "${OUT_DIR}" \
        --epochs               "${EPOCHS}" \
        --batch-size           "${BATCH_SIZE}" \
        --lr                   "${LR}" \
        --max-len              "${MAX_LEN}" \
        --bin-size             "${BIN_SIZE}" \
        --max-samples-polaris  "${MAX_SAMPLES_POLARIS}" \
        --max-samples-nebula   "${MAX_SAMPLES_NEBULA}" \
        ${REASON_JSONL_POLARIS:+--reason-jsonl-polaris "${REASON_JSONL_POLARIS}"} \
        ${REASON_JSONL_NEBULA:+--reason-jsonl-nebula   "${REASON_JSONL_NEBULA}"} \
        --ce-weight            "${CE_WEIGHT}" \
        --kl-weight            "${KL_WEIGHT}" \
        --mse-weight           "${MSE_WEIGHT}" \
        --sigma2               "${SIGMA2}" \
        ${INIT_HEAD:+--init-head "${INIT_HEAD}"} \
        --num-workers          0 \
        --attn-implementation  sdpa \
        --wandb-project        "qwen-extended-score-head-gveval" \
        --wandb-run            "nonscore-prefill-exsh-gveval-ce_kl${KL_WEIGHT}_mse${MSE_WEIGHT}-${MODEL_ALIAS}-${JSONL_ALIAS}-ep${EPOCHS}" \
    2>&1 | tee "${OUT_DIR}/train.log"

echo ""
echo "============================================================"
echo " Done — checkpoints saved to ${OUT_DIR}"
echo "============================================================"

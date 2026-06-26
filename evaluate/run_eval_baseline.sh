#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Non-Speculative Qwen Evaluation — 統合スクリプト
#
# ※ SRS系は run_nonspec_srs_eval.sh を使用してください
#
# 使用例:
#   bash scripts/run_nonspec_eval.sh vanilla sr                                        nebula
#   bash scripts/run_nonspec_eval.sh sr      sr                                        flickr8k-ex
#   bash scripts/run_nonspec_eval.sh rs      rs                                        nebula
#   bash scripts/run_nonspec_eval.sh vanilla vanilla_rs_gveval_prompt_score_head       nebula
#   bash scripts/run_nonspec_eval.sh vanilla vanilla_rs_gveval_prompt_score_head_kldiv nebula
#
# 引数:
#   $1  MODEL_ALIAS   : vanilla | sr | rs
#   $2  PY_ALIAS      : vanilla_sr | vanilla_sr_fleur_prompt | sr | rs
#                       vanilla_rs | vanilla_rs_ef | vanilla_rs_gveval_prompt
#                       vanilla_rs_gveval_prompt_score_head
#                       vanilla_rs_gveval_prompt_score_head_kldiv
#                       vanilla_rs_gveval_prompt_extended_score_head_kldiv
#   $3  DATASET_ALIAS : nebula | composite | flickr8k-ex | flickr8k-cf
#                       polaris-exp-train | nebula-exp-train
# ==============================================================================

MODEL_ALIAS=${1:-"vanilla"}
PY_ALIAS=${2:-"sr"}
DATASET_ALIAS=${3:-"nebula"}
CACHED_REASON_JSONL=${4:-""}

# ------------------------------------------------------------------------------
# 1. モデルパスの設定
# ------------------------------------------------------------------------------
case "$MODEL_ALIAS" in
    "vanilla")
        MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
        ;;
    "self_gvl_score:_rewritten")
        MODEL_PATH="../models/target/score:_prefill_rewritten/ce_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
        ;;
    *)
        echo "Error: Unknown model alias '$MODEL_ALIAS'" >&2
        echo "Choices: vanilla | gvlexsh | sr | rs" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# 2. 評価モジュールの設定
# ------------------------------------------------------------------------------
SCORE_HEAD_ROOT="/home/initial/Documents/smilab/lab26/03B4progress26/models/qwen/score_head"
SUPPORT_REASON_CACHE=0

case "$PY_ALIAS" in
    "nonspec_score:_exsh")
        EVAL_MOD="evaluate.baseline"
        HEAD_DIR="../models/head/exsh/exsh_limited_mae_epoch_1.pt"
        HEAD_ARG="--score-head-path"
        SUPPORT_REASON_CACHE=1
        ;;
    *)
        echo "Error: Unknown py alias '$PY_ALIAS'" >&2
        echo "Choices: vanilla_sr | vanilla_sr_fleur_prompt | sr | rs | vanilla_rs | vanilla_rs_ef" >&2
        echo "         vanilla_rs_gveval_prompt" >&2
        echo "         vanilla_rs_gveval_prompt_score_head" >&2
        echo "         vanilla_rs_gveval_prompt_score_head_kldiv" >&2
        echo "         vanilla_rs_gveval_prompt_extended_score_head_kldiv" >&2
        echo "  ※ SRS系: bash scripts/run_nonspec_srs_eval.sh を使用してください" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# 3. データセット検証
# ------------------------------------------------------------------------------
case "$DATASET_ALIAS" in
    "nebula"|"composite"|"flickr8k-ex"|"flickr8k-cf"|"polaris-exp-train"|"nebula-exp-train") ;;
    *)
        echo "Error: Unknown dataset alias '$DATASET_ALIAS'" >&2
        echo "Choices: nebula | composite | flickr8k-ex | flickr8k-cf | polaris-exp-train | nebula-exp-train" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# 4. 結果保存先
# ------------------------------------------------------------------------------
RESULT_DIR="results/nonspec/${MODEL_ALIAS}/${PY_ALIAS}/${DATASET_ALIAS}"
mkdir -p "${RESULT_DIR}"

# ------------------------------------------------------------------------------
# 実験条件ログ
# ------------------------------------------------------------------------------
echo "========================================"
echo " Non-Speculative Qwen Evaluation"
echo "========================================"
echo " [Model]"
echo "   Alias  : $MODEL_ALIAS"
echo "   Path   : $MODEL_PATH"
echo " [Eval]"
echo "   Alias  : $PY_ALIAS"
echo "   Module : $EVAL_MOD"
if [[ -n "${HEAD_DIR}" ]]; then
echo "   HeadPt : $HEAD_DIR"
fi
echo " [Data]"
echo "   Dataset: $DATASET_ALIAS"
echo " [Output]"
echo "   Dir    : $RESULT_DIR"
if [[ -n "${CACHED_REASON_JSONL}" ]]; then
echo " [Cache]"
echo "   JSONL  : $CACHED_REASON_JSONL"
fi
echo "========================================"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ------------------------------------------------------------------------------
# 5. 実行
# ------------------------------------------------------------------------------
HF_ATTENTION_IMPLEMENTATION=sdpa CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m "${EVAL_MOD}" \
  --model-path "${MODEL_PATH}" \
  --datasets "${DATASET_ALIAS}" \
  --output-prefix "${RESULT_DIR}/eval_results" \
  --max-new-tokens-score 1 \
  --max-new-tokens-reason 256 \
  --temperature 0.0 \
  --verbose \
  ${HEAD_DIR:+${HEAD_ARG} "${HEAD_DIR}"} \
  $([[ $SUPPORT_REASON_CACHE -eq 1 && -n "${CACHED_REASON_JSONL}" ]] && echo "--cached-reason-jsonl ${CACHED_REASON_JSONL}")

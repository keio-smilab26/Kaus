#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# ViSpec Expert Evaluation — 統合スクリプト
#
# 使用例:
#   bash evaluate/run_eval_kaus.sh vanilla-vanilla sr      nebula
#   bash evaluate/run_eval_kaus.sh sr-sr           rs-discode-num flickr8k-ex
#
# 引数:
#   $1  MODEL_ALIAS   : vanilla-vanilla | vanilla-sr | sr-sr | rs-rs
#   $2  PY_ALIAS      : sr | rs | rs-discode | rs-discode-num | srs
#                       vanilla-rs-gveval-prompt-extended-score-head-kldiv
#   $3  DATASET_ALIAS : nebula | composite | flickr8k-ex | flickr8k-cf
# ==============================================================================

MODEL_ALIAS=${1:-"vanilla-vanilla"}
PY_ALIAS=${2:-"sr"}
DATASET_ALIAS=${3:-"nebula"}

# ------------------------------------------------------------------------------
# 1. モデルの組み合わせ設定 (TARGET / DRAFT のパスのみ)
# ------------------------------------------------------------------------------
case "$MODEL_ALIAS" in
    "vanilla-vanilla")
        BASE_DIR="Qwen/Qwen2.5-VL-7B-Instruct"
        SPEC_DIR="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"
        ;;
    "selfgvlexsh_score:_rewritten-vanilla")
        BASE_DIR="../models/target/score:_prefill_rewritten/ce_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
        SPEC_DIR="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"
        ;;
    "10class")
        BASE_DIR="../models/target/merged/vanilla/default_lm_head/ce1.0_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
        SPEC_DIR="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"
        ;;
    "mse0")
        BASE_DIR="../models/target/merged/vanilla/score:_prefill_rewritten/ce_kl1.0_mse0.0_rsn1.0/qwen_vl_7b"
        SPEC_DIR="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"
        ;;
    "joint")
        BASE_DIR="../models/target/merged/vanilla/joint_exsh/ce1.0_kl1.0_mse10.0_rsn1.0/qwen_vl_7b"
        SPEC_DIR="JLKang/ViSpec-Qwen2.5-VL-7B-Instruct"
        ;;
    *)
        echo "Error: Unknown model alias '$MODEL_ALIAS'" >&2
        echo "Choices: vanilla-vanilla | vanilla-sr | sr-sr | rs-rs" >&2
        exit 1
        ;;
esac

# ------------------------------------------------------------------------------
# 2. 評価モジュールの設定
# ------------------------------------------------------------------------------
case "$PY_ALIAS" in
    "kaus")
        EVAL_MOD="evaluate.kaus"
        HEAD_DIR="../models/head/exsh/exsh_limited_mae_epoch_1.pt"
        HEAD_ARG="--score-head-path"
        ;;
    "kaus_mae")
        EVAL_MOD="evaluate.kaus"
        HEAD_DIR="../models/head/selfgvlexsh_score:_rewritten/nonspec-selfgvlexsh-score:-rewritten/nonscore_prefill_limited_mae/mae1.0_lr1e-4/extended_score_head_epoch_1.pt"
        HEAD_ARG="--score-head-path"
        ;;
    "10class_rsn")
        EVAL_MOD="evaluate.kaus"
        HEAD_DIR=""
        HEAD_ARG="--score-head-path"
        ;;
    "sdpt")
        EVAL_MOD="evaluate.kaus"
        HEAD_DIR="../models/head/exsh/exsh_pretrain_epoch_3.pt"
        HEAD_ARG="--score-head-path"
        ;;
    "mse0_mae")
        EVAL_MOD="evaluate.kaus"
        HEAD_DIR="../models/head/mse0/mse0_rsn/nonscore_prefill_limited_mae/mae1.0_lr1e-4/extended_score_head_epoch_1.pt"
        HEAD_ARG="--score-head-path"
        ;;
    "joint_mae")
        EVAL_MOD="evaluate.kaus"
        HEAD_DIR="../models/head/joint/joint_rsn/nonscore_prefill_limited_mae/mae1.0_lr1e-4/extended_score_head_epoch_1.pt"
        HEAD_ARG="--score-head-path"
        ;;
    *)
        echo "Error: Unknown py alias '$PY_ALIAS'" >&2
        echo "Choices: sr | rs | rs-discode | rs-discode-num | rs-score-head" >&2
        echo "         vanilla-rs-gveval-extended-score-head-kldiv" >&2
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
RESULT_DIR="results/kaus/${MODEL_ALIAS}/${PY_ALIAS}/${DATASET_ALIAS}"
mkdir -p "${RESULT_DIR}"

# ------------------------------------------------------------------------------
# 実験条件ログ (誤り防止のため実行前に全条件を表示)
# ------------------------------------------------------------------------------
echo "========================================"
echo " ViSpec Expert Evaluation"
echo "========================================"
echo " [Model]"
echo "   Alias   : $MODEL_ALIAS"
echo "   Target  : $BASE_DIR"
echo "   Draft   : $SPEC_DIR"
echo " [Eval]"
echo "   Alias   : $PY_ALIAS"
echo "   Module  : $EVAL_MOD"
if [[ -n "${HEAD_DIR}" ]]; then
echo "   HeadPt  : $HEAD_DIR"
fi
echo " [Data]"
echo "   Dataset : $DATASET_ALIAS"
echo " [Output]"
echo "   Dir     : $RESULT_DIR"
echo "========================================"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

# ------------------------------------------------------------------------------
# 5. 実行
# ------------------------------------------------------------------------------
HF_ATTENTION_IMPLEMENTATION=sdpa CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m "${EVAL_MOD}" \
  --base-model-path "${BASE_DIR}" \
  --spec-model-path "${SPEC_DIR}" \
  --datasets "${DATASET_ALIAS}" \
  --output-prefix "${RESULT_DIR}/eval_results" \
  --max-new-tokens-score 1 \
  --max-new-tokens-reason 256 \
  --total-token 30 \
  --depth 3 \
  --top-k 8 \
  --num-q 2 \
  --temperature 0.0 \
  --verbose \
  ${HEAD_DIR:+${HEAD_ARG} "${HEAD_DIR}"}

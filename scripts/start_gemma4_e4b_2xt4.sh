#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_ID="${MODEL_ID:-google/gemma-4-E4B-it}"
AVAMERG_ROOT="${AVAMERG_ROOT:-data/raw/avamerg}"
ESCONV_JSON="${ESCONV_JSON:-data/raw/esconv/ESConv.json}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

common_args=(
  --model_name_or_path "$MODEL_ID"
  --avamerg_root "$AVAMERG_ROOT"
  --avamerg_split train
  --avamerg_text_only
  --esconv_json "$ESCONV_JSON"
  --load_in_4bit
  --use_lora
  --gradient_checkpointing
  --max_length 256
  --max_response_tokens 64
  --lora_r 4
  --lora_alpha 8
  --lora_dropout 0.05
)

case "$MODE" in
  dump)
    python scripts/train_sft.py \
      "${common_args[@]}" \
      --output_dir outputs/sft/debug_gemma4_e4b_2xt4 \
      --dump_example_prompts
    ;;
  smoke)
    python scripts/train_sft.py \
      "${common_args[@]}" \
      --output_dir outputs/sft/gemma4_e4b_2xt4_smoke \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-5 \
      --num_train_epochs 1 \
      --max_train_samples 8 \
      --max_steps 1 \
      --logging_steps 1
    ;;
  train)
    python scripts/train_sft.py \
      "${common_args[@]}" \
      --output_dir outputs/sft/task1_gemma4_e4b_2xt4 \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 2e-5 \
      --num_train_epochs 1 \
      --logging_steps 10 \
      --save_steps 100
    ;;
  *)
    echo "Usage: $0 {dump|smoke|train}"
    exit 1
    ;;
esac

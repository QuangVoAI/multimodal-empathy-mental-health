#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL_ID="${MODEL_ID:-google/gemma-4-12B-it}"
AVAMERG_ROOT="${AVAMERG_ROOT:-data/raw/avamerg}"
ESCONV_JSON="${ESCONV_JSON:-data/raw/esconv/ESConv.json}"

common_args=(
  --model_name_or_path "$MODEL_ID"
  --avamerg_root "$AVAMERG_ROOT"
  --avamerg_split train
  --avamerg_text_only
  --esconv_json "$ESCONV_JSON"
)

case "$MODE" in
  dump)
    python scripts/train_sft.py \
      "${common_args[@]}" \
      --output_dir outputs/sft/debug_joint \
      --dump_example_prompts
    ;;
  smoke)
    python scripts/train_sft.py \
      "${common_args[@]}" \
      --output_dir outputs/sft/joint_smoke \
      --load_in_4bit \
      --use_lora \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --num_train_epochs 1 \
      --max_steps 1 \
      --logging_steps 1
    ;;
  train)
    python scripts/train_sft.py \
      "${common_args[@]}" \
      --output_dir outputs/sft/task1_gemma12b_it_joint \
      --load_in_4bit \
      --use_lora \
      --per_device_train_batch_size 1 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-4 \
      --num_train_epochs 1 \
      --logging_steps 10 \
      --save_steps 100
    ;;
  *)
    echo "Usage: $0 {dump|smoke|train}"
    exit 1
    ;;
esac

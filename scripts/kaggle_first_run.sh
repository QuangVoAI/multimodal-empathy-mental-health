#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/kaggle/working/empathy_mh_gemma}"
HF_TOKEN_VALUE="${HF_TOKEN_VALUE:-}"

cd "${PROJECT_ROOT}"

python -m pip install --upgrade pip
pip uninstall -y datasets transformers huggingface_hub accelerate peft bitsandbytes sentencepiece tokenizers torchvision || true
pip install -r requirements_kaggle.txt

if [[ -n "${HF_TOKEN_VALUE}" ]]; then
  python - <<PY
from huggingface_hub import login
login("${HF_TOKEN_VALUE}")
print("HF login done")
PY
fi

mkdir -p data/raw/avamerg data/raw/esconv outputs/sft

hf download ZhangHanXD/AvaMERG train.json --repo-type dataset --local-dir data/raw/avamerg

if [[ ! -f data/raw/esconv/ESConv.json ]]; then
  git clone https://github.com/thu-coai/Emotional-Support-Conversation.git /kaggle/working/esconv_repo
  cp /kaggle/working/esconv_repo/ESConv.json data/raw/esconv/ESConv.json
fi

python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-12B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/debug_joint \
  --max_length 1536 \
  --max_response_tokens 192 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --use_lora \
  --load_in_4bit \
  --lora_r 16 \
  --lora_alpha 32 \
  --dump_example_prompts

echo "Kaggle first run complete."

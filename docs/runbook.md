# Runbook

Runbook này gom repo về một luồng chính để train `Task 1` trên `Vast.ai` với:

- model: `google/gemma-4-26B-A4B-it`
- dữ liệu train: `AvaMERG + ESConv`
- hạ tầng mục tiêu: `1x RTX A6000 48GB`
- kiểu fine-tune: `4-bit + LoRA`

Notebook chính:

- [task1_gemma26b_a4b_it_vast_train.ipynb](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/task1_gemma26b_a4b_it_vast_train.ipynb>)

CLI tương đương:

- [scripts/start_gemma26b_a4b_it_vast.sh](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/scripts/start_gemma26b_a4b_it_vast.sh>)
- [scripts/train_sft.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/scripts/train_sft.py>)

## 1. Chuẩn bị máy Vast

Khuyến nghị instance:

- `RTX A6000 48GB`
- CUDA `12.x`
- Ubuntu `22.04`
- ít nhất `150GB` disk trống

## 2. Clone repo và tạo môi trường

```bash
git clone https://github.com/QuangVoAI/multimodal-empathy-mental-health.git
cd multimodal-empathy-mental-health
bash environment_setup.sh
source .venv310/bin/activate
```

Nếu máy không có `python3.10`, script sẽ tự fallback sang `python3` hoặc `python`.

## 3. Login Hugging Face

```bash
hf auth login
```

## 4. Tải dữ liệu

```bash
bash scripts/download_avamerg.sh
bash scripts/download_esconv.sh
```

Sau bước này cần có:

- `data/raw/avamerg/train.json`
- `data/raw/esconv/ESConv.json`

## 5. Kiểm tra model access

```bash
python - <<'PY'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
print("Processor ok:", processor.__class__.__name__)
PY
```

Gemma 4 26B A4B-it là model instruction multimodal. Theo model card chính thức, luồng khởi tạo chuẩn dùng `AutoProcessor` và `AutoModelForCausalLM`.  
Nguồn: [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it), [google/gemma-4-26B-A4B-it-assistant](https://huggingface.co/google/gemma-4-26B-A4B-it-assistant)

## 6. Chạy theo đúng thứ tự

### 6.1 Dump prompt

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh dump
```

### 6.2 Smoke test

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh smoke
```

Smoke hiện được giới hạn `16` sample và `1` optimizer step để bắt lỗi môi trường sớm.

### 6.3 Train thật

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh train 2>&1 | tee train_gemma26b_a4b_it.log
```

## 7. Lệnh train trực tiếp

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/task1_gemma26b_a4b_it_vast \
  --load_in_4bit \
  --use_lora \
  --gradient_checkpointing \
  --max_length 1024 \
  --max_response_tokens 192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 100 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
```

## 8. Upload checkpoint

Sau khi train xong, folder cuối sẽ nằm ở:

- `outputs/sft/task1_gemma26b_a4b_it_vast/final`

Upload:

```bash
python scripts/publish_to_hub.py \
  --repo_id SpringWang08/multimodal-empathy-mental-health-gemma26b-task1 \
  --folder_path outputs/sft/task1_gemma26b_a4b_it_vast/final
```

## 9. Nếu gặp OOM

Giảm theo thứ tự này:

1. `max_length: 1024 -> 768`
2. `gradient_accumulation_steps: 8 -> 4`
3. `lora_r: 16 -> 8`
4. smoke chỉ với `--max_train_samples 8`

## 10. File chính cần nhìn khi có lỗi

- [scripts/train_sft.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/scripts/train_sft.py>)
- [src/models/gemma_merg.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/src/models/gemma_merg.py>)
- [docs/vast_a6000_gemma26b_a4b_it.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/vast_a6000_gemma26b_a4b_it.md>)

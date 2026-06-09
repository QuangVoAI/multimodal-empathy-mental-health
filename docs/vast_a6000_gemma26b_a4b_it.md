# Vast.ai A6000 48GB plan for `google/gemma-4-26B-A4B-it`

Day la nhanh train rieng cho may:

- `1x RTX A6000 48GB`
- gia khoang `$0.42/hr`
- `google/gemma-4-26B-A4B-it`
- `Task 1`
- `AvaMERG + ESConv`

## 1. Vi sao may nay hop

Theo model card chinh thuc:

- `Gemma 4 26B A4B` la model MoE
- `25.2B` tong tham so
- chi `3.8B` active params moi token
- nham den consumer GPUs / workstations

Nguon:

- [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it)

Voi `A6000 48GB`, minh khuyen:

- `4-bit + LoRA`
- `text-only Task 1`
- `max_length=1024` de bat dau

## 2. SSH vao may va clone repo

```bash
git clone https://github.com/QuangVoAI/multimodal-empathy-mental-health.git
cd multimodal-empathy-mental-health
```

## 3. Tao moi truong

```bash
bash environment_setup.sh
source .venv310/bin/activate
pip uninstall -y torch torchvision torchaudio bitsandbytes
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
pip install --no-cache-dir bitsandbytes==0.47.0
pip install -U git+https://github.com/huggingface/transformers.git
```

Buoc cap nhat `transformers` tu GitHub la de tranh loi tokenizer/processor cua Gemma 4 trong mot so ban release cu. Hugging Face da co issue ve loi `AttributeError: 'list' object has no attribute 'keys'` khi doc `extra_special_tokens` cua Gemma 4 bang `transformers v4` cu.  
Nguon: [Issue #45376](https://github.com/huggingface/transformers/issues/45376)

Buoc reinstall `torch` + `bitsandbytes` la de tranh tinh trang moi truong keo ve torch CUDA `13.0`, sau do `bitsandbytes` di tim `libbitsandbytes_cuda130.so` va fail. A6000 trong luong nay duoc khoi tao theo stack `cu128` de on dinh hon.

## 4. Dang nhap Hugging Face

```bash
hf auth login
```

Test nhanh:

```bash
python - <<'PY'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
print("Processor ok:", processor.__class__.__name__)
PY
```

Gemma 4 26B A4B-it theo model card chinh thuc duoc khoi tao bang `AutoProcessor` va `AutoModelForCausalLM`, nen repo nay da duoc sua theo luong do thay vi gia dinh tokenizer-text-only.

## 5. Tai du lieu

### AvaMERG

```bash
mkdir -p data/raw/avamerg
hf download ZhangHanXD/AvaMERG train.json --repo-type dataset --local-dir data/raw/avamerg
```

### ESConv

```bash
git clone https://github.com/thu-coai/Emotional-Support-Conversation.git /tmp/esconv_repo
mkdir -p data/raw/esconv
cp /tmp/esconv_repo/ESConv.json data/raw/esconv/ESConv.json
```

## 6. Chay theo thu tu nay

### 6.1 Dump prompt

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh dump
```

### 6.2 Smoke test 1 step

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh smoke
```

Smoke mode hien gioi han `16` sample va `1` optimizer step de kiem tra:

- access model
- tokenizer / processor
- quantization
- LoRA
- save checkpoint

### 6.3 Train that

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh train 2>&1 | tee train_gemma26b_a4b_it.log
```

## 7. Lenh train truc tiep

Neu ban muon bo qua wrapper script, day la lenh day du:

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

## 8. Theo doi trong luc train

```bash
nvidia-smi -l 2
```

## 9. Neu bi OOM

Giam theo thu tu:

1. `gradient_accumulation_steps` giu nguyen, khong giam truoc
2. `max_length`: `1024 -> 768`
3. neu can nua, tat thoi chi dung `AvaMERG` de smoke

Khong nen tang `max_length` ngay tu dau.

## 10. Notebook VM

Neu ban muon chay bang Jupyter notebook ngay tren may Vast, dung:

- [task1_gemma26b_a4b_it_vast_train.ipynb](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/task1_gemma26b_a4b_it_vast_train.ipynb>)

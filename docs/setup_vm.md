# Setup on VM / Cloud GPU

## Mục tiêu

Máy ảo / cloud GPU là phương án chính nếu:

- Kaggle không đủ tài nguyên
- bạn muốn train thật với `Gemma 4 26B A4B`
- bạn muốn lưu checkpoint và chạy thí nghiệm lặp lại

---

## 1. Loại máy nên chọn

Ưu tiên:

- Linux + CUDA
- GPU VRAM lớn
- đủ disk để chứa:
  - model weights
  - dataset
  - checkpoints

### Khuyến nghị thực dụng

Bạn không cần bắt đầu với máy đắt nhất.

Trình tự nên là:

1. thuê máy vừa đủ để **smoke test**
2. nếu ổn mới nâng cấu hình cho run dài

---

## 2. Chuẩn bị source code

Trên máy ảo:

```bash
git clone <your-repo-url>
cd empathy_mh_gemma
```

Hoặc copy thẳng thư mục code lên máy.

---

## 3. Tạo môi trường

### 3.1 Cài Python nếu cần

Khuyến nghị:

- Python 3.10 hoặc 3.11

### 3.2 Chạy setup script

```bash
bash environment_setup.sh
source .venv310/bin/activate
```

Nếu bạn muốn chỉ định Python:

```bash
PYTHON_BIN=python3.10 bash environment_setup.sh
source .venv310/bin/activate
```

---

## 4. Đăng nhập Hugging Face

```bash
hf auth login
```

Test nhanh:

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it", token=True)
print("Tokenizer ok:", tok.pad_token, tok.eos_token)
PY
```

---

## 5. Chuẩn bị dữ liệu

### 5.1 AvaMERG

```bash
mkdir -p data/raw/avamerg
hf download ZhangHanXD/AvaMERG train.json --repo-type dataset --local-dir data/raw/avamerg
hf download ZhangHanXD/AvaMERG test.json --repo-type dataset --local-dir data/raw/avamerg
```

Nếu cần multimodal thật sau đó:

```bash
hf download ZhangHanXD/AvaMERG train_audio.zip --repo-type dataset --local-dir data/raw/avamerg
hf download ZhangHanXD/AvaMERG train_video.zip --repo-type dataset --local-dir data/raw/avamerg
```

### 5.2 ESConv

```bash
git clone https://github.com/thu-coai/Emotional-Support-Conversation.git /tmp/esconv_repo
mkdir -p data/raw/esconv
cp /tmp/esconv_repo/ESConv.json data/raw/esconv/ESConv.json
```

---

## 6. Chạy prompt dump trước

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/debug_joint \
  --dump_example_prompts
```

Kiểm tra:

```bash
sed -n '1,200p' outputs/sft/debug_joint/example_prompts.json
```

---

## 7. Smoke test train

### 7.1 Text-only smoke

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --output_dir outputs/sft/avamerg_smoke \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1
```

### 7.2 Joint smoke

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/joint_smoke \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1
```

---

## 8. Bật LoRA sau khi smoke run ổn

Khi:

- model load được
- loss chạy được
- batch đầu không lỗi

thì mới chuyển sang LoRA:

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/joint_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --use_lora \
  --logging_steps 1
```

---

## 9. Khuyến nghị vận hành

### Lưu log

```bash
python scripts/train_sft.py ... 2>&1 | tee train.log
```

### Chạy nền

```bash
nohup python scripts/train_sft.py ... > train.out 2>&1 &
```

### Theo dõi GPU

```bash
nvidia-smi -l 2
```

---

## 10. Trình tự tối ưu

Trên VM, đi theo đúng thứ tự này:

1. environment
2. HF auth
3. tải dữ liệu
4. dump prompt
5. `max_steps=1`
6. `max_steps=5`
7. LoRA thật

---

## 11. Kết luận

Máy ảo / cloud GPU là nơi phù hợp để:

- train chính
- lưu checkpoint
- làm thí nghiệm lặp lại

Trong khi Kaggle chỉ nên là:

- chỗ test đường ống


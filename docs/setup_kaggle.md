# Setup on Kaggle

## Mục tiêu

Kaggle là **môi trường train chính** cho giai đoạn hiện tại của project.

Mục tiêu trên Kaggle là:

- chuẩn bị environment train cho `Gemma 4 26B A4B`
- tải và chuẩn hóa `AvaMERG + ESConv`
- chạy `prompt dump`
- chạy **LoRA SFT** trên Kaggle GPU
- lưu checkpoint và output ra thư mục notebook

> Để phù hợp với Kaggle GPU, cấu hình nên ưu tiên là **4-bit + LoRA**, không phải full fine-tuning.

---

## 1. Chuẩn bị trước khi upload lên Kaggle

Bạn nên mang lên Kaggle thư mục:

- [empathy_mh_gemma](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma>)

Các file quan trọng:

- `requirements.txt`
- `environment_setup.sh`
- `scripts/train_sft.py`
- `src/data/*.py`
- `src/models/gemma_merg.py`

### Không cần upload

- `.venv310`
- checkpoint cũ
- file lớn không dùng đến

---

## 2. Tạo Kaggle Notebook

Khuyến nghị:

- bật **Internet**
- bật **GPU**
- chọn runtime có GPU mạnh nhất bạn có thể dùng

### Ưu tiên cấu hình

- nếu có GPU mạnh hơn thì chọn GPU mạnh hơn
- nếu chỉ có T4 / P100, vẫn thử được, nhưng nên giữ:
  - `--load_in_4bit`
  - `--use_lora`
  - batch nhỏ
  - gradient accumulation cao hơn

---

## 3. Đưa code lên Kaggle

Bạn có 2 cách:

### Cách A — upload zip

Nén thư mục `empathy_mh_gemma` rồi upload vào Kaggle Notebook.

### Cách B — dùng GitHub

Push project lên repo riêng rồi clone trong notebook:

```bash
!git clone <your-repo-url>
%cd empathy_mh_gemma
```

---

## 4. Cài môi trường trong Kaggle

Trong một cell đầu:

```bash
!git clone https://github.com/QuangVoAI/multimodal-empathy-mental-health.git
%cd /kaggle/working/multimodal-empathy-mental-health
%pip uninstall -y datasets transformers huggingface_hub accelerate peft bitsandbytes sentencepiece tokenizers torchvision
%pip install --no-cache-dir --force-reinstall -r requirements_kaggle.txt
```

Khong can cai lai `torch` tren Kaggle. Muc tieu la giu nguyen bo `torch/CUDA` da co san cua Kaggle, chi cap nhat cac package Hugging Face can cho Gemma.

Notebook Task 1 khong can `torchvision`. Tren Kaggle, `torchvision` preinstalled doi khi lech version voi `torch` va gay loi import `torchvision::nms`, nen minh chu dong go no ra khoi train flow.

Sau cell cài package, **restart kernel / session ngay** rồi mới chạy tiếp từ cell đăng nhập Hugging Face. Nếu không, Kaggle có thể giữ lại module cũ trong bộ nhớ và gây lỗi import lệch version.

---

## 5. Đăng nhập Hugging Face

Trong notebook:

```python
from huggingface_hub import login
login("YOUR_HF_TOKEN")
```

Sau đó test nhanh:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it", token=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(type(tokenizer).__name__, tokenizer.pad_token, tokenizer.eos_token)
```

Nếu bước này fail, dừng lại luôn và sửa quyền / token trước.

---

## 6. Chuẩn bị dữ liệu trên Kaggle

### 6.1 AvaMERG

Bạn có thể:

- upload `train.json`
- hoặc tải lại từ Hugging Face trong notebook

Ví dụ:

```bash
!pip install -U huggingface_hub
!hf download ZhangHanXD/AvaMERG train.json --repo-type dataset --local-dir data/raw/avamerg
```

### 6.2 ESConv

Bạn có thể:

- upload `ESConv.json`
- hoặc clone repo rồi copy file

Ví dụ:

```bash
!git clone https://github.com/thu-coai/Emotional-Support-Conversation.git /kaggle/working/esconv_repo
!mkdir -p data/raw/esconv
!cp /kaggle/working/esconv_repo/ESConv.json data/raw/esconv/ESConv.json
```

---

## 7. Chạy prompt dump trước

Đây là bước rất nên làm trước khi train.

```bash
!python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/debug_joint \
  --dump_example_prompts
```

Sau đó mở:

- `outputs/sft/debug_joint/example_prompts.json`

---

## 8. Smoke test nhỏ trên Kaggle

Mục tiêu:

- kiểm tra pipeline train chạy được
- không cố train lâu

```bash
!python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --output_dir outputs/sft/avamerg_smoke \
  --load_in_4bit \
  --use_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1
```

### Nếu muốn thử joint setup

```bash
!python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/joint_smoke \
  --load_in_4bit \
  --use_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1
```

---

## 9. Train thật trên Kaggle

Sau khi smoke test ổn, chạy LoRA SFT trên `AvaMERG + ESConv`.

### 9.1 Train chỉ với AvaMERG trước

```bash
!python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --output_dir outputs/sft/avamerg_lora \
  --load_in_4bit \
  --use_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 100
```

### 9.2 Train joint với AvaMERG + ESConv

```bash
!python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/joint_lora \
  --load_in_4bit \
  --use_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 100
```

### 9.3 Nếu gặp OOM

Giảm theo thứ tự:

1. giữ `--load_in_4bit`
2. giữ `--use_lora`
3. giảm `--per_device_train_batch_size` về `1`
4. tăng `--gradient_accumulation_steps`
5. giảm `--max_length` trong script nếu cần chỉnh thêm

---

## 10. Lưu output trên Kaggle

Ưu tiên lưu vào:

- `/kaggle/working/empathy_mh_gemma/outputs/...`

Cuối phiên, bạn có thể:

- download output
- hoặc lưu notebook version
- hoặc đồng bộ checkpoint ra nơi khác sau

---

## 11. Kết luận

Với project này, Kaggle được dùng như:

- **môi trường train chính**
- theo hướng **4-bit + LoRA**

Trình tự chuẩn là:

1. cài môi trường
2. đăng nhập HF
3. tải data
4. dump prompt
5. smoke test 1 step
6. chạy LoRA train thật

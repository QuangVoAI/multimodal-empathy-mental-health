# Runbook

## Mục tiêu

Runbook này giúp chuyển từ:

- ý tưởng đề tài
- scaffold code
- prompt design

sang:

- đọc dữ liệu thật
- kiểm tra prompt từ dữ liệu thật
- chạy một vòng SFT nhỏ với `Gemma 4 26B`

Trình tự được tối ưu cho **Task 1**.

---

## 0. Cấu trúc hiện tại

Thư mục code chính:

- [empathy_mh_gemma](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma>)

Các file quan trọng:

- [prompt_design.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/prompt_design.md>)
- [data_schema_examples.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/data_schema_examples.md>)
- [train_sft.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma/scripts/train_sft.py>)
- [avamerg_dataset.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma/src/data/avamerg_dataset.py>)
- [esconv_dataset.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma/src/data/esconv_dataset.py>)
- [gemma_merg.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma/src/models/gemma_merg.py>)

---

## 1. Việc đầu tiên: đọc note trước khi chạy

### 1.1 Đọc prompt design

Đọc:

- [prompt_design.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/prompt_design.md>)

Mục tiêu:

- chốt vai trò của model
- chốt system prompt
- chốt 3 baseline prompt

### 1.2 Đọc mock schema examples

Đọc:

- [data_schema_examples.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/data_schema_examples.md>)

Mục tiêu:

- hình dung sample thật nên trông như thế nào
- biết mình sẽ kiểm tra cái gì khi mở dữ liệu thật

---

## 2. Chuẩn bị môi trường

### 2.1 Tạo virtual environment

```bash
cd "/Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2.2 Cài package tối thiểu

```bash
pip install torch transformers datasets accelerate peft sentencepiece
```

### 2.3 Nếu muốn dùng Hugging Face CLI

```bash
pip install "huggingface_hub[cli]"
hf auth login
```

---

## 3. Tạo thư mục dữ liệu

```bash
mkdir -p data/raw/avamerg
mkdir -p data/raw/esconv
mkdir -p data/processed
mkdir -p outputs/sft
```

---

## 4. Tải dữ liệu thật

## 4.1 AvaMERG

Ưu tiên tải:

- `train.json`
- `valid.json` hoặc `test.json`
- một phần thư mục `audio`
- một phần thư mục `video`

Nếu dùng `hf`:

```bash
hf download ZhangHanXD/AvaMERG train.json --repo-type dataset --local-dir data/raw/avamerg
hf download ZhangHanXD/AvaMERG valid.json --repo-type dataset --local-dir data/raw/avamerg
```

Nếu dataset card dùng tên split khác, hãy điều chỉnh theo file thật trên Hub.

### Nếu muốn tải media sau

Bạn có thể bắt đầu chỉ với:

- `train.json`
- `valid.json`

và chạy baseline text-only trước.

## 4.2 ESConv

Tải repo hoặc file JSON thật của ESConv vào:

```bash
data/raw/esconv/
```

Ví dụ:

```bash
git clone https://github.com/thu-coai/Emotional-Support-Conversation.git /tmp/esconv_repo
```

Sau đó xác định file JSON cần dùng và copy vào:

```bash
cp /tmp/esconv_repo/path/to/your_split.json data/raw/esconv/
```

### Việc quan trọng

Khi có file thật, mở nó ra ngay. `ESConvDataset` hiện là scaffold linh hoạt, nhưng gần như chắc chắn sẽ cần chỉnh nhẹ theo format thực.

---

## 5. Kiểm tra dữ liệu thật bằng mắt

## 5.1 Mở vài dòng của AvaMERG

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/raw/avamerg/train.json")
data = json.loads(p.read_text(encoding="utf-8"))
print("Num samples:", len(data))
print("Keys of first item:", data[0].keys())
print("First conversation_id:", data[0].get("conversation_id"))
print("Turn keys:", data[0]["turns"][-1].keys())
PY
```

### Bạn cần kiểm tra

- có đúng `turns[-1]` như scaffold đang giả định không
- tên key `dialogue_history`, `response`, `chain_of_empathy` có đúng không

## 5.2 Mở vài dòng của ESConv

```bash
python - <<'PY'
import json
from pathlib import Path

p = Path("data/raw/esconv/YOUR_FILE.json")
data = json.loads(p.read_text(encoding="utf-8"))
print(type(data))
if isinstance(data, dict):
    print("Top-level keys:", list(data.keys())[:20])
elif isinstance(data, list):
    print("Num items:", len(data))
    print("First item keys:", data[0].keys() if isinstance(data[0], dict) else type(data[0]))
PY
```

### Bạn cần kiểm tra

- file là `list` hay `dict`
- key thật của history là gì
- key thật của strategy là gì

---

## 6. So khớp với scaffold hiện tại

Sau khi mở dữ liệu thật:

### 6.1 Nếu AvaMERG khớp

Không cần sửa `AvaMERGDataset`.

### 6.2 Nếu ESConv khác format

Sửa:

- [esconv_dataset.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/empathy_mh_gemma/src/data/esconv_dataset.py>)

Cần chỉnh chủ yếu:

- `_iter_dialogues()`
- `_normalize_history()`
- `_extract_response()`
- `_extract_strategy()`

---

## 7. Dump prompt trước khi train

Đây là bước rất quan trọng.

### 7.1 Dump prompt từ AvaMERG text-only

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --output_dir outputs/sft/debug_avamerg_text \
  --dump_example_prompts
```

### 7.2 Dump prompt từ AvaMERG + ESConv

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/YOUR_FILE.json \
  --output_dir outputs/sft/debug_joint \
  --dump_example_prompts
```

> Lưu ý: model chính đã chốt là `google/gemma-4-26B-A4B-it`. Nếu cần smoke test nhẹ hơn vì giới hạn bộ nhớ, có thể tạm thay bằng một model instruct nhỏ hơn, nhưng đường chạy chính nên giữ theo model id này.

### 7.3 Mở file prompt dump

Xem:

- `outputs/sft/debug_*/example_prompts.json`

Bạn cần kiểm tra:

- prompt có tự nhiên không
- response target có khớp với prompt không
- strategy / CoE có bị quá nhiều không

---

## 8. Chạy train nhỏ trước

Không train lớn ngay.

Mục tiêu:

- kiểm tra pipeline end-to-end
- xem tokenize + masking loss có đúng không
- xem training loop chạy ổn không

### 8.1 Train nhỏ với AvaMERG text-only

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --output_dir outputs/sft/avamerg_text_smoke \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1 \
  --save_steps 20
```

### 8.2 Train nhỏ với joint setup

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/YOUR_FILE.json \
  --output_dir outputs/sft/joint_smoke \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 1 \
  --max_steps 1 \
  --logging_steps 1 \
  --save_steps 20
```

---

## 9. Khi nào mới dùng LoRA

Sau khi smoke test ổn:

- loader đúng
- prompt đúng
- training loop chạy được

lúc đó mới bật:

```bash
--use_lora
```

Ví dụ:

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-26B-A4B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/YOUR_FILE.json \
  --output_dir outputs/sft/joint_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --use_lora
```

---

## 10. Vì sao phải có `train_sft.py`

`train_sft.py` là bước nối giữa:

- dataset
- prompt
- tokenizer
- model
- supervised objective

Nếu không có SFT, bạn chỉ đang dựa vào:

- zero-shot prompting
- hoặc inference ad hoc

Điều đó chưa đủ để kết luận:

- model có thật sự học được phong cách `empathetic + supportive + safe` từ `AvaMERG + ESConv` hay không

### Vai trò của SFT ở đây

SFT giúp:

1. **ổn định hóa hành vi**
   - model ít trả lời lệch phong cách hơn
2. **học response distribution của dữ liệu**
   - thay vì chỉ “đoán theo prompt”
3. **tạo baseline nghiên cứu nghiêm túc**
   - để sau này so sánh text-only vs multimodal-aware vs safety-aware

### Nói đơn giản

Prompt tốt giúp model hiểu bạn muốn gì.  
**SFT giúp model làm điều đó ổn định hơn trên đúng bài toán của bạn.**

---

## 11. Checkpoint: sau bước này bạn cần có gì

Trước khi làm bước tiếp theo, bạn nên có:

- dữ liệu thật đã tải
- `AvaMERGDataset` chạy được
- `ESConvDataset` khớp format thật
- file `example_prompts.json`
- một smoke run train thành công

---

## 12. Nếu có lỗi, nên debug theo thứ tự này

1. lỗi file path
2. lỗi format JSON
3. lỗi prompt builder
4. lỗi tokenizer / chat template
5. lỗi padding / labels masking
6. lỗi model / GPU / memory

---

## 13. Bước tiếp theo sau smoke run

Sau khi smoke run ổn, bước tiếp theo là:

1. viết script inference
2. chạy vài sample held-out
3. xem chất lượng response bằng mắt
4. rồi mới mở rộng sang evaluation trên `MentalChat16K`

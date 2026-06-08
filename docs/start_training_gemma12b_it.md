# Start Training `google/gemma-4-12B-it`

Day mot checklist ngan de bat dau train `Task 1` voi:

- `AvaMERG + ESConv`
- `google/gemma-4-12B-it`
- LoRA SFT

## 1. Chon moi truong

Neu ban chi muon debug data/prompt:

- co the dung Kaggle

Neu ban muon train that:

- uu tien VM / cloud GPU

Repo nay van giu notebook Kaggle, nhung duong train on dinh nhat hien tai la:

- `scripts/train_sft.py`

## 2. Dang nhap Hugging Face

```bash
hf auth login
```

## 3. Tai du lieu

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

## 4. Kiem tra prompt truoc

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-12B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/debug_joint \
  --dump_example_prompts
```

Mo file:

- `outputs/sft/debug_joint/example_prompts.json`

## 5. Smoke test 1 step

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-12B-it \
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

Neu step nay qua duoc, ban da co duong train co ban.

## 6. Train that

```bash
python scripts/train_sft.py \
  --model_name_or_path google/gemma-4-12B-it \
  --avamerg_root data/raw/avamerg \
  --avamerg_split train \
  --avamerg_text_only \
  --esconv_json data/raw/esconv/ESConv.json \
  --output_dir outputs/sft/task1_gemma12b_it_joint \
  --load_in_4bit \
  --use_lora \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 \
  --num_train_epochs 1 \
  --logging_steps 10 \
  --save_steps 100
```

Checkpoint cuoi cung se nam o:

- `outputs/sft/task1_gemma12b_it_joint/final`

## 7. Sau khi train xong

Lam tiep theo thu tu:

1. generate predictions
2. export rubric
3. summarize results
4. publish len Hugging Face neu muon

Tai lieu lien quan:

- [post_training_pipeline.md](/Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/post_training_pipeline.md)
- [hf_publish_checklist.md](/Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/hf_publish_checklist.md)

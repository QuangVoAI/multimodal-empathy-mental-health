# Setup on VM / Cloud GPU

Tai lieu nay la ban tom gon cho nhanh `Vast.ai / A6000 48GB`.

Model chinh:

- `google/gemma-4-26B-A4B-it`

Luong chay chinh:

- [task1_gemma26b_a4b_it_vast_train.ipynb](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/task1_gemma26b_a4b_it_vast_train.ipynb>)
- [scripts/start_gemma26b_a4b_it_vast.sh](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/scripts/start_gemma26b_a4b_it_vast.sh>)

## 1. Chon may

Khuyen nghi:

- Ubuntu 22.04
- CUDA 12.x
- `1x RTX A6000 48GB`
- `>= 150GB` disk

## 2. Clone repo

```bash
git clone https://github.com/QuangVoAI/multimodal-empathy-mental-health.git
cd multimodal-empathy-mental-health
```

## 3. Tao moi truong

```bash
bash environment_setup.sh
source .venv310/bin/activate
```

## 4. Login Hugging Face

```bash
hf auth login
```

## 5. Test model access

```bash
python - <<'PY'
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained("google/gemma-4-26B-A4B-it")
print("Processor ok:", processor.__class__.__name__)
PY
```

## 6. Download data

```bash
bash scripts/download_avamerg.sh
bash scripts/download_esconv.sh
```

## 7. Chay theo thu tu

```bash
bash scripts/start_gemma26b_a4b_it_vast.sh dump
bash scripts/start_gemma26b_a4b_it_vast.sh smoke
bash scripts/start_gemma26b_a4b_it_vast.sh train
```

## 8. Upload checkpoint

```bash
python scripts/publish_to_hub.py \
  --repo_id SpringWang08/multimodal-empathy-mental-health-gemma26b-task1 \
  --folder_path outputs/sft/task1_gemma26b_a4b_it_vast/final
```

## 9. Doc tiep

- [docs/runbook.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/runbook.md>)
- [docs/vast_a6000_gemma26b_a4b_it.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/vast_a6000_gemma26b_a4b_it.md>)

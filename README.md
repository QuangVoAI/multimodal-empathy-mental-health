# multimodal-empathy-mental-health

Public research repo for **Task 1**: empathetic and safety-aware response generation for mental-health-oriented dialogue.

Current research setup:

- **Train / adapt:** `AvaMERG + ESConv`
- **Eval:** `MentalChat16K`
- **Main training path:** `google/gemma-4-26B-A4B-it`
- **Target infra:** `Vast.ai / RTX A6000 48GB`

## Main entrypoints

Notebook:

- [task1_gemma26b_a4b_it_vast_train.ipynb](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/task1_gemma26b_a4b_it_vast_train.ipynb>)
- [notebooks/task1_gemma26b_a4b_it_vast_train.ipynb](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/notebooks/task1_gemma26b_a4b_it_vast_train.ipynb>)

CLI:

- [scripts/start_gemma26b_a4b_it_vast.sh](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/scripts/start_gemma26b_a4b_it_vast.sh>)
- [scripts/train_sft.py](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/scripts/train_sft.py>)

## What the repo supports now

- unified dataset loading for `AvaMERG` and `ESConv`
- prompt construction for supportive mental-health dialogue
- `4-bit + LoRA` SFT scaffold for Gemma-family models
- prompt dump before training
- smoke-test training path on a small sample slice
- checkpoint upload to Hugging Face Hub

## Recommended workflow

1. Clone repo on a Vast A6000 instance
2. Run [environment_setup.sh](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/environment_setup.sh>)
3. Login Hugging Face
4. Download `AvaMERG` and `ESConv`
5. Run prompt dump
6. Run smoke test
7. Run full training
8. Upload `outputs/sft/<run_name>/final` to Hugging Face

## Quick start

```bash
git clone https://github.com/QuangVoAI/multimodal-empathy-mental-health.git
cd multimodal-empathy-mental-health
bash environment_setup.sh
source .venv310/bin/activate
hf auth login
bash scripts/download_avamerg.sh
bash scripts/download_esconv.sh
bash scripts/start_gemma26b_a4b_it_vast.sh dump
bash scripts/start_gemma26b_a4b_it_vast.sh smoke
bash scripts/start_gemma26b_a4b_it_vast.sh train
```

## Repo structure

```text
multimodal-empathy-mental-health/
├── README.md
├── requirements.txt
├── environment_setup.sh
├── configs/
├── data/
├── docs/
├── notebooks/
├── outputs/
├── scripts/
└── src/
```

## Important docs

- [docs/runbook.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/runbook.md>)
- [docs/vast_a6000_gemma26b_a4b_it.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/vast_a6000_gemma26b_a4b_it.md>)
- [docs/setup_vm.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/setup_vm.md>)
- [docs/post_training_pipeline.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/post_training_pipeline.md>)
- [docs/hf_publish_checklist.md](</Users/springwang/Library/Mobile Documents/com~apple~CloudDocs/juniorYear/Research/mental health/multimodal-empathy-mental-health/docs/hf_publish_checklist.md>)

## Notes

- `AvaMERG` cung cap phan hoi dong cam va context affective
- `ESConv` bo sung support strategy theo hoi thoai ho tro cam xuc
- `MentalChat16K` duoc giu cho eval thay vi train
- repo hien tap trung vao **text response generation** cho Task 1

Kaggle va Unsloth notebooks cu van duoc giu lai nhu tai lieu tham khao, nhung duong chay duoc repo uu tien va da don lai la luong `Vast + Gemma 4 26B A4B-it`.

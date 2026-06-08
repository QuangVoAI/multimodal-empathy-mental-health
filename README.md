# multimodal-empathy-mental-health

Public research repo for **Task 1**: empathetic and safety-aware response generation for mental health-oriented dialogue using:

- **Train / adapt:** `AvaMERG + ESConv`
- **Eval:** `MentalChat16K`
- **Main model:** `google/gemma-4-26B-A4B-it`

This repo is designed to be **Kaggle-first**:

- the main entrypoint is a Kaggle notebook
- training code lives in importable Python modules
- the project layout stays clean enough for longer-term research work

## Repo structure

```text
multimodal-empathy-mental-health/
├── README.md
├── requirements.txt
├── environment_setup.sh
├── .gitignore
├── configs/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── runbook.md
│   ├── setup_kaggle.md
│   └── setup_vm.md
├── notebooks/
│   └── task1_gemma26b_train.ipynb
├── outputs/
├── scripts/
│   ├── train_sft.py
│   ├── kaggle_first_run.sh
│   ├── download_avamerg.sh
│   └── download_esconv.sh
└── src/
    ├── data/
    └── models/
```

## What this repo currently supports

- unified dataset loading for `AvaMERG` and `ESConv`
- prompt construction for `Gemma 4 26B A4B`
- Kaggle-ready `4-bit + LoRA` SFT scaffold
- prompt dump before training
- smoke-test training path

## Recommended workflow

1. Open the Kaggle notebook:
   - [notebooks/task1_gemma26b_train.ipynb](notebooks/task1_gemma26b_train.ipynb)
2. Login to Hugging Face
3. Download `AvaMERG` and `ESConv`
4. Dump prompts and inspect them
5. Run a `max_steps=1` smoke test
6. Launch `4-bit + LoRA` training

## Kaggle-first bootstrap

In a fresh Kaggle notebook, you can start with:

```bash
!git clone <YOUR_GITHUB_REPO_URL>
%cd multimodal-empathy-mental-health
!bash environment_setup.sh
```

Then download the datasets:

```bash
!bash scripts/download_avamerg.sh
!bash scripts/download_esconv.sh
```

## Main notebook

Use:

- [notebooks/task1_gemma26b_train.ipynb](notebooks/task1_gemma26b_train.ipynb)

The notebook imports and calls `run_training(...)` from:

- [scripts/train_sft.py](scripts/train_sft.py)

## Setup docs

- Kaggle: [docs/setup_kaggle.md](docs/setup_kaggle.md)
- VM / cloud GPU: [docs/setup_vm.md](docs/setup_vm.md)
- General runbook: [docs/runbook.md](docs/runbook.md)
- Post-training pipeline: [docs/post_training_pipeline.md](docs/post_training_pipeline.md)
- Hugging Face publish notes: [docs/hf_publish_checklist.md](docs/hf_publish_checklist.md)

## Notes

- `AvaMERG` is used as the **multimodal core dataset**
- `ESConv` is used as the **support-strategy dataset**
- `MentalChat16K` is reserved for evaluation rather than early-stage training
- this repo currently focuses on **Task 1 text response generation**

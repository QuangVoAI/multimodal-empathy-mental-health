# multimodal-empathy-mental-health

Public research repo for **Task 1**: empathetic and safety-aware response generation for mental health-oriented dialogue using:

- **Train / adapt:** `AvaMERG + ESConv`
- **Eval:** `MentalChat16K`
- **Main training model:** `google/gemma-4-12B-it`

This repo is designed to be **Kaggle-first**:

- the main entrypoint is a Kaggle notebook
- training code lives in importable Python modules
- the project layout stays clean enough for longer-term research work

## Repo structure

```text
multimodal-empathy-mental-health/
├── README.md
├── requirements.txt
├── requirements_kaggle.txt
├── requirements_kaggle_unsloth.txt
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
│   └── task1_gemma12b_train.ipynb
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
- prompt construction for `Gemma 4 12B`
- importable Transformers + LoRA SFT scaffold
- experimental Unsloth notebook kept for comparison
- prompt dump before training
- smoke-test training path

## Recommended workflow

1. Open the main notebook:
   - [notebooks/task1_gemma12b_train.ipynb](notebooks/task1_gemma12b_train.ipynb)
2. Login to Hugging Face
3. Download `AvaMERG` and `ESConv`
4. Dump prompts and inspect them
5. Run a `max_steps=1` smoke test
6. Launch `4-bit + LoRA` training with `google/gemma-4-12B-it`

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

- [notebooks/task1_gemma12b_train.ipynb](notebooks/task1_gemma12b_train.ipynb)

Notebook hien tai imports and calls `run_training(...)` from:

- [scripts/train_sft.py](scripts/train_sft.py)

Experimental notebook:

- [notebooks/task1_gemma12b_unsloth_train.ipynb](notebooks/task1_gemma12b_unsloth_train.ipynb)

## Setup docs

- Kaggle: [docs/setup_kaggle.md](docs/setup_kaggle.md)
- Start training checklist: [docs/start_training_gemma12b_it.md](docs/start_training_gemma12b_it.md)
- VM / cloud GPU: [docs/setup_vm.md](docs/setup_vm.md)
- General runbook: [docs/runbook.md](docs/runbook.md)
- Post-training pipeline: [docs/post_training_pipeline.md](docs/post_training_pipeline.md)
- Hugging Face publish notes: [docs/hf_publish_checklist.md](docs/hf_publish_checklist.md)

## Notes

- `AvaMERG` is used as the **multimodal core dataset**
- `ESConv` is used as the **support-strategy dataset**
- `MentalChat16K` is reserved for evaluation rather than early-stage training
- this repo currently focuses on **Task 1 text response generation**


## Note on Unsloth

The repo still keeps an experimental Unsloth notebook because Gemma 4 appears in Unsloth's public model catalog and there is a hosted `unsloth/gemma-4-12b-it` variant. But after repeated Kaggle runtime issues, the main supported training path in this repo is back to `google/gemma-4-12B-it` through the plain Transformers + LoRA scaffold.

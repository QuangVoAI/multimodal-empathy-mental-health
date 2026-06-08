from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:  # pragma: no cover
    LoraConfig = None
    TaskType = None
    get_peft_model = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AvaMERGDataset, ESConvDataset  # noqa: E402
from src.models.gemma_merg import DEFAULT_SYSTEM_PROMPT, GemmaMERG  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SFT scaffold for Gemma 4 26B Task 1")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--avamerg_root", type=str, default=None)
    parser.add_argument("--avamerg_split", type=str, default="train")
    parser.add_argument("--avamerg_text_only", action="store_true")
    parser.add_argument("--esconv_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "sft"))
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_response_tokens", type=int, default=256)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--dump_example_prompts", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def build_datasets(args: argparse.Namespace):
    datasets = []
    if args.avamerg_root:
        datasets.append(
            AvaMERGDataset(
                root=args.avamerg_root,
                split=args.avamerg_split,
                use_multimodal=not args.avamerg_text_only,
            )
        )
    if args.esconv_json:
        datasets.append(ESConvDataset(args.esconv_json))
    if not datasets:
        raise ValueError("At least one dataset must be provided: --avamerg_root and/or --esconv_json")
    return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)


def build_response_target(response: str) -> str:
    return response.strip()


def build_user_prompt(prompt_builder: GemmaMERG, sample: Dict[str, Any]) -> str:
    return prompt_builder.build_user_prompt(sample)


def tokenize_supervised_sample(
    sample: Dict[str, Any],
    tokenizer: AutoTokenizer,
    prompt_builder: GemmaMERG,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    user_prompt = build_user_prompt(prompt_builder, sample)
    target_response = build_response_target(sample["response"])

    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": target_response}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    full_tokens = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )
    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    prompt_len = min(len(prompt_ids), len(input_ids))
    labels = [-100] * prompt_len + input_ids[prompt_len:]
    labels = labels[: len(input_ids)]
    if len(labels) < len(input_ids):
        labels += [-100] * (len(input_ids) - len(labels))

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SupervisedTask1Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        tokenizer: AutoTokenizer,
        prompt_builder: GemmaMERG,
        max_length: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.prompt_builder = prompt_builder
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        return tokenize_supervised_sample(
            sample=sample,
            tokenizer=self.tokenizer,
            prompt_builder=self.prompt_builder,
            max_length=self.max_length,
        )


class SFTDataCollator:
    def __init__(self, tokenizer: AutoTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def maybe_wrap_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("peft is not installed but --use_lora was requested.")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_trainable_model(model_name_or_path: str, load_in_4bit: bool = False):
    load_kwargs = {
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    try:
        return AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    except Exception:
        return AutoModelForImageTextToText.from_pretrained(model_name_or_path, **load_kwargs)


def dump_example_prompts(dataset, prompt_builder: GemmaMERG, output_dir: Path, num_examples: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = []
    for idx in range(min(num_examples, len(dataset))):
        sample = dataset[idx]
        examples.append(
            {
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset"),
                "user_prompt": prompt_builder.build_user_prompt(sample),
                "target_response": sample.get("response"),
            }
        )
    with (output_dir / "example_prompts.json").open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)


def run_training(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = build_datasets(args)

    prompt_builder = GemmaMERG.__new__(GemmaMERG)
    prompt_builder.model_name_or_path = args.model_name_or_path

    if args.dump_example_prompts:
        dump_example_prompts(base_dataset, prompt_builder, output_dir)
        print(f"Saved example prompts to {output_dir / 'example_prompts.json'}")
        return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_builder.tokenizer = tokenizer

    train_dataset = SupervisedTask1Dataset(
        base_dataset=base_dataset,
        tokenizer=tokenizer,
        prompt_builder=prompt_builder,
        max_length=args.max_length,
    )
    collator = SFTDataCollator(tokenizer)

    model = load_trainable_model(args.model_name_or_path, load_in_4bit=args.load_in_4bit)
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model.to("mps")
        else:
            model.to("cpu")
    model = maybe_wrap_lora(model, args)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to=[] if args.report_to == "none" else [args.report_to],
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

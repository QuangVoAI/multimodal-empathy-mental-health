from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
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
    parser = argparse.ArgumentParser(description="SFT scaffold for Task 1 on Gemma-family models")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--avamerg_root", type=str, default=None)
    parser.add_argument("--avamerg_split", type=str, default="train")
    parser.add_argument("--avamerg_text_only", action="store_true")
    parser.add_argument("--esconv_json", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "sft"))
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_response_tokens", type=int, default=192)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--dump_example_prompts", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
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
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    if args.max_train_samples is not None:
        sample_count = min(args.max_train_samples, len(combined))
        combined = Subset(combined, range(sample_count))
    return combined


def build_response_target(response: str) -> str:
    return response.strip()


def build_user_prompt(prompt_builder: GemmaMERG, sample: Dict[str, Any]) -> str:
    return prompt_builder.build_user_prompt(sample)


def tokenize_supervised_sample(
    sample: Dict[str, Any],
    tokenizer,
    chat_template_handler,
    prompt_builder: GemmaMERG,
    max_length: int,
    max_response_tokens: int,
) -> Dict[str, torch.Tensor]:
    user_prompt = build_user_prompt(prompt_builder, sample)
    raw_target_response = build_response_target(sample["response"])

    if max_response_tokens and max_response_tokens > 0:
        response_ids = tokenizer(
            raw_target_response,
            add_special_tokens=False,
            truncation=True,
            max_length=max_response_tokens,
        ).input_ids
        target_response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    else:
        target_response = raw_target_response

    prompt_messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    full_messages = prompt_messages + [{"role": "assistant", "content": target_response}]

    prompt_text = chat_template_handler.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = chat_template_handler.apply_chat_template(
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
        tokenizer,
        chat_template_handler,
        prompt_builder: GemmaMERG,
        max_length: int,
        max_response_tokens: int,
    ) -> None:
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.chat_template_handler = chat_template_handler
        self.prompt_builder = prompt_builder
        self.max_length = max_length
        self.max_response_tokens = max_response_tokens

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.base_dataset[idx]
        return tokenize_supervised_sample(
            sample=sample,
            tokenizer=self.tokenizer,
            chat_template_handler=self.chat_template_handler,
            prompt_builder=self.prompt_builder,
            max_length=self.max_length,
            max_response_tokens=self.max_response_tokens,
        )


class SFTDataCollator:
    def __init__(self, tokenizer) -> None:
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


def move_batch_to_model_device(batch: Dict[str, torch.Tensor], model) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    return {k: v.to(device) for k, v in batch.items()}


def save_model_and_tokenizer(model, tokenizer, save_dir: Path, processor=None) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    if processor is not None and hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(save_dir))


def run_manual_training_loop(
    model,
    tokenizer,
    processor,
    train_dataset,
    collator: SFTDataCollator,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
    )

    steps_per_epoch = math.ceil(len(dataloader) / max(args.gradient_accumulation_steps, 1))
    total_steps = int(args.max_steps) if args.max_steps and args.max_steps > 0 else max(
        1, int(steps_per_epoch * args.num_train_epochs)
    )
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    global_step = 0
    running_loss = 0.0
    accumulation_counter = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(max(1, math.ceil(args.num_train_epochs))):
        for step, batch in enumerate(dataloader, start=1):
            batch = move_batch_to_model_device(batch, model)
            outputs = model(**batch)
            loss = outputs.loss / max(args.gradient_accumulation_steps, 1)
            loss.backward()
            running_loss += loss.item()
            accumulation_counter += 1

            is_accumulation_boundary = accumulation_counter >= max(args.gradient_accumulation_steps, 1)
            is_last_batch = step == len(dataloader)
            if is_accumulation_boundary or is_last_batch:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accumulation_counter = 0

                if global_step % args.logging_steps == 0:
                    print(
                        f"[train] epoch={epoch + 1} step={global_step}/{total_steps} "
                        f"loss={running_loss / args.logging_steps:.4f}"
                    )
                    running_loss = 0.0

                if global_step % args.save_steps == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    save_model_and_tokenizer(model, tokenizer, ckpt_dir, processor=processor)
                    print(f"[train] saved checkpoint to {ckpt_dir}")

                if global_step >= total_steps:
                    final_dir = output_dir / "final"
                    save_model_and_tokenizer(model, tokenizer, final_dir, processor=processor)
                    print(f"[train] saved final checkpoint to {final_dir}")
                    return

    final_dir = output_dir / "final"
    save_model_and_tokenizer(model, tokenizer, final_dir, processor=processor)
    print(f"[train] saved final checkpoint to {final_dir}")


def maybe_wrap_lora(model, args: argparse.Namespace):
    if not args.use_lora:
        return model
    if get_peft_model is None:
        raise ImportError("peft is not installed but --use_lora was requested.")

    # Gemma 4 wraps quantized linears inside Gemma4ClippableLinear, so LoRA
    # has to target the inner `.linear` modules instead of the wrapper.
    # On a single A6000 48GB we keep the first stable path narrow and target
    # attention projections only; this trims memory compared with adding MLP
    # adapters during the initial VM runs.
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def load_tokenizer_and_chat_template_handler(model_name_or_path: str):
    processor = None
    chat_template_handler = None
    try:
        processor = AutoProcessor.from_pretrained(model_name_or_path)
        if hasattr(processor, "apply_chat_template"):
            chat_template_handler = processor
    except Exception:
        processor = None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if chat_template_handler is None:
        chat_template_handler = tokenizer

    return tokenizer, chat_template_handler, processor


def load_trainable_model(
    model_name_or_path: str,
    load_in_4bit: bool = False,
    gradient_checkpointing: bool = False,
):
    load_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["device_map"] = "auto"
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


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
    with (output_dir / "train_args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    base_dataset = build_datasets(args)

    prompt_builder = GemmaMERG.__new__(GemmaMERG)
    prompt_builder.model_name_or_path = args.model_name_or_path

    if args.dump_example_prompts:
        dump_example_prompts(base_dataset, prompt_builder, output_dir)
        print(f"Saved example prompts to {output_dir / 'example_prompts.json'}")
        return

    tokenizer, chat_template_handler, processor = load_tokenizer_and_chat_template_handler(args.model_name_or_path)
    prompt_builder.tokenizer = tokenizer
    prompt_builder.chat_template_handler = chat_template_handler

    train_dataset = SupervisedTask1Dataset(
        base_dataset=base_dataset,
        tokenizer=tokenizer,
        chat_template_handler=chat_template_handler,
        prompt_builder=prompt_builder,
        max_length=args.max_length,
        max_response_tokens=args.max_response_tokens,
    )
    collator = SFTDataCollator(tokenizer)

    model = load_trainable_model(
        args.model_name_or_path,
        load_in_4bit=args.load_in_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model.to("mps")
        else:
            model.to("cpu")
    model = maybe_wrap_lora(model, args)

    run_manual_training_loop(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        train_dataset=train_dataset,
        collator=collator,
        args=args,
        output_dir=output_dir,
    )


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import AvaMERGDataset, ESConvDataset  # noqa: E402
from src.models.gemma_merg import GemmaMERG  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Task 1 predictions from a base model or LoRA adapter.")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, choices=["avamerg", "esconv", "mentalchat16k"], required=True)
    parser.add_argument("--avamerg_root", type=str, default=None)
    parser.add_argument("--avamerg_split", type=str, default="valid")
    parser.add_argument("--avamerg_text_only", action="store_true")
    parser.add_argument("--esconv_json", type=str, default=None)
    parser.add_argument("--mentalchat16k_name", type=str, default="ShenLab/MentalChat16K")
    parser.add_argument("--mentalchat16k_split", type=str, default="train")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--include_prompt", action="store_true")
    return parser


def load_dataset_from_args(args: argparse.Namespace):
    if args.dataset == "avamerg":
        if not args.avamerg_root:
            raise ValueError("--avamerg_root is required when --dataset avamerg")
        return AvaMERGDataset(args.avamerg_root, split=args.avamerg_split, use_multimodal=not args.avamerg_text_only)
    if args.dataset == "esconv":
        if not args.esconv_json:
            raise ValueError("--esconv_json is required when --dataset esconv")
        return ESConvDataset(args.esconv_json)
    from src.data.mentalchat16k_dataset import MentalChat16KDataset  # noqa: E402

    return MentalChat16KDataset(
        dataset_name=args.mentalchat16k_name,
        split=args.mentalchat16k_split,
        max_samples=args.max_samples,
    )


def iter_samples(dataset, max_samples: int) -> Iterable[Dict[str, Any]]:
    for idx in range(min(max_samples, len(dataset))):
        yield dataset[idx]


def main() -> None:
    args = build_parser().parse_args()
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset_from_args(args)
    generator = GemmaMERG(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
    )

    with output_path.open("w", encoding="utf-8") as f:
        for sample in tqdm(iter_samples(dataset, args.max_samples), total=min(args.max_samples, len(dataset))):
            prediction = generator.generate_response(
                sample,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            row = {
                "sample_id": sample.get("sample_id"),
                "source_dataset": sample.get("source_dataset"),
                "history": sample.get("history"),
                "context": sample.get("context"),
                "reference_response": sample.get("response"),
                "prediction": prediction,
                "emotion": sample.get("emotion"),
                "support_strategy": sample.get("support_strategy"),
                "metadata": sample.get("metadata", {}),
            }
            if args.include_prompt:
                row["user_prompt"] = generator.build_user_prompt(sample)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()

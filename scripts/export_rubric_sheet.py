from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = [
    "sample_id",
    "source_dataset",
    "emotion",
    "support_strategy",
    "reference_response",
    "prediction",
    "empathy_score",
    "safety_score",
    "helpfulness_score",
    "relevance_score",
    "naturalness_score",
    "overall_score",
    "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a rubric CSV from prediction jsonl.")
    parser.add_argument("--predictions_jsonl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.predictions_jsonl, "r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=FIELDS)
        writer.writeheader()
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            writer.writerow(
                {
                    "sample_id": row.get("sample_id"),
                    "source_dataset": row.get("source_dataset"),
                    "emotion": row.get("emotion"),
                    "support_strategy": row.get("support_strategy"),
                    "reference_response": row.get("reference_response"),
                    "prediction": row.get("prediction"),
                    "empathy_score": "",
                    "safety_score": "",
                    "helpfulness_score": "",
                    "relevance_score": "",
                    "naturalness_score": "",
                    "overall_score": "",
                    "notes": "",
                }
            )
    print(f"Saved rubric sheet to {output_path}")


if __name__ == "__main__":
    main()

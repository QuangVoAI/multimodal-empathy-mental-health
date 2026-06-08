from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


SCORE_FIELDS = [
    "empathy_score",
    "safety_score",
    "helpfulness_score",
    "relevance_score",
    "naturalness_score",
    "overall_score",
]


def parse_score(value: str):
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a filled rubric CSV into JSON + Markdown.")
    parser.add_argument("--rubric_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.rubric_csv, "r", encoding="utf-8")))
    summary = {"num_rows": len(rows), "num_scored_rows": 0, "metrics": {}}

    valid_rows = 0
    for field in SCORE_FIELDS:
        values = [parse_score(row.get(field, "")) for row in rows]
        values = [v for v in values if v is not None]
        if values:
            summary["metrics"][field] = {"mean": round(mean(values), 4), "count": len(values)}
            valid_rows = max(valid_rows, len(values))
    summary["num_scored_rows"] = valid_rows

    (output_dir / "results_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Results Summary",
        "",
        f"- Total rows: {summary['num_rows']}",
        f"- Scored rows: {summary['num_scored_rows']}",
        "",
        "| Metric | Mean | Count |",
        "|---|---:|---:|",
    ]
    for field, payload in summary["metrics"].items():
        lines.append(f"| {field} | {payload['mean']:.4f} | {payload['count']} |")

    (output_dir / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summaries to {output_dir}")


if __name__ == "__main__":
    main()

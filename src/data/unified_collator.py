from __future__ import annotations

from typing import Any, Dict, List


class UnifiedCollator:
    """
    Minimal collator for mixed AvaMERG + ESConv training.

    The collator keeps raw fields intact and groups them into batch-level lists
    so prompt construction can happen inside the model or trainer.
    """

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "sample_ids": [item["sample_id"] for item in batch],
            "source_datasets": [item["source_dataset"] for item in batch],
            "task_types": [item["task_type"] for item in batch],
            "histories": [item["history"] for item in batch],
            "responses": [item["response"] for item in batch],
            "contexts": [item["context"] for item in batch],
            "emotions": [item.get("emotion") for item in batch],
            "support_strategies": [item.get("support_strategy") for item in batch],
            "coes": [item.get("coe", {}) for item in batch],
            "audio_paths": [item.get("audio_paths", []) for item in batch],
            "video_paths": [item.get("video_paths", []) for item in batch],
            "metadata": [item.get("metadata", {}) for item in batch],
        }


from __future__ import annotations

from typing import Any, Dict, List, Optional

from datasets import load_dataset
from torch.utils.data import Dataset


class MentalChat16KDataset(Dataset):
    """
    Lightweight inference/eval adapter for MentalChat16K.

    The schema on the Hub may evolve, so this loader uses best-effort heuristics:
    - if a row already contains a multi-turn conversation, keep it
    - otherwise wrap the user-side text as a single-turn history
    - keep a best-effort reference response for later evaluation
    """

    def __init__(
        self,
        dataset_name: str = "ShenLab/MentalChat16K",
        split: str = "train",
        max_samples: Optional[int] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        raw = load_dataset(dataset_name, split=split)
        if max_samples is not None:
            raw = raw.select(range(min(max_samples, len(raw))))
        self.samples = [self._normalize_row(dict(row), idx) for idx, row in enumerate(raw)]

    @staticmethod
    def _pick_first(row: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        for key in keys:
            value = row.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    @staticmethod
    def _normalize_history_from_messages(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, str]], str]:
        history: List[Dict[str, str]] = []
        reference = ""
        for msg in messages:
            role = msg.get("role") or msg.get("speaker") or "user"
            content = msg.get("content") or msg.get("text") or msg.get("utterance") or ""
            role = "assistant" if role in {"assistant", "supporter", "bot"} else "user"
            if role == "assistant":
                reference = str(content).strip()
            else:
                history.append({"role": role, "utterance": str(content).strip()})
        return history, reference

    def _normalize_row(self, row: Dict[str, Any], idx: int) -> Dict[str, Any]:
        messages = self._pick_first(row, ["messages", "dialogue", "conversation", "history"])
        history: List[Dict[str, str]] = []
        reference_response = ""
        if isinstance(messages, list) and messages and isinstance(messages[0], dict):
            history, reference_response = self._normalize_history_from_messages(messages)
        else:
            user_text = self._pick_first(
                row,
                ["question", "query", "prompt", "input", "instruction", "context", "situation"],
            ) or ""
            history = [{"role": "user", "utterance": str(user_text).strip()}]
            reference_response = str(
                self._pick_first(row, ["response", "answer", "output", "assistant", "target"]) or ""
            ).strip()

        context = str(self._pick_first(row, ["context", "situation", "background", "scenario"]) or "").strip()
        emotion = self._pick_first(row, ["emotion", "emotion_type", "speaker_emotion"])
        strategy = self._pick_first(row, ["strategy", "support_strategy", "response_strategy"])
        sample_id = str(self._pick_first(row, ["id", "sample_id", "uid"]) or f"mentalchat16k_{idx}")

        return {
            "sample_id": sample_id,
            "source_dataset": "mentalchat16k",
            "task_type": "text_only",
            "history": history,
            "response": reference_response,
            "context": context,
            "emotion": str(emotion).strip() if emotion else None,
            "support_strategy": str(strategy).strip() if strategy else None,
            "coe": {
                "event_scenario": context or None,
                "speaker_emotion": str(emotion).strip() if emotion else None,
                "emotion_cause": context or None,
                "goal_to_response": str(strategy).strip() if strategy else None,
            },
            "audio_paths": [],
            "video_paths": [],
            "metadata": {
                "raw_keys": sorted(row.keys()),
                "split": self.split,
            },
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

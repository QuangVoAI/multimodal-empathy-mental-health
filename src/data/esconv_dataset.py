from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset


@dataclass
class UnifiedSample:
    sample_id: str
    source_dataset: str
    task_type: str
    history: List[Dict[str, str]]
    response: str
    context: str
    emotion: Optional[str]
    support_strategy: Optional[str]
    coe: Dict[str, Optional[str]]
    audio_paths: List[str]
    video_paths: List[str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source_dataset": self.source_dataset,
            "task_type": self.task_type,
            "history": self.history,
            "response": self.response,
            "context": self.context,
            "emotion": self.emotion,
            "support_strategy": self.support_strategy,
            "coe": self.coe,
            "audio_paths": self.audio_paths,
            "video_paths": self.video_paths,
            "metadata": self.metadata,
        }


class ESConvDataset(Dataset):
    """
    Flexible ESConv normalizer.

    The public ESConv files are not always packaged in one exact JSON schema,
    so this scaffold uses light heuristics and keeps the normalization logic
    easy to edit after the real files are downloaded.
    """

    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        if not self.json_path.exists():
            raise FileNotFoundError(f"ESConv file not found: {self.json_path}")
        self.samples = self._load_samples()

    def _load_raw(self) -> Any:
        with self.json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _iter_dialogues(self, raw_data: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_data, list):
            return raw_data
        if isinstance(raw_data, dict):
            for key in ("data", "dialogs", "dialogues", "train", "valid", "test"):
                if key in raw_data and isinstance(raw_data[key], list):
                    return raw_data[key]
        raise ValueError("Unsupported ESConv JSON structure. Please adapt _iter_dialogues.")

    @staticmethod
    def _normalize_role(raw_role: Optional[str], index: int) -> str:
        if raw_role == "seeker":
            return "user"
        if raw_role == "supporter":
            return "assistant"
        if raw_role in {"user", "assistant"}:
            return raw_role
        return "user" if index % 2 == 0 else "assistant"

    def _normalize_history(self, dialogue: Dict[str, Any]) -> List[Dict[str, str]]:
        turns = (
            dialogue.get("dialog")
            or dialogue.get("dialogue")
            or dialogue.get("history")
            or []
        )
        normalized: List[Dict[str, str]] = []
        for index, turn in enumerate(turns):
            if isinstance(turn, dict):
                utterance = (
                    turn.get("text")
                    or turn.get("utterance")
                    or turn.get("content")
                    or ""
                ).strip()
                role = self._normalize_role(turn.get("speaker") or turn.get("role"), index)
            else:
                utterance = str(turn).strip()
                role = self._normalize_role(None, index)
            normalized.append({"role": role, "utterance": utterance})
        return normalized

    def _extract_context(self, dialogue: Dict[str, Any]) -> str:
        for key in ("situation", "context", "problem", "description"):
            if key in dialogue and dialogue[key]:
                return str(dialogue[key]).strip()
        return ""

    def _extract_strategy(self, dialogue: Dict[str, Any]) -> Optional[str]:
        for key in ("strategy", "support_strategy", "strategy_label", "response_strategy"):
            if key in dialogue and dialogue[key]:
                return str(dialogue[key]).strip()
        return None

    @staticmethod
    def _extract_turn_strategy(turn: Dict[str, Any]) -> Optional[str]:
        annotation = turn.get("annotation", {}) if isinstance(turn, dict) else {}
        strategy = annotation.get("strategy")
        if strategy:
            return str(strategy).strip()
        return None

    def _build_turn_level_samples(self, dialogue: Dict[str, Any], dialogue_idx: int) -> List[UnifiedSample]:
        raw_turns = dialogue.get("dialog") or dialogue.get("dialogue") or dialogue.get("history") or []
        normalized_history = self._normalize_history(dialogue)
        context = self._extract_context(dialogue)
        emotion = dialogue.get("emotion_type") or dialogue.get("emotion")
        problem_type = dialogue.get("problem_type")
        experience_type = dialogue.get("experience_type")

        samples: List[UnifiedSample] = []
        for turn_index, raw_turn in enumerate(raw_turns):
            if not isinstance(raw_turn, dict):
                continue

            role = self._normalize_role(raw_turn.get("speaker") or raw_turn.get("role"), turn_index)
            if role != "assistant":
                continue

            response = (
                raw_turn.get("text")
                or raw_turn.get("utterance")
                or raw_turn.get("content")
                or ""
            ).strip()
            if not response:
                continue

            history = normalized_history[:turn_index]
            if not history:
                continue

            strategy = self._extract_turn_strategy(raw_turn)
            sample_id = f"esconv_{dialogue_idx}_turn{turn_index}"
            samples.append(
                UnifiedSample(
                    sample_id=sample_id,
                    source_dataset="esconv",
                    task_type="text_only",
                    history=history,
                    response=response,
                    context=context,
                    emotion=emotion,
                    support_strategy=strategy,
                    coe={
                        "event_scenario": problem_type,
                        "speaker_emotion": emotion,
                        "emotion_cause": context,
                        "goal_to_response": strategy,
                    },
                    audio_paths=[],
                    video_paths=[],
                    metadata={
                        "dialogue_index": dialogue_idx,
                        "turn_index": turn_index,
                        "problem_type": problem_type,
                        "experience_type": experience_type,
                        "raw_keys": sorted(dialogue.keys()),
                    },
                )
            )
        return samples

    def _load_samples(self) -> List[UnifiedSample]:
        raw_data = self._load_raw()
        dialogues = self._iter_dialogues(raw_data)
        samples: List[UnifiedSample] = []

        for idx, dialogue in enumerate(dialogues):
            samples.extend(self._build_turn_level_samples(dialogue, idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx].to_dict()

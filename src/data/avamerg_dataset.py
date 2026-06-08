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


class AvaMERGDataset(Dataset):
    """
    Normalizes AvaMERG samples into the unified schema used by Task 1.

    Expected structure:
    - root/
      - train.json | valid.json | test.json
      - train/audio/*.wav
      - train/video/*.mp4
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        use_multimodal: bool = True,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.use_multimodal = use_multimodal
        self.samples = self._load_samples()

    def _load_samples(self) -> List[UnifiedSample]:
        json_path = self.root / f"{self.split}.json"
        if not json_path.exists():
            raise FileNotFoundError(f"AvaMERG split file not found: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            raw_items = json.load(f)

        normalized: List[UnifiedSample] = []
        for item in raw_items:
            normalized.append(self._normalize_item(item))
        return normalized

    @staticmethod
    def _transform_conv_id(conversation_id: str) -> str:
        return conversation_id.lstrip("0") or "0"

    def _normalize_history(self, dialogue_history: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized_history: List[Dict[str, str]] = []
        for turn_index, turn in enumerate(dialogue_history):
            utterance = turn.get("utterance", "").strip()
            role = turn.get("role")
            if role not in {"user", "assistant"}:
                role = "user" if turn_index % 2 == 0 else "assistant"
            normalized_history.append({"role": role, "utterance": utterance})
        return normalized_history

    def _build_media_paths(self, dia_id: str, num_turns: int) -> tuple[List[str], List[str]]:
        if not self.use_multimodal:
            return [], []

        audio_dir = self.root / self.split / "audio"
        video_dir = self.root / self.split / "video"
        audio_paths = [str(audio_dir / f"dia{dia_id}utt{i + 1}.wav") for i in range(num_turns)]
        video_paths = [str(video_dir / f"dia{dia_id}utt{i + 1}.mp4") for i in range(num_turns)]
        return audio_paths, video_paths

    def _normalize_item(self, item: Dict[str, Any]) -> UnifiedSample:
        turn = item["turns"][-1]
        conversation_id = item["conversation_id"]
        dia_id = self._transform_conv_id(conversation_id)
        history = self._normalize_history(turn.get("dialogue_history", []))
        num_turns = len(history)
        coe = turn.get("chain_of_empathy", {}) or {}
        audio_paths, video_paths = self._build_media_paths(dia_id, num_turns)

        context = turn.get("context")
        if context is None:
            context = ""
        response = turn.get("response")
        if response is None:
            response = ""

        return UnifiedSample(
            sample_id=f"avamerg_{conversation_id}",
            source_dataset="avamerg",
            task_type="multimodal" if self.use_multimodal else "text_only",
            history=history,
            response=str(response).strip(),
            context=str(context).strip(),
            emotion=coe.get("speaker_emotion"),
            support_strategy=None,
            coe={
                "event_scenario": coe.get("event_scenario"),
                "speaker_emotion": coe.get("speaker_emotion"),
                "emotion_cause": coe.get("emotion_cause"),
                "goal_to_response": coe.get("goal_to_response"),
            },
            audio_paths=audio_paths,
            video_paths=video_paths,
            metadata={
                "conversation_id": conversation_id,
                "topic": item.get("topic"),
                "speaker_profile": item.get("speaker_profile", {}),
                "listener_profile": item.get("listener_profile", {}),
            },
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx].to_dict()

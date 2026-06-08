from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from peft import PeftModel
except ImportError:  # pragma: no cover
    PeftModel = None


DEFAULT_SYSTEM_PROMPT = """You are a supportive, emotionally aware assistant.
Your job is to respond with empathy, emotional sensitivity, and practical care.

If the user sounds distressed, overwhelmed, hopeless, or unsafe:
- acknowledge the emotion clearly
- avoid judgment or minimizing language
- avoid overconfident clinical advice
- gently encourage appropriate support when needed

Keep the response warm, grounded, and concise.
Output only the final response to the user."""


@dataclass
class PromptParts:
    history_block: str
    context_block: str
    audio_summary: str
    video_summary: str
    support_strategy: str
    coe_block: str


class GemmaMERG:
    """
    Task-1-oriented scaffold for Gemma 4 26B.

    This class does not yet implement multimodal embedding injection.
    Instead, it provides:
    - unified prompt construction
    - chat template formatting
    - an insertion point for future audio/video encoders
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.load_in_4bit = load_in_4bit

        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self._load_model()
        if self.device != "cuda":
            self.model.to(self.device)

    def _load_model(self):
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": self.torch_dtype,
        }
        if self.device == "cuda":
            load_kwargs["device_map"] = "auto"
            if self.load_in_4bit:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_name_or_path, **load_kwargs)
        except Exception:
            model = AutoModelForImageTextToText.from_pretrained(self.model_name_or_path, **load_kwargs)

        if self.adapter_path:
            if PeftModel is None:
                raise ImportError("peft is required to load a LoRA adapter checkpoint.")
            model = PeftModel.from_pretrained(model, self.adapter_path)
        return model

    @staticmethod
    def _format_history(history: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for turn in history:
            role = turn.get("role", "user").capitalize()
            utterance = turn.get("utterance", "").strip()
            lines.append(f"{role}: {utterance}")
        return "\n".join(lines).strip() or "N/A"

    @staticmethod
    def _format_coe(coe: Dict[str, Any]) -> str:
        useful = []
        if coe.get("speaker_emotion"):
            useful.append(f"- Emotion: {coe['speaker_emotion']}")
        if coe.get("emotion_cause"):
            useful.append(f"- Likely cause: {coe['emotion_cause']}")
        if coe.get("goal_to_response"):
            useful.append(f"- Response goal: {coe['goal_to_response']}")
        return "\n".join(useful) if useful else "N/A"

    @staticmethod
    def _summarize_modalities(
        audio_paths: List[str],
        video_paths: List[str],
        audio_summary: Optional[str] = None,
        video_summary: Optional[str] = None,
    ) -> tuple[str, str]:
        # Placeholder for future encoder integration.
        if audio_summary is None:
            audio_summary = "N/A" if not audio_paths else "Audio available but not yet summarized."
        if video_summary is None:
            video_summary = "N/A" if not video_paths else "Video available but not yet summarized."
        return audio_summary, video_summary

    def build_prompt_parts(self, sample: Dict[str, Any]) -> PromptParts:
        audio_summary, video_summary = self._summarize_modalities(
            sample.get("audio_paths", []),
            sample.get("video_paths", []),
            sample.get("audio_summary"),
            sample.get("video_summary"),
        )
        return PromptParts(
            history_block=self._format_history(sample["history"]),
            context_block=sample.get("context", "").strip() or "N/A",
            audio_summary=audio_summary,
            video_summary=video_summary,
            support_strategy=sample.get("support_strategy") or "N/A",
            coe_block=self._format_coe(sample.get("coe", {})),
        )

    def build_user_prompt(self, sample: Dict[str, Any]) -> str:
        parts = self.build_prompt_parts(sample)
        return f"""Conversation history:
{parts.history_block}

User's current situation:
{parts.context_block}

Inferred emotional context:
{parts.coe_block}

Additional affective cues:
- Audio cues: {parts.audio_summary}
- Visual cues: {parts.video_summary}

Preferred support strategy:
{parts.support_strategy}

Write one empathetic, safe, and helpful response to the user.
Keep it natural, grounded, and not overly long.
Output only the response."""

    def build_messages(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": self.build_user_prompt(sample)},
        ]

    def prepare_inputs(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        messages = self.build_messages(sample)
        tokenized = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        return {k: v.to(self.model.device) for k, v in tokenized.items()}

    @torch.no_grad()
    def generate_response(
        self,
        sample: Dict[str, Any],
        max_new_tokens: int = 160,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        model_inputs = self.prepare_inputs(sample)
        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        generated = output_ids[0][model_inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

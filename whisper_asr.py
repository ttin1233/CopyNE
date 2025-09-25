"""Utilities to run Whisper-based ASR inside CopyNE."""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch
import whisper

from utils.process import read_context_table


class WhisperASRParser:
    """Lightweight wrapper around the Whisper transcription model.

    The original project relies on WeNet for acoustic modeling.  This wrapper
    mimics the minimal interface that ``CopyNEASRParser`` exposes for the API
    demo so that the Gradio front-end can remain unchanged while delegating the
    actual speech recognition to Whisper.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.device = self._resolve_device(getattr(args, "device", "-1"))
        model_size = getattr(args, "whisper_model", "base")
        self.model = whisper.load_model(model_size, device=self.device)

    def _resolve_device(self, device_arg: str) -> str:
        """Return a device string understood by ``whisper.load_model``."""

        if device_arg is None or device_arg == "-1":
            return "cpu"
        if torch.cuda.is_available():
            # ``device_arg`` can be ``0`` or ``cuda:0``.  Normalize both cases.
            if device_arg.isdigit():
                return f"cuda:{device_arg}"
            return device_arg
        # GPU requested but not available – fall back to CPU gracefully.
        return "cpu"

    def _build_prompt(self, ne_vocab_file: Optional[str]) -> Optional[str]:
        """Convert the optional NE vocabulary file into a Whisper prompt."""

        if not ne_vocab_file or ne_vocab_file == "None":
            return None
        if not os.path.exists(ne_vocab_file):
            return None

        vocab: Dict[str, int] = read_context_table(ne_vocab_file)
        # ``read_context_table`` returns a dictionary ``token -> index``.  We
        # restore the original order so that the prompt is deterministic.
        ordered_tokens = [""] * len(vocab)
        for token, idx in vocab.items():
            if 0 <= idx < len(ordered_tokens):
                ordered_tokens[idx] = token
            else:
                ordered_tokens.append(token)
        prompt_tokens = [tok for tok in ordered_tokens if tok]
        if not prompt_tokens:
            return None
        return " ".join(prompt_tokens)

    def api(self, audio_file: str, ne_vocab_file: str, copy_threshold: float = 0.9, tmp_dir: str = "tmp_dir") -> str:
        """Transcribe ``audio_file`` with Whisper.

        Parameters other than ``audio_file`` and ``ne_vocab_file`` are accepted
        for API compatibility with :class:`CopyNEASRParser`.  ``copy_threshold``
        is ignored because Whisper does not expose a copy mechanism.
        """

        if not audio_file:
            return ""

        prompt = self._build_prompt(ne_vocab_file)
        language = getattr(self.args, "whisper_language", None)
        task = getattr(self.args, "whisper_task", "transcribe")
        temperature = getattr(self.args, "whisper_temperature", None)

        transcribe_options = {
            "fp16": self.device.startswith("cuda") and torch.cuda.is_available(),
        }
        if language and language.lower() != "auto":
            transcribe_options["language"] = language
        if prompt:
            transcribe_options["initial_prompt"] = prompt
        if task:
            transcribe_options["task"] = task
        if temperature is not None:
            transcribe_options["temperature"] = temperature

        result = self.model.transcribe(audio_file, **transcribe_options)
        return result.get("text", "").strip()


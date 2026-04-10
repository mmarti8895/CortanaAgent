from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import webrtcvad  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    webrtcvad = None


@dataclass(slots=True)
class VoiceActivityDetector:
    sample_rate: int = 16000
    aggressiveness: int = 2
    _engine: Any = field(init=False, default=None)

    def __post_init__(self) -> None:
        self._engine = webrtcvad.Vad(self.aggressiveness) if webrtcvad else None

    def is_speech(self, frame: bytes) -> bool:
        if self._engine is not None:
            return bool(self._engine.is_speech(frame, self.sample_rate))
        pcm = np.frombuffer(frame, dtype=np.int16)
        energy = float(np.abs(pcm).mean()) if pcm.size else 0.0
        return energy > 220

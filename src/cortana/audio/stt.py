from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from queue import Queue
from typing import Protocol

import numpy as np

from cortana.audio.vad import VoiceActivityDetector

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover
    sd = None

try:
    from faster_whisper import WhisperModel  # type: ignore
except ImportError:  # pragma: no cover
    WhisperModel = None


class SpeechToText(Protocol):
    async def stream(self) -> AsyncIterator[str]: ...


@dataclass(slots=True)
class BufferedWhisperSTT:
    model_size: str
    device: str
    compute_type: str
    sample_rate: int
    chunk_seconds: float
    vad: VoiceActivityDetector = field(default_factory=VoiceActivityDetector)

    def __post_init__(self) -> None:
        if WhisperModel is None or sd is None:
            msg = "faster-whisper and sounddevice are required for BufferedWhisperSTT"
            raise RuntimeError(msg)
        self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)

    async def stream(self) -> AsyncIterator[str]:
        audio_queue: Queue[np.ndarray] = Queue()
        chunk_frames = int(self.sample_rate * self.chunk_seconds)

        def callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
            del frames, time
            if status:
                return
            audio_queue.put(indata.copy().reshape(-1))

        stream = sd.InputStream(channels=1, samplerate=self.sample_rate, dtype="int16", callback=callback)
        stream.start()
        try:
            pending = np.array([], dtype=np.int16)
            while True:
                block = await asyncio.to_thread(audio_queue.get)
                pending = np.concatenate([pending, block.astype(np.int16)])
                if pending.size < chunk_frames:
                    continue

                chunk = pending[:chunk_frames]
                pending = pending[chunk_frames:]
                if not self.vad.is_speech(chunk.tobytes()):
                    continue

                float_audio = chunk.astype(np.float32) / 32768.0
                segments, _ = await asyncio.to_thread(
                    self._model.transcribe,
                    float_audio,
                    language="en",
                    vad_filter=True,
                    beam_size=1,
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    yield text
        finally:
            stream.stop()
            stream.close()


@dataclass(slots=True)
class TextInputSTT:
    prompt: str = "You> "

    async def stream(self) -> AsyncIterator[str]:
        while True:
            text = await asyncio.to_thread(input, self.prompt)
            cleaned = text.strip()
            if cleaned:
                yield cleaned

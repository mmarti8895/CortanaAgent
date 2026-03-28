from __future__ import annotations

import asyncio
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cortana.avatar.events import AvatarEmitter

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover
    sd = None


class TextToSpeechError(RuntimeError):
    pass


def _pcm16_rms(chunk: bytes) -> float:
    frame_bytes = len(chunk) - (len(chunk) % 2)
    if frame_bytes == 0:
        return 0.0

    samples = np.frombuffer(chunk[:frame_bytes], dtype=np.int16).astype(np.float32)
    if samples.size == 0:
        return 0.0

    return min(float(np.sqrt(np.mean(np.square(samples))) / 32768.0), 1.0)


@dataclass(slots=True)
class PiperTTS:
    executable: str
    model_path: str
    config_path: str | None
    avatar: AvatarEmitter

    def __post_init__(self) -> None:
        if sd is None:
            raise TextToSpeechError("sounddevice is required for PiperTTS")

        model = Path(self.model_path)
        if not model.is_file():
            msg = f"Piper model not found: {model}"
            raise TextToSpeechError(msg)

        if self.config_path is not None:
            config = Path(self.config_path)
            if not config.is_file():
                msg = f"Piper config not found: {config}"
                raise TextToSpeechError(msg)

        executable_path = Path(self.executable)
        if executable_path.is_absolute():
            if not executable_path.is_file():
                msg = f"Piper executable not found: {executable_path}"
                raise TextToSpeechError(msg)
        else:
            resolved = shutil.which(self.executable)
            if resolved is None:
                msg = f"Piper executable is not on PATH: {self.executable}"
                raise TextToSpeechError(msg)
            self.executable = resolved

    async def speak_stream(self, text_stream: AsyncIterator[str]) -> str:
        proc_args = [self.executable, "--model", self.model_path, "--output-raw"]
        if self.config_path:
            proc_args.extend(["--config", self.config_path])

        try:
            proc = await asyncio.create_subprocess_exec(
                *proc_args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError:
            console = ConsoleTTS(avatar=self.avatar)
            return await console.speak_stream(text_stream)

        await self.avatar.speaking_state(True)
        assembled = []
        try:
            async for part in text_stream:
                assembled.append(part)
                assert proc.stdin is not None
                proc.stdin.write(part.encode("utf-8"))
                await proc.stdin.drain()
            assert proc.stdin is not None
            proc.stdin.write(b"\n")
            await proc.stdin.drain()
            proc.stdin.close()

            assert proc.stdout is not None
            with sd.RawOutputStream(samplerate=22050, channels=1, dtype="int16") as stream:
                while True:
                    chunk = await proc.stdout.read(2048)
                    if not chunk:
                        break
                    rms = _pcm16_rms(chunk)
                    await self.avatar.jaw_movement(rms)
                    stream.write(chunk)
        finally:
            await self.avatar.speaking_state(False)
            await proc.wait()

        return "".join(assembled)


@dataclass(slots=True)
class ConsoleTTS:
    avatar: AvatarEmitter

    async def speak_stream(self, text_stream: AsyncIterator[str]) -> str:
        await self.avatar.speaking_state(True)
        full = []
        async for token in text_stream:
            print(token, end="", flush=True)
            full.append(token)
            await self.avatar.jaw_movement(min(len(token) / 12, 1.0))
        print()
        await self.avatar.speaking_state(False)
        return "".join(full)

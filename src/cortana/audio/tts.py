from __future__ import annotations

import asyncio
import audioop
from collections.abc import AsyncIterator
from dataclasses import dataclass

from cortana.avatar.events import AvatarEmitter

try:
    import sounddevice as sd  # type: ignore
except ImportError:  # pragma: no cover
    sd = None


class TextToSpeechError(RuntimeError):
    pass


@dataclass(slots=True)
class PiperTTS:
    executable: str
    model_path: str
    config_path: str | None
    avatar: AvatarEmitter

    async def speak_stream(self, text_stream: AsyncIterator[str]) -> str:
        if sd is None:
            raise TextToSpeechError("sounddevice is required for PiperTTS")

        proc_args = [self.executable, "--model", self.model_path, "--output-raw"]
        if self.config_path:
            proc_args.extend(["--config", self.config_path])

        proc = await asyncio.create_subprocess_exec(
            *proc_args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

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
                    rms = audioop.rms(chunk, 2) / 32768.0
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

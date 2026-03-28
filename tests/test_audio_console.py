from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from cortana.audio import tts as tts_module
from cortana.audio.stt import TextInputSTT
from cortana.audio.tts import ConsoleTTS, PiperTTS, TextToSpeechError


class DummyAvatar:
    def __init__(self) -> None:
        self.events: list[tuple[str, float | bool]] = []

    async def speaking_state(self, speaking: bool) -> None:
        self.events.append(("s", speaking))

    async def jaw_movement(self, amplitude: float) -> None:
        self.events.append(("j", amplitude))


async def token_stream() -> AsyncIterator[str]:
    yield "hello"
    yield " world"


@pytest.mark.asyncio
async def test_console_tts(monkeypatch) -> None:
    avatar = DummyAvatar()
    tts = ConsoleTTS(avatar=avatar)  # type: ignore[arg-type]
    text = await tts.speak_stream(token_stream())
    assert text == "hello world"
    assert avatar.events[0] == ("s", True)
    assert avatar.events[-1] == ("s", False)


@pytest.mark.asyncio
async def test_text_input_stt(monkeypatch) -> None:
    values = iter(["", "Cortana status"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(values))
    stt = TextInputSTT(prompt="> ")
    stream = stt.stream()
    first = await anext(stream)
    assert first == "Cortana status"


def test_piper_tts_requires_executable_and_model(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_module, "sd", object())
    monkeypatch.setattr(tts_module.shutil, "which", lambda name: None)

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"model")

    with pytest.raises(TextToSpeechError, match="not on PATH"):
        PiperTTS(
            executable="piper",
            model_path=str(model),
            config_path=None,
            avatar=DummyAvatar(),  # type: ignore[arg-type]
        )

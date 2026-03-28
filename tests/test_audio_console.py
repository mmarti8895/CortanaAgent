from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from cortana.audio.stt import TextInputSTT
from cortana.audio.tts import ConsoleTTS


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

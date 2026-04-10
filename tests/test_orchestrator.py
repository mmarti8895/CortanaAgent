from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from cortana.audio.wakeword import WakeWordDetector
from cortana.core.models import AssistantState, ConversationMemory
from cortana.core.orchestrator import Orchestrator
from cortana.plugins.manager import PluginManager


class FakeStt:
    def __init__(self, phrases: list[str]) -> None:
        self.phrases = phrases

    async def stream(self) -> AsyncIterator[str]:
        for phrase in self.phrases:
            yield phrase


class FakeTts:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak_stream(self, text_stream: AsyncIterator[str]) -> str:
        parts = [part async for part in text_stream]
        joined = "".join(parts)
        self.spoken.append(joined)
        return joined


class FakeLlm:
    async def stream_reply(self, memory: ConversationMemory, user_text: str) -> AsyncIterator[str]:
        del memory
        yield f"Echo: {user_text}"


@pytest.mark.asyncio
async def test_orchestrator_requires_wake_word() -> None:
    stt = FakeStt(["hello", "cortana", "what time is it"])
    tts = FakeTts()
    orchestrator = Orchestrator(
        stt=stt,
        tts=tts,
        llm=FakeLlm(),
        wakeword=WakeWordDetector("cortana"),
        plugins=PluginManager(),
        memory=ConversationMemory(max_turns=6),
    )

    await orchestrator.run()

    assert tts.spoken[0] == "Yes?"
    assert tts.spoken[1].startswith("Echo:")
    assert orchestrator.state == AssistantState.IDLE


@pytest.mark.asyncio
async def test_orchestrator_plugin_short_circuit() -> None:
    class StaticPlugin:
        name = "static"

        def matches(self, text: str) -> bool:
            return "ping" in text

        async def handle(self, ctx):
            del ctx
            return "pong"

    stt = FakeStt(["cortana", "ping"])
    tts = FakeTts()
    manager = PluginManager([StaticPlugin()])
    orchestrator = Orchestrator(
        stt=stt,
        tts=tts,
        llm=FakeLlm(),
        wakeword=WakeWordDetector("cortana"),
        plugins=manager,
        memory=ConversationMemory(max_turns=6),
    )

    await orchestrator.run()
    assert "pong" in tts.spoken[-1]

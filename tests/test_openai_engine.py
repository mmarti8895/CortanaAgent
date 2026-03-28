from __future__ import annotations

import pytest
from openai import APIConnectionError

from cortana.core.models import ConversationMemory
from cortana.llm.engine import HonorificRotator, LocalFallbackEngine, OpenAiLlmEngine


class FakeChatCompletions:
    async def create(self, **kwargs):
        del kwargs
        raise APIConnectionError(request=None)


class FakeClient:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {"completions": FakeChatCompletions()})()


@pytest.mark.asyncio
async def test_openai_engine_uses_fallback_on_connection_error() -> None:
    fallback = LocalFallbackEngine(honorifics=HonorificRotator(names=["Commander"]))
    engine = OpenAiLlmEngine(
        api_key="dummy",
        model="x",
        temperature=0,
        timeout_seconds=1,
        personality="p",
        fallback=fallback,
    )
    engine._client = FakeClient()  # type: ignore[assignment]

    memory = ConversationMemory(max_turns=2)
    output = "".join([chunk async for chunk in engine.stream_reply(memory, "status")])
    assert "Commander" in output

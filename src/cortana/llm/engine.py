from __future__ import annotations

import asyncio
import random
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from openai import APIConnectionError, AsyncOpenAI

from cortana.core.models import ConversationMemory


@dataclass(slots=True)
class HonorificRotator:
    names: list[str] = field(
        default_factory=lambda: [
            "Chief",
            "Master Chief",
            "Commander",
            "Spartan",
            "Kai",
            "Commander Kai",
            "Spartan Kai",
        ]
    )
    _pool: list[str] = field(default_factory=list)

    def next_name(self) -> str:
        if not self._pool:
            self._pool = self.names[:]
            random.shuffle(self._pool)
        return self._pool.pop()


@dataclass(slots=True)
class LocalFallbackEngine:
    honorifics: HonorificRotator

    async def stream_reply(self, memory: ConversationMemory, user_text: str) -> AsyncIterator[str]:
        del memory
        prefix = self.honorifics.next_name()
        response = f"{prefix}, I heard: '{user_text}'. API is unavailable, running local fallback reasoning."
        for token in response.split(" "):
            await asyncio.sleep(0.01)
            yield token + " "


@dataclass(slots=True)
class OpenAiLlmEngine:
    api_key: str
    model: str
    temperature: float
    timeout_seconds: float
    personality: str
    fallback: LocalFallbackEngine
    _client: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(api_key=self.api_key, timeout=self.timeout_seconds)

    async def stream_reply(self, memory: ConversationMemory, user_text: str) -> AsyncIterator[str]:
        system = (
            self.personality
            + " Keep responses under 120 words unless explicitly asked. "
            + "Use the requested military honorific naturally at times."
        )
        messages = memory.to_messages(system)
        messages.append({"role": "user", "content": user_text})
        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                stream=True,
            )
            async for event in stream:
                if not event.choices:
                    continue
                delta = event.choices[0].delta.content or ""
                if delta:
                    yield delta
        except APIConnectionError:
            async for token in self.fallback.stream_reply(memory, user_text):
                yield token

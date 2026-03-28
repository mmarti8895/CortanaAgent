from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from cortana.audio.wakeword import WakeWordDetector
from cortana.core.models import AssistantState, ConversationMemory
from cortana.plugins.manager import PluginManager
from cortana.utils.logging import get_logger

_LISTEN_TIMEOUT = object()


class SttLike(Protocol):
    async def stream(self) -> AsyncIterator[str]: ...


class TtsLike(Protocol):
    async def speak_stream(self, text_stream: AsyncIterator[str]) -> str: ...


class LlmLike(Protocol):
    async def stream_reply(self, memory: ConversationMemory, user_text: str) -> AsyncIterator[str]: ...


@dataclass(slots=True)
class Orchestrator:
    stt: SttLike
    tts: TtsLike
    llm: LlmLike
    wakeword: WakeWordDetector
    plugins: PluginManager
    memory: ConversationMemory
    listen_timeout_seconds: float = 2.5
    listen_max_phrases: int = 6
    state: AssistantState = AssistantState.IDLE
    logger: object = field(default_factory=lambda: get_logger("cortana.orchestrator"))

    async def run(self) -> None:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        producer = asyncio.create_task(self._pump_phrases(queue))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                await self._handle_phrase(queue, item)
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await producer

    async def _pump_phrases(self, queue: asyncio.Queue[str | None]) -> None:
        async for phrase in self.stt.stream():
            cleaned = phrase.strip()
            if cleaned:
                await queue.put(cleaned)
        await queue.put(None)

    async def _handle_phrase(self, queue: asyncio.Queue[str | None], text: str) -> None:
        self.logger.info("phrase.received", text=text, state=self.state)
        lower = text.lower()

        if self.state == AssistantState.IDLE:
            if not self.wakeword.is_wake(lower):
                return

            inline_request = self.wakeword.strip_wake(text)
            if inline_request:
                await self._respond(inline_request)
                return

            self.state = AssistantState.LISTENING
            await self._speak_once(_single_item("Yes?"), next_state=AssistantState.LISTENING)

            request = await self._collect_follow_up(queue)
            if request is None:
                self.logger.info("listen.timeout", timeout_seconds=self.listen_timeout_seconds)
                self.state = AssistantState.IDLE
                return

            await self._respond(request)
            return

        await self._respond(text)

    async def _collect_follow_up(self, queue: asyncio.Queue[str | None]) -> str | None:
        phrases: list[str] = []
        while len(phrases) < self.listen_max_phrases:
            item = await self._wait_for_phrase(queue, timeout=self.listen_timeout_seconds)
            if item is _LISTEN_TIMEOUT:
                break
            if item is None:
                await queue.put(None)
                break
            phrases.append(item)
        combined = " ".join(phrases).strip()
        return combined or None

    async def _wait_for_phrase(
        self,
        queue: asyncio.Queue[str | None],
        timeout: float,
    ) -> str | None | object:
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except TimeoutError:
            return _LISTEN_TIMEOUT

    async def _respond(self, text: str) -> None:
        self.state = AssistantState.THINKING
        plugin_reply = await self.plugins.dispatch(text)
        if plugin_reply is not None:
            await self._speak_once(_single_item(plugin_reply))
            self.memory.append("user", text)
            self.memory.append("assistant", plugin_reply)
            self.state = AssistantState.IDLE
            return

        self.memory.append("user", text)
        self.state = AssistantState.SPEAKING
        reply_stream = self.llm.stream_reply(self.memory, text)
        full = await self.tts.speak_stream(reply_stream)
        self.memory.append("assistant", full)
        self.state = AssistantState.IDLE

    async def _speak_once(
        self,
        text_stream: AsyncIterator[str],
        next_state: AssistantState = AssistantState.IDLE,
    ) -> None:
        self.state = AssistantState.SPEAKING
        await self.tts.speak_stream(text_stream)
        self.state = next_state


async def _single_item(text: str) -> AsyncIterator[str]:
    yield text

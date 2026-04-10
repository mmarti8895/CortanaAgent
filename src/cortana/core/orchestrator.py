from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from cortana.audio.wakeword import WakeWordDetector
from cortana.core.models import AssistantState, ConversationMemory
from cortana.plugins.manager import PluginManager
from cortana.utils.logging import get_logger


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
    state: AssistantState = AssistantState.IDLE
    logger: object = field(default_factory=lambda: get_logger("cortana.orchestrator"))

    async def run(self) -> None:
        async for phrase in self.stt.stream():
            await self._handle_phrase(phrase)

    async def _handle_phrase(self, text: str) -> None:
        self.logger.info("phrase.received", text=text, state=self.state)
        lower = text.lower()

        if self.state == AssistantState.IDLE:
            if self.wakeword.is_wake(lower):
                self.state = AssistantState.LISTENING
                await self._speak_once(_single_item("Yes?"))
            return

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

    async def _speak_once(self, text_stream: AsyncIterator[str]) -> None:
        previous_state = self.state
        self.state = AssistantState.SPEAKING
        try:
            await self.tts.speak_stream(text_stream)
        finally:
            self.state = previous_state


async def _single_item(text: str) -> AsyncIterator[str]:
    yield text

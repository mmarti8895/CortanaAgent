from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class AssistantState(StrEnum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(slots=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(slots=True)
class ConversationMemory:
    max_turns: int
    turns: list[ConversationTurn] = field(default_factory=list)

    def append(self, role: str, content: str) -> None:
        self.turns.append(ConversationTurn(role=role, content=content))
        overflow = len(self.turns) - self.max_turns
        if overflow > 0:
            self.turns = self.turns[overflow:]

    def to_messages(self, system_prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend({"role": turn.role, "content": turn.content} for turn in self.turns)
        return messages

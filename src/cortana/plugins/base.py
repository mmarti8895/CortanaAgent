from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class CommandContext:
    text: str


class CommandPlugin(Protocol):
    name: str

    def matches(self, text: str) -> bool: ...

    async def handle(self, ctx: CommandContext) -> str: ...

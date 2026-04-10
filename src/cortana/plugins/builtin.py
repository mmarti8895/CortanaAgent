from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass

from cortana.plugins.base import CommandContext


@dataclass(slots=True)
class TimeCommandPlugin:
    name: str = "time"

    def matches(self, text: str) -> bool:
        return bool(re.search(r"\b(time|clock)\b", text.lower()))

    async def handle(self, ctx: CommandContext) -> str:
        now = dt.datetime.now().strftime("%H:%M")
        return f"Current local time is {now}."


@dataclass(slots=True)
class GreetingPlugin:
    name: str = "greeting"

    def matches(self, text: str) -> bool:
        return bool(re.search(r"\b(hello|hi|hey)\b", text.lower()))

    async def handle(self, ctx: CommandContext) -> str:
        variants = ["Ready for orders.", "Standing by.", "Awaiting your command."]
        return random.choice(variants)

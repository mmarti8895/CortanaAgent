from __future__ import annotations

from dataclasses import dataclass, field

from cortana.plugins.base import CommandContext, CommandPlugin


@dataclass(slots=True)
class PluginManager:
    plugins: list[CommandPlugin] = field(default_factory=list)

    def register(self, plugin: CommandPlugin) -> None:
        self.plugins.append(plugin)

    async def dispatch(self, text: str) -> str | None:
        for plugin in self.plugins:
            if plugin.matches(text):
                return await plugin.handle(CommandContext(text=text))
        return None

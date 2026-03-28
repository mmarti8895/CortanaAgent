import pytest

from cortana.plugins.builtin import GreetingPlugin, TimeCommandPlugin
from cortana.plugins.manager import PluginManager


@pytest.mark.asyncio
async def test_plugin_dispatches_time() -> None:
    manager = PluginManager([TimeCommandPlugin()])
    reply = await manager.dispatch("what time is it")
    assert reply is not None
    assert "time" in reply.lower()


@pytest.mark.asyncio
async def test_plugin_none_when_no_match() -> None:
    manager = PluginManager([GreetingPlugin()])
    reply = await manager.dispatch("calculate pi")
    assert reply is None

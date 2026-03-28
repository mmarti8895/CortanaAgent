import pytest

from cortana.core.models import ConversationMemory
from cortana.llm.engine import HonorificRotator, LocalFallbackEngine


def test_honorific_rotates_without_loss() -> None:
    rotator = HonorificRotator()
    seen = {rotator.next_name() for _ in range(7)}
    assert len(seen) == 7


@pytest.mark.asyncio
async def test_local_fallback_stream() -> None:
    engine = LocalFallbackEngine(honorifics=HonorificRotator(names=["Chief"]))
    memory = ConversationMemory(max_turns=4)
    chunks = [chunk async for chunk in engine.stream_reply(memory, "status report")]
    text = "".join(chunks)
    assert "Chief" in text
    assert "status report" in text

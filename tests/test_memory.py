from cortana.core.models import ConversationMemory


def test_memory_caps_turns() -> None:
    memory = ConversationMemory(max_turns=2)
    memory.append("user", "one")
    memory.append("assistant", "two")
    memory.append("user", "three")
    assert [t.content for t in memory.turns] == ["two", "three"]


def test_memory_messages() -> None:
    memory = ConversationMemory(max_turns=4)
    memory.append("user", "hello")
    msgs = memory.to_messages("system prompt")
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "hello"

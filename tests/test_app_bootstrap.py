from __future__ import annotations

from cortana import app
from cortana.audio.stt import TextInputSTT
from cortana.audio.tts import ConsoleTTS
from cortana.llm.engine import LocalFallbackEngine


def test_build_orchestrator_fallbacks(monkeypatch) -> None:
    monkeypatch.setattr(app.settings, "openai_api_key", None)
    monkeypatch.setattr(app.settings, "tts_backend", "console")

    def fail_stt(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("no audio")

    monkeypatch.setattr(app, "BufferedWhisperSTT", fail_stt)

    orchestrator = app.build_orchestrator()

    assert isinstance(orchestrator.stt, TextInputSTT)
    assert isinstance(orchestrator.tts, ConsoleTTS)
    assert isinstance(orchestrator.llm, LocalFallbackEngine)

import numpy as np

from cortana.audio import vad as vad_module
from cortana.audio.vad import VoiceActivityDetector


def test_vad_energy_fallback_detects_speech() -> None:
    vad = VoiceActivityDetector(sample_rate=16000)
    loud = (np.ones(160, dtype=np.int16) * 1000).tobytes()
    assert vad.is_speech(loud)


def test_vad_energy_fallback_detects_silence() -> None:
    vad = VoiceActivityDetector(sample_rate=16000)
    silent = (np.zeros(160, dtype=np.int16)).tobytes()
    assert vad.is_speech(silent) is False


def test_vad_splits_long_frames_for_webrtc(monkeypatch) -> None:
    calls: list[int] = []

    class FakeVad:
        def __init__(self, aggressiveness: int) -> None:
            del aggressiveness

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            assert sample_rate == 16000
            calls.append(len(frame))
            return len(calls) == 2

    class FakeWebRtcVad:
        Error = RuntimeError
        Vad = FakeVad

    monkeypatch.setattr(vad_module, "webrtcvad", FakeWebRtcVad)

    vad = VoiceActivityDetector(sample_rate=16000)
    frame = (np.ones(16000, dtype=np.int16) * 1000).tobytes()

    assert vad.is_speech(frame)
    assert calls == [960, 960]


def test_vad_falls_back_to_energy_when_webrtc_rejects_frame(monkeypatch) -> None:
    class FakeVad:
        def __init__(self, aggressiveness: int) -> None:
            del aggressiveness

        def is_speech(self, frame: bytes, sample_rate: int) -> bool:
            del frame, sample_rate
            raise RuntimeError("bad frame")

    class FakeWebRtcVad:
        Error = RuntimeError
        Vad = FakeVad

    monkeypatch.setattr(vad_module, "webrtcvad", FakeWebRtcVad)

    vad = VoiceActivityDetector(sample_rate=16000)
    loud = (np.ones(160, dtype=np.int16) * 1000).tobytes()

    assert vad.is_speech(loud)

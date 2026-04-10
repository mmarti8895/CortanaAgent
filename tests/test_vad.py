import numpy as np

from cortana.audio.vad import VoiceActivityDetector


def test_vad_energy_fallback_detects_speech() -> None:
    vad = VoiceActivityDetector(sample_rate=16000)
    loud = (np.ones(160, dtype=np.int16) * 1000).tobytes()
    assert vad.is_speech(loud)


def test_vad_energy_fallback_detects_silence() -> None:
    vad = VoiceActivityDetector(sample_rate=16000)
    silent = (np.zeros(160, dtype=np.int16)).tobytes()
    assert vad.is_speech(silent) is False

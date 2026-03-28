import numpy as np

from cortana.audio.stt import SpeechSegmentAccumulator


class StubVad:
    def __init__(self, decisions: list[bool]) -> None:
        self._decisions = iter(decisions)

    def is_speech(self, frame: bytes) -> bool:
        del frame
        return next(self._decisions)


def test_segment_accumulator_waits_for_silence() -> None:
    accumulator = SpeechSegmentAccumulator(
        vad=StubVad([True, True, False, False]),  # type: ignore[arg-type]
        sample_rate=4,
        end_silence_seconds=1.0,
        max_phrase_seconds=10.0,
    )
    chunk_a = np.array([1, 1], dtype=np.int16)
    chunk_b = np.array([2, 2], dtype=np.int16)
    silence = np.zeros(2, dtype=np.int16)

    assert accumulator.push(chunk_a) is None
    assert accumulator.push(chunk_b) is None
    assert accumulator.push(silence) is None

    phrase = accumulator.push(silence)

    assert phrase is not None
    np.testing.assert_array_equal(phrase, np.array([1, 1, 2, 2], dtype=np.int16))


def test_segment_accumulator_flushes_when_phrase_gets_too_long() -> None:
    accumulator = SpeechSegmentAccumulator(
        vad=StubVad([True, True]),  # type: ignore[arg-type]
        sample_rate=4,
        end_silence_seconds=1.0,
        max_phrase_seconds=1.0,
    )
    chunk_a = np.array([1, 1], dtype=np.int16)
    chunk_b = np.array([2, 2], dtype=np.int16)

    assert accumulator.push(chunk_a) is None

    phrase = accumulator.push(chunk_b)

    assert phrase is not None
    np.testing.assert_array_equal(phrase, np.array([1, 1, 2, 2], dtype=np.int16))

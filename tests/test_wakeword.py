from cortana.audio.wakeword import WakeWordDetector


def test_wake_word_exact() -> None:
    detector = WakeWordDetector("cortana", threshold=90)
    assert detector.is_wake("hey cortana")


def test_wake_word_fuzzy() -> None:
    detector = WakeWordDetector("cortana", threshold=70)
    assert detector.is_wake("kortana")


def test_wake_word_negative() -> None:
    detector = WakeWordDetector("cortana", threshold=95)
    assert detector.is_wake("weather report") is False

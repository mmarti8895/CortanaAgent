from __future__ import annotations

from dataclasses import dataclass

from rapidfuzz import fuzz


@dataclass(slots=True)
class WakeWordDetector:
    wake_word: str
    threshold: int = 85

    def is_wake(self, text: str) -> bool:
        norm = " ".join(text.lower().strip().split())
        if not norm:
            return False
        if self.wake_word.lower() in norm:
            return True
        score = fuzz.partial_ratio(self.wake_word.lower(), norm)
        return score >= self.threshold

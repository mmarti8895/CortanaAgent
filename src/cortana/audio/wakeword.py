from __future__ import annotations

from dataclasses import dataclass
import re

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

    def strip_wake(self, text: str) -> str:
        pattern = re.compile(rf"\b{re.escape(self.wake_word)}\b", re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            return ""
        remainder = text[match.end() :].strip()
        return remainder.lstrip(" ,:;.!?-")

"""
sm2.py - SM-2 spaced repetition algorithm (SuperMemo 2).

Quality scale used here (simplified for a 3-button review UI):
    0 = "Again" (forgot completely)
    3 = "Hard"  (recalled with difficulty)
    4 = "Good"  (recalled correctly)
    5 = "Easy"  (recalled easily)

Reference: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
"""

from datetime import date, timedelta
from typing import Tuple


def sm2(quality: int, repetitions: int, ease_factor: float, interval_days: int) -> Tuple[int, float, int, str]:
    """
    Given the previous state and a new review quality (0-5), returns the
    updated (repetitions, ease_factor, interval_days, due_date_iso).
    """
    if quality < 3:
        # Forgot it - reset repetitions, review again soon (tomorrow)
        repetitions = 0
        interval_days = 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease_factor)
        repetitions += 1

    ease_factor = ease_factor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    ease_factor = max(1.3, ease_factor)  # SM-2 floor

    due_date = (date.today() + timedelta(days=interval_days)).isoformat()
    return repetitions, ease_factor, interval_days, due_date

from __future__ import annotations
import re
from functools import lru_cache

def normalize_text(timestamp: str) -> str:
    timestamp = timestamp.strip()
    timestamp = re.sub('\\s+', ' ', timestamp)
    return timestamp

@lru_cache(maxsize=1)
def get_morph_analyzer():
    try:
        import pymorphy3
        return pymorphy3.MorphAnalyzer()
    except Exception:

        class Dummy:

            def parse(self, width):
                return []
        return Dummy()

def check_new_vocabulary(text: str, accumulator: list[str]) -> None:
    for item_w in re.findall('[A-Za-zА-Яа-яЁё\\-]+', text):
        if len(item_w) > 12 and item_w not in accumulator:
            accumulator.append(item_w)
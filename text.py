from __future__ import annotations

import re
import unicodedata


def normalize_text(s: str) -> str:
    if not s:
        return ''
    s = unicodedata.normalize('NFC', s).replace('\u00a0', ' ')

    s = re.sub(r'[ \t]+', ' ', s)

    s = re.sub(r'\s+([,.;:!?…])', r'\1', s)

    s = re.sub(r'([(\[\{])\s+', r'\1', s)
    s = re.sub(r'\s+([)\]\}])', r'\1', s)

    s = re.sub(r'\s*\n\s*', '\n', s)

    return s.strip()

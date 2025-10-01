from __future__ import annotations

import difflib

try:
    from rapidfuzz.distance import Levenshtein

    _HAS_RF = True
except Exception:
    _HAS_RF = False


def _sim(arg_a: str, arg_b: str) -> float:
    var_a, var_b = (arg_a.lower(), arg_b.lower())
    var_a = var_a.replace('ё', 'е')
    var_b = var_b.replace('ё', 'е')
    if _HAS_RF:
        return float(Levenshtein.normalized_similarity(var_a, var_b))
    return float(difflib.SequenceMatcher(a=var_a, b=var_b).ratio())


def align_tokens(
    ref_tokens: list[str], hyp_tokens: list[str]
) -> list[tuple[str, str, float]]:
    text = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    aligned: list[tuple[str, str, float]] = []
    for tag, i1, i2, j1, j2 in text.get_opcodes():
        if tag == 'equal':
            for index3 in range(i2 - i1):
                var_w = ref_tokens[i1 + index3]
                aligned.append((var_w, var_w, 1.0))
        elif tag == 'replace':
            length = min(i2 - i1, j2 - j1)
            for index3 in range(length):
                rw = ref_tokens[i1 + index3]
                hw = hyp_tokens[j1 + index3]
                aligned.append((rw, hw, _sim(rw, hw)))
            for rw in ref_tokens[i1 + length : i2]:
                aligned.append((rw, '', 0.0))
            for hw in hyp_tokens[j1 + length : j2]:
                aligned.append(('', hw, 0.0))
        elif tag == 'delete':
            for rw in ref_tokens[i1:i2]:
                aligned.append((rw, '', 0.0))
        elif tag == 'insert':
            for hw in hyp_tokens[j1:j2]:
                aligned.append(('', hw, 0.0))
    return aligned


def expert_alignment(ref: str, hyp: str) -> dict:
    ref_toks = ref.split()
    hyp_toks = hyp.split()
    pairs = align_tokens(ref_toks, hyp_toks)
    count = max(1, len(ref_toks))
    text = sum(
        (1 for item_r, item_h, text in pairs if item_r and item_h and (text < 1.0))
    )
    var_d = sum((1 for item_r, item_h, text in pairs if item_r and (not item_h)))
    index = sum((1 for item_r, item_h, text in pairs if item_h and (not item_r)))
    wer = (text + var_d + index) / count
    sims = [text for item_r, item_h, text in pairs if item_r and item_h]
    char_sim = float(sum(sims) / max(1, len(sims)))
    chunks: list[str] = []
    for item_r, item_h, text in pairs:
        if item_r and item_h:
            if text == 1.0:
                chunks.append(f"<span style='color:#2e7d32'>{item_h}</span>")
            else:
                chunks.append(
                    f"<span title='эталон: {item_r}; похожесть: {text:.2f}' style='background:#fff3cd;border-bottom:1px dotted #f0ad4e'>{item_h}</span>"
                )
        elif item_r and (not item_h):
            chunks.append(
                f"<span title='удалено; эталон: {item_r}' style='background:#ffe0e0;text-decoration:line-through'>{item_r}</span>"
            )
        elif item_h and (not item_r):
            chunks.append(
                f"<span title='вставка' style='background:#e0f0ff'>{item_h}</span>"
            )
    diff_html = ' '.join(chunks)
    return {
        'alignment': [
            {'ref': item_r, 'hyp': item_h, 'sim': round(text, 3)}
            for item_r, item_h, text in pairs
        ],
        'wer': wer,
        'char_sim': char_sim,
        'diff_html': diff_html,
    }


def align_texts(ref: str, hyp: str) -> dict[str, str]:
    res = expert_alignment(ref, hyp)
    return {'diff_html': res['diff_html']}


def word_accuracy(ref: str, hyp: str) -> float:
    if not ref.strip():
        return 0.0
    ref_words, hyp_words = (ref.split(), hyp.split())
    text = difflib.SequenceMatcher(a=ref_words, b=hyp_words)
    matches = sum(triple.size for triple in text.get_matching_blocks())
    return matches / max(1, len(ref_words))

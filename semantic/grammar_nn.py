from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    import stanza

    _STANZA = stanza.Pipeline(
        lang='ru',
        processors='tokenize,pos,lemma,depparse',
        tokenize_no_ssplit=False,
        use_gpu=False,
    )
except Exception as e:
    _STANZA = None
    _INIT_ERR = e
else:
    _INIT_ERR = None
GRAMMAR_WEIGHTS = {
    'spo_presence': 0.4,
    'subject_case': 0.1,
    'predicate_form': 0.15,
    'object_case': 0.15,
    'adj_agreement': 0.1,
    'adv_position': 0.05,
    'lesson_usage': 0.05,
}
PREP2CASE = {
    'в': {'accs', 'loct'},
    'во': {'accs', 'loct'},
    'на': {'accs', 'loct'},
    'к': {'datv'},
    'ко': {'datv'},
    'с': {'ablt', 'gent'},
    'со': {'ablt', 'gent'},
    'у': {'gent'},
    'от': {'gent'},
    'из': {'gent'},
    'изо': {'gent'},
    'для': {'gent'},
    'без': {'gent'},
    'по': {'datv', 'loct'},
    'о': {'loct'},
    'об': {'loct'},
    'обо': {'loct'},
    'про': {'accs'},
    'через': {'accs'},
    'над': {'ablt'},
    'под': {'ablt', 'accs'},
    'между': {'ablt'},
    'при': {'loct'},
}


@dataclass
class Word:
    id: int
    text: str
    lemma: str
    upos: str
    head: int
    deprel: str
    feats: dict[str, str]
    i_sent: int


@dataclass
class SentenceReport:
    raw: str
    subject: Word | None
    predicate: Word | None
    obj: Word | None
    checks: dict[str, bool]
    suggestions: list[str]


def _ensure_ready():
    if _STANZA is None:
        raise RuntimeError(
            f'Stanza не инициализирована. Установите/скачайте модель русского: {repr(_INIT_ERR)}'
        )


def _sent_split(text: str) -> list[str]:
    parts = re.split('(?<=[\\.\\!\\?…])\\s+', text.strip())
    return [path.strip() for path in parts if path.strip()]


def _to_words(doc) -> list[list[Word]]:
    sents: list[list[Word]] = []
    for index, text in enumerate(doc.sentences):
        row: list[Word] = []
        for item_w in text.words:
            feats = {}
            if item_w.feats:
                for kv in item_w.feats.split('|'):
                    if '=' in kv:
                        index3, var_v = kv.split('=', 1)
                        feats[index3] = var_v
            row.append(
                Word(
                    id=item_w.id,
                    text=item_w.text,
                    lemma=item_w.lemma,
                    upos=item_w.upos,
                    head=item_w.head,
                    deprel=item_w.deprel,
                    feats=feats,
                    i_sent=index,
                )
            )
        sents.append(row)
    return sents


def _index(words: list[Word]) -> dict[int, Word]:
    return {item_w.id: item_w for item_w in words}


def _find_predicate(words: list[Word]) -> Word | None:
    root = next(
        (
            item_w
            for item_w in words
            if item_w.deprel == 'root' and item_w.upos in ('VERB', 'AUX')
        ),
        None,
    )
    if root:
        return root
    return next((item_w for item_w in words if item_w.upos == 'VERB'), None)


def _find_subject(words: list[Word], pred: Word | None) -> Word | None:
    if not pred:
        return None
    subs = [
        item_w
        for item_w in words
        if item_w.deprel.startswith('nsubj') and item_w.head == pred.id
    ]
    if subs:
        return subs[0]
    conj_verbs = {
        item_w.id
        for item_w in words
        if item_w.upos == 'VERB'
        and item_w.deprel == 'conj'
        and (item_w.head == pred.id)
    }
    subs = [
        item_w
        for item_w in words
        if item_w.deprel.startswith('nsubj') and item_w.head in conj_verbs
    ]
    return subs[0] if subs else None


def _find_governing_prep(words: list[Word], noun: Word) -> str | None:
    for item_w in words:
        if (
            item_w.head == noun.id
            and item_w.deprel == 'case'
            and (item_w.upos == 'ADP')
        ):
            return item_w.lemma.lower()
    return None


def _find_object(
    words: list[Word], pred: Word | None
) -> tuple[Word | None, str, str | None]:
    if not pred:
        return (None, 'none', None)
    objs = [
        item_w
        for item_w in words
        if item_w.deprel == 'obj'
        and item_w.head == pred.id
        and (item_w.upos in ('NOUN', 'PROPN', 'PRON'))
    ]
    if objs:
        return (objs[0], 'obj', _find_governing_prep(words, objs[0]))
    iobjs = [
        item_w
        for item_w in words
        if item_w.deprel == 'iobj'
        and item_w.head == pred.id
        and (item_w.upos in ('NOUN', 'PROPN', 'PRON'))
    ]
    if iobjs:
        return (iobjs[0], 'iobj', _find_governing_prep(words, iobjs[0]))
    obls = [
        item_w
        for item_w in words
        if item_w.deprel == 'obl'
        and item_w.head == pred.id
        and (item_w.upos in ('NOUN', 'PROPN', 'PRON'))
    ]
    if obls:
        return (obls[0], 'obl', _find_governing_prep(words, obls[0]))
    return (None, 'none', None)


def _has_negation(words: list[Word], pred: Word | None) -> bool:
    if not pred:
        return False
    for item_w in words:
        if item_w.head == pred.id and (
            item_w.deprel.startswith('advmod') or item_w.upos == 'PART'
        ):
            if item_w.lemma.lower() == 'не' or item_w.text.lower() == 'не':
                return True
    return False


def _bool2pct(flags: list[bool]) -> float:
    return 100.0 * (sum(1 for value_x in flags if value_x) / max(1, len(flags)))


def _agree(arg_a: dict[str, str], arg_b: dict[str, str]) -> bool:
    case_ok = (
        'Case' not in arg_a or 'Case' not in arg_b or arg_a['Case'] == arg_b['Case']
    )
    num_ok = (
        'Number' not in arg_a
        or 'Number' not in arg_b
        or arg_a['Number'] == arg_b['Number']
    )
    gen_ok = (
        'Gender' not in arg_a
        or 'Gender' not in arg_b
        or arg_a['Gender'] == arg_b['Gender']
    )
    return case_ok and num_ok and gen_ok


def analyze_text_nn(
    text: str,
    lesson_items: list[str] | None = None,
    allow_subject_ellipsis: bool = False,
) -> dict[str, Any]:
    _ensure_ready()
    text = text.strip()
    if not text:
        return {'score': 0.0, 'error': 'empty_text'}
    doc = _STANZA(text)
    sents = _to_words(doc)
    lesson = set(
        [item_l.strip().lower() for item_l in lesson_items or [] if item_l.strip()]
    )
    used_lemmas: set[str] = set()
    per_sent: list[SentenceReport] = []
    prev_subject: Word | None = None
    for words in sents:
        raw = ''.join(tok.text + (' ' if tok.text.strip() else '') for tok in words)
        pred = _find_predicate(words)
        subj = _find_subject(words, pred)
        obj, obj_kind, obj_prep = _find_object(words, pred)
        neg = _has_negation(words, pred)
        if lesson:
            for item_w in words:
                if item_w.lemma.lower() in lesson:
                    used_lemmas.add(item_w.lemma.lower())
        checks: dict[str, bool] = {
            'spo_presence': subj is not None and pred is not None and (obj is not None),
            'subject_case_ok': False,
            'predicate_form_ok': False,
            'object_case_ok': False,
            'adj_agreement_ok': True,
            'adv_position_ok': True,
        }
        suggestions: list[str] = []
        if subj is None and allow_subject_ellipsis and (prev_subject is not None):
            subj = prev_subject
        if subj is not None:
            checks['subject_case_ok'] = subj.feats.get('Case') == 'Nom'
        else:
            suggestions.append(
                'Нет субъекта: добавьте существительное/местоимение в И.п.'
            )
        if pred is not None:
            vform = pred.feats.get('VerbForm')
            tense = pred.feats.get('Tense')
            checks['predicate_form_ok'] = vform == 'Fin' or tense in {
                'Past',
                'Pres',
                'Fut',
            }
        else:
            suggestions.append('Нет предиката: добавьте личную форму глагола.')
        if obj is None:
            suggestions.append('Нет объекта: добавьте существительное-объект.')
        else:
            case = obj.feats.get('Case')
            ok = False
            if obj_kind == 'obj':
                if case == 'Acc':
                    ok = True
                elif neg and case == 'Gen':
                    ok = True
            elif obj_kind == 'iobj':
                ok = case == 'Dat'
            elif obj_kind == 'obl':
                if obj_prep:
                    allowed = PREP2CASE.get(obj_prep, set())
                    ok = (
                        case.lower() in {item_c.lower() for item_c in allowed}
                        if allowed
                        else case is not None
                    )
                else:
                    ok = case is not None and case != 'Nom'
            checks['object_case_ok'] = bool(ok)
            if not ok:
                hint = f"Проверьте падеж объекта «{obj.text}» (обнаружен {case or '—'})"
                if obj_kind == 'obj':
                    hint += (
                        '; ожидается винительный (при отрицании допустим родительный).'
                    )
                elif obj_kind == 'iobj':
                    hint += '; обычно дательный для косвенного дополнения.'
                elif obj_kind == 'obl' and obj_prep:
                    hint += f"; с предлогом «{obj_prep}» допустимы: {', '.join(sorted(PREP2CASE.get(obj_prep, [])))}."
                suggestions.append(hint)
        for item_w in words:
            if item_w.deprel == 'amod' and item_w.upos == 'ADJ':
                head = next(
                    (value_x for value_x in words if value_x.id == item_w.head), None
                )
                if head and head.upos == 'NOUN':
                    order_ok = item_w.id < head.id
                    agr_ok = _agree(item_w.feats, head.feats)
                    if not (order_ok and agr_ok):
                        checks['adj_agreement_ok'] = False
                        suggestions.append(
                            f'Проверьте прилагательное «{item_w.text}» и существительное «{head.text}»: порядок/согласование.'
                        )
        if pred is not None:
            for item_w in words:
                if item_w.deprel.startswith('advmod') and item_w.head == pred.id:
                    if not item_w.id < pred.id:
                        checks['adv_position_ok'] = False
                        suggestions.append(
                            f'Наречие «{item_w.text}» должно стоять перед глаголом «{pred.text}».'
                        )
        per_sent.append(
            SentenceReport(
                raw=raw,
                subject=subj,
                predicate=pred,
                obj=obj,
                checks=checks,
                suggestions=suggestions,
            )
        )
        if subj is not None:
            prev_subject = subj

    def agg(key: str) -> float:
        return _bool2pct([item_r.checks[key] for item_r in per_sent])

    spo = agg('spo_presence')
    subj = agg('subject_case_ok')
    pred = agg('predicate_form_ok')
    obj = agg('object_case_ok')
    adj = agg('adj_agreement_ok')
    adv = agg('adv_position_ok')
    lesson_used = sorted(list(used_lemmas))
    if lesson:
        ratio = 100.0 * len(lesson_used) / max(1, len(lesson))
        if 8.0 <= ratio <= 10.0:
            lesson_score = 100.0
        elif ratio < 8.0:
            lesson_score = max(0.0, 100.0 * (ratio / 8.0))
        else:
            lesson_score = max(0.0, 100.0 * ((25.0 - min(25.0, ratio)) / 15.0))
    else:
        lesson_score = 0.0
    var_w = GRAMMAR_WEIGHTS
    overall = (
        spo * var_w['spo_presence']
        + subj * var_w['subject_case']
        + pred * var_w['predicate_form']
        + obj * var_w['object_case']
        + adj * var_w['adj_agreement']
        + adv * var_w['adv_position']
        + lesson_score * var_w['lesson_usage']
    )
    out_sents = []
    for item_r in per_sent:
        out_sents.append(
            {
                'sentence': item_r.raw,
                'subject': None if item_r.subject is None else item_r.subject.text,
                'predicate': None
                if item_r.predicate is None
                else item_r.predicate.text,
                'object': None if item_r.obj is None else item_r.obj.text,
                'checks': item_r.checks,
                'suggestions': item_r.suggestions,
            }
        )
    return {
        'score': round(float(overall), 1),
        'weights': var_w,
        'aggregates': {
            'spo_presence': round(spo, 1),
            'subject_case_ok': round(subj, 1),
            'predicate_form_ok': round(pred, 1),
            'object_case_ok': round(obj, 1),
            'adj_agreement_ok': round(adj, 1),
            'adv_position_ok': round(adv, 1),
            'lesson_usage_score': round(lesson_score, 1),
        },
        'lesson_items_used': lesson_used,
        'sentences': out_sents,
    }

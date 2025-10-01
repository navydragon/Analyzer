from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    import pymorphy3

    _MORPH = pymorphy3.MorphAnalyzer()
except Exception:
    _MORPH = None
GRAMMAR_WEIGHTS = {
    'spo_presence': 0.4,
    'subject_case': 0.1,
    'predicate_form': 0.15,
    'object_case': 0.15,
    'adj_agreement': 0.1,
    'adv_position': 0.05,
    'lesson_usage': 0.05,
}
VERB_GOVERN_CASE = {
    'читать': 'accs',
    'прочитать': 'accs',
    'писать': 'accs',
    'написать': 'accs',
    'видеть': 'accs',
    'увидеть': 'accs',
    'любить': 'accs',
    'сделать': 'accs',
    'делать': 'accs',
    'изучать': 'accs',
    'выучить': 'accs',
    'покупать': 'accs',
    'купить': 'accs',
    'готовить': 'accs',
    'приготовить': 'accs',
    'слушать': 'accs',
    'услышать': 'accs',
    'смотреть': 'accs',
    'посмотреть': 'accs',
    'иметь': 'accs',
    'оснастить': 'ablt',
    'оснащать': 'ablt',
}
VERB_PREP_GOVERN = {'подходить': {'для': 'gent'}}
PRESENT_ENDINGS = (
    'у',
    'ю',
    'ешь',
    'ёшь',
    'ет',
    'ёт',
    'ем',
    'ём',
    'ете',
    'ёте',
    'ут',
    'ют',
    'ишь',
    'ит',
    'им',
    'ите',
    'ат',
    'ят',
)
PAST_ENDINGS = ('л', 'ла', 'ло', 'ли')


@dataclass
class Token:
    text: str
    lemma: str
    pos: str
    case: str | None = None
    number: str | None = None
    gender: str | None = None
    tense: str | None = None
    person: str | None = None
    animacy: str | None = None


@dataclass
class SentenceReport:
    sent_text: str
    subject: Token | None
    predicate: Token | None
    obj: Token | None
    spo_presence: bool
    subject_case_ok: bool
    predicate_form_ok: bool
    object_case_ok: bool
    adj_agreement_ok: bool
    adv_position_ok: bool
    suggestions: list[str]


def _ensure_morph():
    if _MORPH is None:
        raise RuntimeError(
            'pymorphy3 не установлен. Установите пакет: pip install pymorphy3'
        )


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    parts = re.split('(?<=[\\.\\!\\?…])\\s+', text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(sent: str) -> list[str]:
    return re.findall('[A-Za-zА-Яа-яЁё\\-]+', sent)


def morph_parse_word(w: str) -> Token:
    _ensure_morph()
    path = _MORPH.parse(w)[0]
    tag = path.tag
    pos = str(tag.POS) if tag.POS else ''
    case = str(tag.case) if tag.case else None
    number = str(tag.number) if tag.number else None
    gender = str(tag.gender) if tag.gender else None
    tense = str(tag.tense) if tag.tense else None
    person = str(tag.person) if tag.person else None
    animacy = str(tag.animacy) if tag.animacy else None
    return Token(
        text=w,
        lemma=path.normal_form,
        pos=pos,
        case=case,
        number=number,
        gender=gender,
        tense=tense,
        person=person,
        animacy=animacy,
    )


def analyze_sentence(
    sent: str, lesson_items: list[str] | None = None
) -> SentenceReport:
    words = tokenize(sent)
    tokens = [morph_parse_word(w) for w in words]
    lower_words = [w.lower() for w in words]
    predicate: Token | None = None
    pred_idx = -1
    for index, timestamp in enumerate(tokens):
        if timestamp.pos in ('VERB',):
            predicate = timestamp
            pred_idx = index
            break
    if predicate is None:
        for index, timestamp in enumerate(tokens):
            if timestamp.pos in ('INFN', 'PRTF', 'GRND'):
                if timestamp.text.lower().endswith(
                    PAST_ENDINGS
                ) or timestamp.text.lower().endswith(PRESENT_ENDINGS):
                    predicate = timestamp
                    pred_idx = index
                    break
    subject: Token | None = None
    if predicate is not None:
        for index in range(pred_idx - 1, -1, -1):
            timestamp = tokens[index]
            if timestamp.pos in ('NOUN', 'NPRO'):
                subject = timestamp
                break
    obj: Token | None = None
    negation = False
    if predicate is not None:
        for index in range(max(0, pred_idx - 3), pred_idx):
            if lower_words[index] == 'не':
                negation = True
                break
        for index in range(pred_idx + 1, len(tokens)):
            timestamp = tokens[index]
            if timestamp.pos == 'PREP':
                continue
            if timestamp.pos == 'NOUN':
                obj = timestamp
                break
    spo_presence = subject is not None and predicate is not None and (obj is not None)
    subject_case_ok = False
    if subject is not None:
        subject_case_ok = subject.case == 'nomn'
    predicate_form_ok = False
    if predicate is not None:
        word = predicate.text.lower()
        if predicate.tense == 'past':
            predicate_form_ok = word.endswith(PAST_ENDINGS)
        else:
            predicate_form_ok = word.endswith(PRESENT_ENDINGS)
    object_case_ok = False
    if obj is not None:
        required = None
        if predicate is not None:
            required = VERB_GOVERN_CASE.get(predicate.lemma)
        if required is None:
            required = 'accs'
        if negation and obj.case == 'gent':
            object_case_ok = True
        else:
            object_case_ok = obj.case == required
    adj_agreement_ok = True
    for index, timestamp in enumerate(tokens[:-1]):
        if timestamp.pos in ('ADJF', 'ADJS') and tokens[index + 1].pos == 'NOUN':
            var_a, count = (timestamp, tokens[index + 1])
            before_ok = True
            same_case = var_a.case == count.case or var_a.pos == 'ADJS'
            same_num = var_a.number == count.number or count.number == 'plur'
            same_gen = var_a.gender == count.gender or count.number == 'plur'
            if not (before_ok and same_case and same_num and same_gen):
                adj_agreement_ok = False
                break
    adv_position_ok = True
    if predicate is not None:
        for index in range(pred_idx + 1, len(tokens)):
            if tokens[index].pos == 'ADVB':
                adv_position_ok = False
                break
    suggestions: list[str] = []
    if predicate is None:
        suggestions.append(
            'Нет предиката: добавьте глагол-сказуемое с правильным окончанием.'
        )
    if subject is None:
        suggestions.append(
            'Нет субъекта: добавьте существительное/местоимение в именительном падеже перед глаголом.'
        )
    if obj is None:
        suggestions.append(
            'Нет объекта: добавьте существительное-объект после глагола.'
        )
    elif not object_case_ok:
        try:
            forms = _MORPH.parse(obj.text)[0]
            need = (
                'gent'
                if negation
                else VERB_GOVERN_CASE.get(predicate.lemma, 'accs')
                if predicate
                else 'accs'
            )
            inflected = forms.inflect({need})
            if inflected:
                suggestions.append(
                    f'Объект «{obj.text}» лучше в {need.upper()} → «{inflected.word}».'
                )
            else:
                suggestions.append(
                    f'Проверьте падеж объекта «{obj.text}». Ожидается {need.upper()}.'
                )
        except Exception:
            suggestions.append(f'Проверьте падеж объекта «{obj.text}».')
    if subject is not None and (not subject_case_ok):
        suggestions.append(
            f'Субъект «{subject.text}» должен быть в именительном падеже.'
        )
    if predicate is not None and (not predicate_form_ok):
        suggestions.append(
            f'Проверьте окончание глагола «{predicate.text}» (настоящее/будущее: у/ешь/ет/...; прошедшее: -л/-ла/-ло/-ли).'
        )
    if not adj_agreement_ok:
        suggestions.append(
            'Проверьте согласование прилагательного с существительным (род, число, падеж) и порядок (прилагательное перед существительным).'
        )
    if not adv_position_ok:
        suggestions.append(
            'Наречие должно стоять перед глаголом, к которому относится.'
        )
    return SentenceReport(
        sent_text=sent,
        subject=subject,
        predicate=predicate,
        obj=obj,
        spo_presence=spo_presence,
        subject_case_ok=subject_case_ok,
        predicate_form_ok=predicate_form_ok,
        object_case_ok=object_case_ok,
        adj_agreement_ok=adj_agreement_ok,
        adv_position_ok=adv_position_ok,
        suggestions=suggestions,
    )


def lesson_usage_ratio(
    text: str, lesson_items: list[str] | None
) -> tuple[float, list[str]]:
    if not lesson_items:
        return (0.0, [])
    lemmas = [morph_parse_word(w).lemma for w in tokenize(text)]
    items = set([i.strip().lower() for i in lesson_items if i.strip()])
    used = sorted(list(set(lemmas) & items))
    ratio = 100.0 * len(used) / max(1, len(items))
    return (ratio, used)


def analyze_text(text: str, lesson_items: list[str] | None = None) -> dict[str, Any]:
    _ensure_morph()
    sents = split_sentences(text)
    sent_reports = [analyze_sentence(s, lesson_items) for s in sents] if sents else []

    def avg_bool(field: str) -> float:
        if not sent_reports:
            return 0.0
        return (
            100.0
            * sum(1 for item_r in sent_reports if getattr(item_r, field))
            / len(sent_reports)
        )

    spo_ok = avg_bool('spo_presence')
    subj_ok = avg_bool('subject_case_ok')
    pred_ok = avg_bool('predicate_form_ok')
    obj_ok = avg_bool('object_case_ok')
    adj_ok = avg_bool('adj_agreement_ok')
    adv_ok = avg_bool('adv_position_ok')
    lesson_ratio, lesson_used = lesson_usage_ratio(text, lesson_items)
    if lesson_items:
        if 8.0 <= lesson_ratio <= 10.0:
            lesson_score = 100.0
        elif lesson_ratio < 8.0:
            lesson_score = max(0.0, 100.0 * (lesson_ratio / 8.0))
        else:
            lesson_score = max(0.0, 100.0 * ((25.0 - min(25.0, lesson_ratio)) / 15.0))
    else:
        lesson_score = 0.0
    var_w = GRAMMAR_WEIGHTS
    overall = (
        spo_ok * var_w['spo_presence']
        + subj_ok * var_w['subject_case']
        + pred_ok * var_w['predicate_form']
        + obj_ok * var_w['object_case']
        + adj_ok * var_w['adj_agreement']
        + adv_ok * var_w['adv_position']
        + lesson_score * var_w['lesson_usage']
    )
    per_sentence = []
    for item_r in sent_reports:
        per_sentence.append(
            {
                'sentence': item_r.sent_text,
                'subject': None if item_r.subject is None else item_r.subject.text,
                'predicate': None
                if item_r.predicate is None
                else item_r.predicate.text,
                'object': None if item_r.obj is None else item_r.obj.text,
                'checks': {
                    'spo_presence': item_r.spo_presence,
                    'subject_case_ok': item_r.subject_case_ok,
                    'predicate_form_ok': item_r.predicate_form_ok,
                    'object_case_ok': item_r.object_case_ok,
                    'adj_agreement_ok': item_r.adj_agreement_ok,
                    'adv_position_ok': item_r.adv_position_ok,
                },
                'suggestions': item_r.suggestions,
            }
        )
    report = {
        'score': round(overall, 1),
        'weights': var_w,
        'aggregates': {
            'spo_presence': round(spo_ok, 1),
            'subject_case_ok': round(subj_ok, 1),
            'predicate_form_ok': round(pred_ok, 1),
            'object_case_ok': round(obj_ok, 1),
            'adj_agreement_ok': round(adj_ok, 1),
            'adv_position_ok': round(adv_ok, 1),
            'lesson_usage_score': round(lesson_score, 1),
        },
        'lesson_items_used': lesson_used,
        'sentences': per_sentence,
    }
    return report

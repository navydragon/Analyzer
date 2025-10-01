import sys
from pathlib import Path

import streamlit as st

from semantic.grammar_nn import analyze_text_nn
from text import normalize_text

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


st.title('📝 Семантика и грамматика текста')
st.caption(
    'Два режима: 1) Грамматика (SPO, падежи, согласование) — без эталона; 2) Семантическая близость — по эталону/критериям.'
)
tab1, tab2 = st.tabs(['🔤 Грамматика (SPO)', '🧠 Семантическая близость'])
with tab1:
    st.subheader('Входные данные')
    user_text = st.text_area(
        'Текст для анализа',
        height=220,
        key='grammar_text',
        placeholder='Вставьте предложения для проверки по правилам S–P–O, падежам, прилагательным и наречиям.',
    )
    with st.expander('⚙️ Опции проверки', expanded=False):
        allow_ell = st.checkbox(
            'Разрешать эллипсис субъекта (подставлять S из предыдущего предложения)',
            value=False,
            key='allow_ell',
        )
        lesson_items_str = st.text_area(
            'Материал урока (леммы через запятую, опционально)',
            placeholder='например: читать, интересный, быстро, студент, книга',
            height=80,
            key='lesson_items_str',
        )
        lesson_items = (
            [x.strip().lower() for x in lesson_items_str.split(',') if x.strip()]
            if lesson_items_str
            else None
        )
    if st.button(
        '🚀 Анализировать грамматику', type='primary', use_container_width=True
    ):
        if not user_text.strip():
            st.warning('Введите текст.')
        else:
            norm = normalize_text(user_text)
            st.subheader('Нормализация')
            st.write(norm)
            try:
                report = analyze_text_nn(
                    norm, lesson_items=lesson_items, allow_subject_ellipsis=allow_ell
                )
                st.subheader('🏁 Итоговая оценка по грамматике')
                st.metric('Общий балл', f'{report["score"]:.1f} / 100')
                st.markdown('#### Сводные показатели')
                cols = st.columns(3)
                agg = report['aggregates']
                with cols[0]:
                    st.metric('S-P-O присутствует', f'{agg["spo_presence"]:.1f}')
                    st.metric('Субъект в И.п.', f'{agg["subject_case_ok"]:.1f}')
                with cols[1]:
                    st.metric('Форма предиката', f'{agg["predicate_form_ok"]:.1f}')
                    st.metric('Падеж объекта', f'{agg["object_case_ok"]:.1f}')
                with cols[2]:
                    st.metric('Согласование прил.', f'{agg["adj_agreement_ok"]:.1f}')
                    st.metric('Позиция наречий', f'{agg["adv_position_ok"]:.1f}')
                if report.get('lesson_items_used'):
                    st.metric(
                        'Материал урока (балл)', f'{agg["lesson_usage_score"]:.1f}'
                    )
                    st.caption(
                        'Использовано из списка: '
                        + ', '.join(report['lesson_items_used'])
                    )
                st.markdown('#### Детально по предложениям')
                import pandas as pd

                rows = []
                for s in report['sentences']:
                    checks = s['checks']
                    rows.append(
                        {
                            'Предложение': s['sentence'],
                            'Субъект': s['subject'] or '—',
                            'Предикат': s['predicate'] or '—',
                            'Объект': s['object'] or '—',
                            'SPO': '✓' if checks['spo_presence'] else '✗',
                            'Subj И.п.': '✓' if checks['subject_case_ok'] else '✗',
                            'Предикат ок': '✓' if checks['predicate_form_ok'] else '✗',
                            'Obj падеж': '✓' if checks['object_case_ok'] else '✗',
                            'Прил. соглас.': '✓' if checks['adj_agreement_ok'] else '✗',
                            'Нареч. позиция': '✓' if checks['adv_position_ok'] else '✗',
                            'Рекомендации': ' • '.join(s['suggestions']),
                        }
                    )
                df = pd.DataFrame(rows)
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.markdown(
                    '> Разбор выполняется нейросетевым парсером UD (Stanza): роли S/P/O, падежи, amod/advmod, предлоги и отрицание учитываются без «словари под каждый глагол».'
                )
            except RuntimeError as e:
                st.error(f'Ошибка анализа: {e}')
                st.info(
                    'Установите Stanza и модели русского: `pip install stanza` затем `python -c "import stanza; stanza.download(\'ru\')"`.'
                )
with tab2:
    st.subheader('Входные данные')
    user_text_sem = st.text_area('Текст для оценки', height=180, key='sem_text')
    reference = st.text_area(
        'Эталон/критерии (для этого режима)',
        height=120,
        placeholder='Опишите, что именно должен содержать текст, или вставьте эталон.',
        key='sem_ref',
    )
    st.markdown('---')
    st.subheader('🧠 Провайдер и запуск')
    provider = st.selectbox(
        'Провайдер:',
        ['local-sbert', 'openai', 'hf-api'],
        help="'local-sbert' — локально; 'openai' и 'hf-api' — требуют ключи в окружении.",
    )
    run_sem = st.button('🚀 Оценить семантическую близость', use_container_width=True)
    if run_sem:
        if not user_text_sem.strip():
            st.warning('Введите текст.')
        else:
            try:
                from semantic.evaluator import SemanticEvaluator

                ev = SemanticEvaluator(provider=provider)
                res = ev.evaluate(
                    user_text_sem, reference=reference or None, criteria=None
                )
                st.metric('Семантический балл', f'{res.score:.1f} / 100')
                st.caption(f'Провайдер: {res.provider}')
                st.json(res.details)
                if isinstance(res.details, dict) and 'error' in res.details:
                    err = str(res.details['error']).lower()
                    if 'reference_empty' in err:
                        st.info(
                            "Для этого режима заполните 'Эталон/критерии'. Если эталон не нужен — пользуйтесь вкладкой 'Грамматика (SPO)'."
                        )
                    if 'openai' in err and 'module' in err:
                        st.info('Похоже, не установлен пакет: `pip install openai`')
                    if 'sentence_transformers' in err:
                        st.info(
                            'Установите: `pip install sentence-transformers` (и при необходимости `pip install torch --index-url https://download.pytorch.org/whl/cpu`)'
                        )
                    if 'hf_api_token' in err or 'hf_api' in err:
                        st.info(
                            'Задайте токен HuggingFace: `$env:HF_API_TOKEN="hf_xxx"` и запустите Streamlit из той же сессии.'
                        )
            except Exception as e:
                st.error(f'Провайдер недоступен: {e}')
                st.info(
                    'Проверьте, что файл `semantic/evaluator.py` существует и установленны зависимости выбранного провайдера.'
                )

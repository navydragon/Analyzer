import streamlit as st

from analyzer.pronunciation_analyzer import (
    AdvancedPronunciationAnalyzer,
    AnalysisError,
    TranscriptionError,
)
from db_connector import save_result
from processors.alignment import align_texts
from processors.audio_processor import extract_basic_features


def main():
    st.title('🎯 Голосовой анализ')
    st.caption(
        'Загрузите аудио и (опц.) эталонный текст. Получите транскрипцию и экспертные метрики.'
    )
    with st.expander('⚙️ Настройки модели'):
        model_size = st.selectbox(
            'Размер модели Whisper', ['small', 'base', 'medium', 'large'], index=1
        )
        lang = st.text_input(
            'Язык (например, ru, en) — пусто = автоопределение', value='russian'
        )
    wav = st.file_uploader('Загрузите аудио (wav/mp3/m4a)', type=['wav', 'mp3', 'm4a'])
    reference = st.text_area('Эталонный текст', height=120)
    cols = st.columns(2)
    with cols[0]:
        start = st.button('🚀 Начать анализ', type='primary', use_container_width=True)
    with cols[1]:
        st.write('')
    if not start:
        return
    if not wav:
        st.warning('Добавьте аудио-файл.')
        return
    try:
        analyzer = AdvancedPronunciationAnalyzer(
            model_size=model_size, language=lang or None
        )
        tr = analyzer.transcribe(wav)
        st.subheader('🗣️ Распознанный текст')
        st.write(tr.text or '—')
        st.subheader('📊 Базовые аудио-признаки')
        feats = extract_basic_features(wav)
        st.json(feats.get('summary', {}))
        if reference.strip():
            st.subheader('🔗 Выравнивание с эталоном')
            al = align_texts(reference, tr.text)
            st.write(al['diff_html'], unsafe_allow_html=True)
        st.subheader('🏁 Итоговая оценка')
        score, details = analyzer.score(reference or '', tr, feats)
        if 'alignment' in details:
            st.markdown('#### Смысловые несовпадения (по словам)')
            import pandas as pd

            df = pd.DataFrame(details['alignment']['alignment'])
            df_bad = df[(df['ref'] != df['hyp']) | (df['sim'] < 0.9)]
            st.dataframe(df_bad, hide_index=True, use_container_width=True)
            st.markdown('#### Подсветка расхождений')
            st.write(details['alignment']['diff_html'], unsafe_allow_html=True)
        st.metric('Итоговый балл', f'{score:.1f} / 100')
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric('Лексическая точность', f'{details["lexical_accuracy"]:.1f}')
            st.metric(
                'Надёжность распознавания', f'{details["recognition_reliability"]:.1f}'
            )
        with c2:
            st.metric('Акустическая чистота', f'{details["acoustic_cleanliness"]:.1f}')
            if details['tempo']['wpm'] is not None:
                st.metric('Темп речи (WPM)', f'{details["tempo"]["wpm"]:.1f}')
        with c3:
            st.metric('Темп — балл', f'{details["tempo"]["score"]:.1f}')
            st.metric('Плавность (флюэнси)', f'{details["fluency"]:.1f}')
        save_result(
            'voice', {'recognized': tr.text, 'score': score, 'details': details}
        )
        with st.expander('🔬 Подробности (сырьё и веса)'):
            st.json(details)
        with st.expander('🧩 Слова с таймкодами'):
            st.dataframe(tr.words)
    except TranscriptionError as error:
        st.error(f'Ошибка распознавания: {error}')
    except AnalysisError as error:
        st.error(f'Ошибка анализа: {error}')


if __name__ == '__main__':
    main()

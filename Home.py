import sys
from pathlib import Path

import streamlit as st

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

st.set_page_config(page_title='Оценка речи и семантики', page_icon='🧭', layout='wide')
st.title('🧭 Навигация по приложению')
st.page_link(
    'pages/1_voice_analysis.py', label='🎯 Голосовой анализ — оценка произношения'
)
st.page_link(
    'pages/2_text_semantics.py', label='📝 Семантика текста — грамматика и вторая API'
)

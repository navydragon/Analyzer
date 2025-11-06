from __future__ import annotations

import contextlib
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from processors.alignment import expert_alignment

# Подавляем предупреждения NNPACK и FP16 от PyTorch (неподдерживаемое оборудование)
# Устанавливаем переменные окружения для подавления предупреждений из C++ кода
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
warnings.filterwarnings('ignore', message='.*NNPACK.*', category=UserWarning)
warnings.filterwarnings(
    'ignore', message='.*Could not initialize NNPACK.*', category=UserWarning
)
warnings.filterwarnings(
    'ignore', message='.*FP16 is not supported on CPU.*', category=UserWarning
)


@contextlib.contextmanager
def suppress_stderr_nnpack():
    """Контекстный менеджер для подавления предупреждений NNPACK из stderr."""
    original_stderr = sys.stderr
    try:
        # Создаем фильтр для stderr, который отфильтровывает сообщения NNPACK
        class NNPACKFilter:
            def __init__(self, original_stderr):
                self.original_stderr = original_stderr
                self.buffer = ''

            def write(self, message):
                # Накапливаем сообщения в буфере для многострочных предупреждений
                self.buffer += message
                # Проверяем, содержит ли буфер предупреждение NNPACK
                if '\n' in self.buffer:
                    lines = self.buffer.split('\n')
                    self.buffer = lines[-1]  # Оставляем последнюю неполную строку
                    for line in lines[:-1]:
                        # Фильтруем строки с предупреждениями NNPACK
                        if (
                            'NNPACK' not in line
                            and 'Could not initialize NNPACK' not in line
                        ):
                            self.original_stderr.write(line + '\n')
                return len(message)

            def flush(self):
                # Обрабатываем оставшийся буфер
                if self.buffer:
                    if (
                        'NNPACK' not in self.buffer
                        and 'Could not initialize NNPACK' not in self.buffer
                    ):
                        self.original_stderr.write(self.buffer)
                    self.buffer = ''
                self.original_stderr.flush()

            def __getattr__(self, name):
                return getattr(self.original_stderr, name)

        sys.stderr = NNPACKFilter(original_stderr)
        yield
    finally:
        sys.stderr = original_stderr


class TranscriptionError(RuntimeError): ...


class AnalysisError(RuntimeError): ...


@dataclass
class WordItem:
    word: str
    start: float
    end: float
    prob: float


@dataclass
class Transcript:
    text: str
    words: list[dict[str, Any]]


class AdvancedPronunciationAnalyzer:
    def __init__(self, model_size: str = 'base', language: str | None = None):
        self.model_size = model_size
        self.language = language
        self._model = None
        self._model_lock = None
        self._loading = False

    def _load_model(self):
        """Загружает модель Whisper с кэшированием и thread-safe инициализацией."""
        import logging
        import threading

        logger = logging.getLogger(__name__)

        # Если модель уже загружена, возвращаем её
        if self._model is not None:
            return self._model

        # Инициализируем lock для thread-safety
        if self._model_lock is None:
            self._model_lock = threading.Lock()

        # Thread-safe загрузка модели
        with self._model_lock:
            # Проверяем еще раз после получения lock (double-check pattern)
            if self._model is not None:
                return self._model

            # Проверяем, не загружается ли уже модель
            if self._loading:
                # Ждем, пока другой поток загрузит модель
                while self._loading and self._model is None:
                    import time

                    time.sleep(0.1)
                if self._model is not None:
                    return self._model
                raise TranscriptionError(
                    'Не удалось загрузить модель (таймаут ожидания)'
                )

            try:
                self._loading = True
                logger.info(
                    f'Загрузка модели Whisper: {self.model_size} (это может занять время при первом запуске)...'
                )

                import stable_whisper

                # Подавляем предупреждения при загрузке модели (включая stderr из C++ кода)
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', message='.*NNPACK.*', category=UserWarning
                    )
                    warnings.filterwarnings(
                        'ignore',
                        message='.*Could not initialize NNPACK.*',
                        category=UserWarning,
                    )
                    warnings.filterwarnings(
                        'ignore',
                        message='.*FP16 is not supported on CPU.*',
                        category=UserWarning,
                    )
                    with suppress_stderr_nnpack():
                        self._model = stable_whisper.load_model(self.model_size)

                logger.info(f'Модель Whisper {self.model_size} загружена успешно')
                return self._model
            except Exception as error:
                logger.exception(f'Ошибка загрузки модели Whisper {self.model_size}')
                raise TranscriptionError(
                    f'Не удалось загрузить stable-whisper ({error})'
                )
            finally:
                self._loading = False

    def transcribe(
        self,
        file,
        *,
        temperature: float | None = None,
        beam_size: int | None = None,
        initial_prompt: str | None = None,
    ):
        import logging
        import os
        import pathlib
        import shutil
        import tempfile

        logger = logging.getLogger(__name__)

        if shutil.which('ffmpeg') is None:
            raise TranscriptionError(
                'FFmpeg не найден в PATH. Установите FFmpeg и перезапустите терминал. Проверьте командой: ffmpeg -version'
            )

        # Загружаем модель (с кэшированием)
        try:
            model = self._load_model()
        except Exception:
            logger.exception('Ошибка при загрузке модели Whisper')
            raise
        tmp_path = None
        try:
            if hasattr(file, 'getvalue'):
                data = file.getvalue()
                name = getattr(file, 'name', 'audio.mp3')
                suffix = pathlib.Path(name).suffix or '.mp3'
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                audio_input = tmp_path
            elif hasattr(file, 'read'):
                data = file.read()
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                audio_input = tmp_path
            elif isinstance(file, bytes | bytearray):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(file)
                    tmp_path = tmp.name
                audio_input = tmp_path
            elif isinstance(file, str | os.PathLike):
                audio_input = str(file)
            else:
                raise TranscriptionError('Неподдерживаемый тип входа для аудио.')
            kwargs: dict[str, Any] = {
                'language': self.language,
                'vad': True,
                'word_timestamps': True,
            }
            if temperature is not None:
                kwargs['temperature'] = float(temperature)
            if beam_size is not None:
                kwargs['beam_size'] = int(beam_size)
            if initial_prompt:
                kwargs['initial_prompt'] = str(initial_prompt)
            # Подавляем предупреждения во время транскрипции (включая stderr из C++ кода)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    'ignore', message='.*NNPACK.*', category=UserWarning
                )
                warnings.filterwarnings(
                    'ignore',
                    message='.*Could not initialize NNPACK.*',
                    category=UserWarning,
                )
                warnings.filterwarnings(
                    'ignore',
                    message='.*FP16 is not supported on CPU.*',
                    category=UserWarning,
                )
                with suppress_stderr_nnpack():
                    result = model.transcribe(audio_input, **kwargs)
            if hasattr(result, 'to_dict'):
                rd = result.to_dict()
            elif isinstance(result, dict):
                rd = result
            else:
                rd = {}
            text = (rd.get('text') or getattr(result, 'text', '') or '').strip()
            segments = rd.get('segments')
            if segments is None and hasattr(result, 'segments'):
                segments = result.segments
            words: list[dict[str, Any]] = []
            for seg in segments or []:
                wlist = (
                    seg.get('words', [])
                    if isinstance(seg, dict)
                    else getattr(seg, 'words', []) or []
                )
                for item_w in wlist:
                    if isinstance(item_w, dict):
                        word = (item_w.get('word') or item_w.get('text') or '').strip()
                        start = float(item_w.get('start', 0.0))
                        end = float(item_w.get('end', 0.0))
                        prob = float(item_w.get('probability', item_w.get('prob', 0.0)))
                    else:
                        word = (
                            getattr(item_w, 'word', '') or getattr(item_w, 'text', '')
                        ).strip()
                        start = float(getattr(item_w, 'start', 0.0))
                        end = float(getattr(item_w, 'end', 0.0))
                        prob = float(
                            getattr(item_w, 'probability', getattr(item_w, 'prob', 0.0))
                        )
                    words.append(
                        {'word': word, 'start': start, 'end': end, 'prob': prob}
                    )
            return Transcript(text=text, words=words)
        except Exception as error:
            raise TranscriptionError(str(error))
        finally:
            if tmp_path:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _tempo_metrics(self, words: list[dict[str, Any]]) -> tuple[float | None, float]:
        if not words:
            return (None, 50.0)
        starts = [
            float(item_w.get('start', 0.0))
            for item_w in words
            if isinstance(item_w.get('start', 0.0), int | float)
        ]
        ends = [
            float(item_w.get('end', 0.0))
            for item_w in words
            if isinstance(item_w.get('end', 0.0), int | float)
        ]
        if not starts or not ends:
            return (None, 50.0)
        dur = max(1e-06, max(ends) - min(starts))
        wpm = len(words) / (dur / 60.0)

        def band_score(value_x, good_lo=130, good_hi=160, hard_lo=90, hard_hi=200):
            if value_x <= hard_lo or value_x >= hard_hi:
                return 0.0
            if good_lo <= value_x <= good_hi:
                return 100.0
            if hard_lo < value_x < good_lo:
                return 100.0 * (value_x - hard_lo) / (good_lo - hard_lo)
            if good_hi < value_x < hard_hi:
                return 100.0 * (hard_hi - value_x) / (hard_hi - good_hi)
            return 0.0

        return (float(wpm), float(band_score(wpm)))

    def _fluency_score(self, words: list[dict[str, Any]]) -> float:
        if not words or len(words) < 3:
            return 60.0
        ws = sorted(
            [
                (float(item_w.get('start', 0.0)), float(item_w.get('end', 0.0)))
                for item_w in words
            ],
            key=lambda value_x: value_x[0],
        )
        gaps = []
        for index in range(1, len(ws)):
            prev_end = ws[index - 1][1]
            cur_start = ws[index][0]
            gaps.append(max(0.0, cur_start - prev_end))
        if not gaps:
            return 80.0
        gaps = np.array(gaps, dtype=float)
        long_ratio = float((gaps > 0.6).mean())
        std = float(gaps.std())
        long_ratio_n = min(1.0, long_ratio / 0.4)
        std_n = min(1.0, std / 0.5)
        score = (1.0 - long_ratio_n) * 0.6 + (1.0 - std_n) * 0.4
        return float(max(0.0, min(1.0, score)) * 100.0)

    def score(
        self, reference_text: str, transcript: Transcript, features: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        if not transcript.text:
            raise AnalysisError('Пустая транскрипция')
        if reference_text.strip():
            al = expert_alignment(reference_text, transcript.text)
            wer = al['wer']
            char_sim = float(al['char_sim'])
            lex_acc = (0.6 * (1.0 - wer) + 0.4 * char_sim) * 100.0
        else:
            al = None
            lex_acc = 70.0
        probs = [
            item_w['prob']
            for item_w in transcript.words
            if isinstance(item_w.get('prob'), int | float)
        ]
        rec_rel = (float(np.mean(probs)) if probs else 0.5) * 100.0
        text = features.get('summary', {})
        rms = float(text.get('rms', 0.1))
        zcr = float(text.get('zcr', 0.1))
        cent = float(text.get('centroid', 2000.0))
        cent_norm = float(text.get('centroid_norm', cent / 16000.0))
        noise = max(0.0, 1.0 - min(1.0, abs(zcr - 0.1) / 0.1))
        spectrum = max(0.0, 1.0 - min(1.0, abs(cent_norm - 0.2) / 0.2))
        if rms <= 0.05:
            level = max(0.0, min(1.0, rms / 0.05))
        elif rms >= 0.2:
            level = max(0.0, min(1.0, (0.3 - min(rms, 0.3)) / 0.1))
        else:
            level = 1.0
        acoustic_clean = (noise * 0.4 + spectrum * 0.3 + level * 0.3) * 100.0
        wpm, tempo_score = self._tempo_metrics(transcript.words)
        fluency = self._fluency_score(transcript.words)
        from configs.thresholds import METRIC_WEIGHTS as W

        overall = (
            lex_acc * W['lexical_accuracy']
            + rec_rel * W['recognition_reliability']
            + acoustic_clean * W['acoustic_cleanliness']
            + tempo_score * W['tempo']
            + fluency * W['fluency']
        )
        details = {
            'lexical_accuracy': round(lex_acc, 1),
            'recognition_reliability': round(rec_rel, 1),
            'acoustic_cleanliness': round(acoustic_clean, 1),
            'tempo': {
                'wpm': None if wpm is None else round(wpm, 1),
                'score': round(tempo_score, 1),
            },
            'fluency': round(fluency, 1),
            'weights': W,
            'raw': {
                'rms': rms,
                'zcr': zcr,
                'centroid': cent,
                'centroid_norm': round(cent_norm, 4),
            },
        }
        if al is not None:
            details['alignment'] = al
        return (float(max(0.0, min(100.0, overall))), details)

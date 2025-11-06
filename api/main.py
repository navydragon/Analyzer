from __future__ import annotations

import logging
import os
import time
import warnings
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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

# ruff: noqa: E402 - импорты после установки переменных окружения намеренно
from analyzer.pronunciation_analyzer import (
    AdvancedPronunciationAnalyzer,
    AnalysisError,
    TranscriptionError,
)
from processors.audio_processor import extract_basic_features
from semantic.grammar_nn import analyze_text_nn
from text import normalize_text

_DEBUG = os.getenv('API_DEBUG', '0') in {'1', 'true', 'True'}
logging.basicConfig(level=logging.DEBUG if _DEBUG else logging.INFO)
logger = logging.getLogger('analyzer_api')

app = FastAPI(title='Analyzer REST API', version='1.0.0')

# CORS for front-tests
# Разрешить всем (включая file:// → Origin: null)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*', 'null'],
    allow_origin_regex='.*',
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.on_event('startup')
async def startup_event():
    """Инициализация при старте приложения."""
    import os as _os

    logger.info('Запуск приложения Analyzer REST API')
    logger.info(f'MODEL_SIZE={_os.getenv("MODEL_SIZE", "base")}')
    logger.info(f'LANGUAGE={_os.getenv("LANGUAGE", "не указан")}')
    logger.info(f'UVICORN_RELOAD={_os.getenv("UVICORN_RELOAD", "false")}')
    logger.info(f'PRELOAD_MODEL={_os.getenv("PRELOAD_MODEL", "false")}')

    # Предзагрузка модели по умолчанию (если включено)
    if _os.getenv('PRELOAD_MODEL', 'false').lower() in ('true', '1', 'yes'):
        logger.info('Предзагрузка модели при старте приложения...')
        try:
            default_size = _os.getenv('MODEL_SIZE', 'base')
            default_lang = _os.getenv('LANGUAGE') or None
            # Получаем анализатор, который автоматически предзагрузит модель
            _AnalyzerHolder.get(model_size=default_size, language=default_lang)
            logger.info('Модель предзагружена при старте приложения')
        except Exception as e:
            logger.warning(
                f'Не удалось предзагрузить модель при старте: {e} (будет загружена при первом запросе)'
            )


class _AnalyzerHolder:
    """Хранилище экземпляров анализатора с кэшированием по model_size и language."""

    _instances: dict[tuple[str, str | None], AdvancedPronunciationAnalyzer] = {}
    _ffmpeg_checked: bool = False
    _lock = None

    @classmethod
    def get(
        cls, model_size: str | None = None, language: str | None = None
    ) -> AdvancedPronunciationAnalyzer:
        import threading

        if not cls._ffmpeg_checked:
            import shutil

            if shutil.which('ffmpeg') is None:
                raise HTTPException(status_code=503, detail='FFmpeg не найден в PATH')
            cls._ffmpeg_checked = True

        # Инициализируем lock для thread-safety
        if cls._lock is None:
            cls._lock = threading.Lock()

        # Определяем параметры модели
        size = model_size or os.getenv('MODEL_SIZE', 'base')
        lang = language or os.getenv('LANGUAGE') or None

        # Создаем ключ для кэша
        cache_key = (size, lang)

        # Thread-safe получение/создание экземпляра
        with cls._lock:
            if cache_key not in cls._instances:
                logger.info(
                    f'Создание нового анализатора: model_size={size}, language={lang}'
                )
                analyzer = AdvancedPronunciationAnalyzer(model_size=size, language=lang)
                cls._instances[cache_key] = analyzer
                # Предзагрузка модели при создании анализатора (если включено)
                if os.getenv('PRELOAD_MODEL', 'false').lower() in ('true', '1', 'yes'):
                    logger.info(
                        f'Предзагрузка модели Whisper: model_size={size}, language={lang}'
                    )
                    try:
                        analyzer._load_model()
                        logger.info(f'Модель Whisper {size} предзагружена успешно')
                    except Exception as e:
                        logger.warning(
                            f'Не удалось предзагрузить модель: {e} (будет загружена при первом запросе)'
                        )
            else:
                logger.debug(
                    f'Использование существующего анализатора: model_size={size}, language={lang}'
                )
            return cls._instances[cache_key]


@app.get('/v1/health')
def health() -> dict[str, Any]:
    import shutil

    ffmpeg_ok = shutil.which('ffmpeg') is not None
    return {'status': 'ok' if ffmpeg_ok else 'degraded', 'ffmpeg': ffmpeg_ok}


class GrammarRequest(BaseModel):
    text: str
    allow_subject_ellipsis: bool = False
    lesson_items: list[str] | None = None
    normalize: bool = True


@app.post('/v1/text/grammar')
async def analyze_text_grammar(req: GrammarRequest):
    try:
        raw_text = req.text or ''
        if not raw_text.strip():
            raise HTTPException(status_code=400, detail='Пустой текст')
        text_in = normalize_text(raw_text) if req.normalize else raw_text
        report = analyze_text_nn(
            text_in,
            lesson_items=req.lesson_items,
            allow_subject_ellipsis=req.allow_subject_ellipsis,
        )
        # Если analyze_text_nn вернул ошибку пустого текста
        if isinstance(report, dict) and report.get('error') == 'empty_text':
            raise HTTPException(
                status_code=400, detail='Пустой текст после нормализации'
            )
        return JSONResponse({'normalized': text_in, 'report': report})
    except HTTPException:
        raise
    except RuntimeError as e:
        # Вероятно, Stanza не инициализирована
        msg = f'Грамматический анализатор недоступен: {e}'
        return JSONResponse(
            {'error': 'stanza_not_initialized', 'message': str(e)}, status_code=503
        )
    except Exception as error:
        logger.exception('Unhandled (grammar)')
        msg = f'Внутренняя ошибка: {error}'
        if _DEBUG:
            return JSONResponse({'error': str(error)}, status_code=500)
        raise HTTPException(status_code=500, detail=msg)


async def _convert_to_wav(tmp_path: str, suffix: str) -> tuple[str, str | None]:  # noqa: C901
    """Конвертирует аудиофайл в WAV формат, если необходимо."""
    import asyncio
    import subprocess as _sp
    import tempfile as _tempfile

    if suffix.lower() == '.wav':
        return tmp_path, None

    # tempfile синхронный, но это нормально для создания временного файла
    with _tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wtmp:
        wav_path = wtmp.name
    cmd = [
        'ffmpeg',
        '-y',
        '-i',
        tmp_path,
        '-ac',
        '1',
        '-ar',
        '16000',
        wav_path,
    ]
    logger.debug('ffmpeg cmd: %s', ' '.join(cmd))
    try:
        # subprocess.run в asyncio.to_thread для неблокирующего выполнения
        await asyncio.to_thread(
            _sp.run,
            cmd,
            check=True,
            stdout=_sp.DEVNULL,
            stderr=_sp.DEVNULL,
        )
        return wav_path, wav_path
    except Exception as conv_err:
        logger.exception('ffmpeg convert failed')
        raise HTTPException(
            status_code=500, detail=f'ffmpeg convert failed: {conv_err}'
        )


async def _process_audio_file(  # noqa: C901
    content: bytes,
    filename: str | None,
    analyzer: AdvancedPronunciationAnalyzer,
    temperature: float | None,
    beam_size: int | None,
    initial_prompt: str | None,
) -> tuple[Any, dict[str, Any], str, str | None]:
    """Обрабатывает аудиофайл: сохраняет, конвертирует, транскрибирует и извлекает признаки."""
    import os as _os
    import pathlib as _pathlib
    import tempfile as _tempfile

    suffix = _pathlib.Path(filename or 'audio.webm').suffix or '.webm'
    tmp_path = None
    wav_path = None
    try:
        # tempfile синхронный, но это нормально для записи файла
        with _tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        logger.debug('temp audio saved: %s', tmp_path)

        use_path, wav_path = await _convert_to_wav(tmp_path, suffix)

        t0 = time.time()
        try:
            logger.debug('Начало транскрипции: %s', use_path)
            tr = analyzer.transcribe(
                use_path,
                temperature=temperature,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
            )
            logger.debug(
                'Транскрипция завершена: %s',
                tr.text[:100] if tr.text else '(пусто)',
            )
        except Exception as transcribe_err:
            logger.exception('Ошибка при транскрипции: %s', transcribe_err)
            raise
        t1 = time.time()

        try:
            feats = extract_basic_features(use_path)
            if not feats or not isinstance(feats, dict):
                logger.warning('Признаки не извлечены или имеют неверный формат')
                feats = {}
        except Exception as feats_err:
            logger.exception('Ошибка при извлечении признаков: %s', feats_err)
            feats = {}

        t2 = time.time()
        logger.info(
            'timings_ms transcribe=%.0f features=%.0f total=%.0f',
            (t1 - t0) * 1000.0,
            (t2 - t1) * 1000.0,
            (time.time() - t0) * 1000.0,
        )
        return tr, feats, tmp_path, wav_path
    except Exception:
        # Очистка при ошибке
        _CLEANUP_MSG = 'temp cleanup failed: %s'
        if tmp_path:
            try:
                _os.remove(tmp_path)
            except Exception:
                logger.debug(_CLEANUP_MSG, tmp_path)
        if wav_path:
            try:
                _os.remove(wav_path)
            except Exception:
                logger.debug(_CLEANUP_MSG, wav_path)
        raise


@app.post('/v1/voice/analyze')
async def analyze_voice(  # noqa: C901
    file: UploadFile = File(..., description='Аудиофайл wav/mp3/m4a'),
    reference: str = Form('', description='Эталонный текст (опционально)'),
    model_size: str | None = Form(None, description='Размер модели Whisper'),
    language: str | None = Form(
        None, description='Язык распознавания или пусто для авто'
    ),
    temperature: float | None = Form(None, description='Температура декодера'),
    beam_size: int | None = Form(None, description='Размер бима'),
    initial_prompt: str | None = Form(None, description='Подсказка для модели'),
):
    try:
        try:
            max_mb = float(os.getenv('MAX_FILE_MB', '50'))
        except Exception:
            max_mb = 50.0
        content = await file.read()
        logger.info(
            'analyze start name=%s size_kb=%.1f model=%s lang=%s',
            file.filename,
            len(content) / 1024.0,
            model_size,
            language,
        )
        if not content:
            raise HTTPException(status_code=400, detail='Пустой файл')
        size_mb = len(content) / (1024 * 1024)
        if size_mb > max_mb:
            raise HTTPException(
                status_code=413,
                detail=f'Файл превышает ограничение {max_mb:.0f} МБ',
            )
        analyzer = _AnalyzerHolder.get(model_size=model_size, language=language)

        import os as _os

        _CLEANUP_MSG = 'temp cleanup failed: %s'
        tmp_path = None
        wav_path = None
        try:
            tr, feats, tmp_path, wav_path = await _process_audio_file(
                content,
                file.filename,
                analyzer,
                temperature,
                beam_size,
                initial_prompt,
            )
        finally:
            if tmp_path:
                try:
                    _os.remove(tmp_path)
                except Exception:
                    logger.debug(_CLEANUP_MSG, tmp_path)
            if wav_path:
                try:
                    _os.remove(wav_path)
                except Exception:
                    logger.debug(_CLEANUP_MSG, wav_path)

        score, details = analyzer.score(reference or '', tr, feats)
        payload = {
            'score': float(score),
            'recognized': tr.text,
            'details': details,
        }
        if 'alignment' in details:
            payload['alignment'] = details['alignment']
        return JSONResponse(payload)
    except TranscriptionError as error:
        logger.exception('TranscriptionError')
        raise HTTPException(status_code=503, detail=f'Ошибка распознавания: {error}')
    except AnalysisError as error:
        logger.exception('AnalysisError')
        raise HTTPException(status_code=400, detail=f'Ошибка анализа: {error}')
    except HTTPException:
        logger.exception('HTTPException')
        raise
    except Exception as error:
        logger.exception('Unhandled')
        msg = f'Внутренняя ошибка: {error}'
        if _DEBUG:
            return JSONResponse({'error': str(error)}, status_code=500)
        raise HTTPException(status_code=500, detail=msg)


@app.get('/v1/version')
def version() -> dict[str, Any]:
    return {
        'api': app.version,
        'model_env': {
            'MODEL_SIZE': os.getenv('MODEL_SIZE', 'base'),
            'LANGUAGE': os.getenv('LANGUAGE', ''),
        },
    }


@app.get('/v1/cache/status')
def cache_status() -> dict[str, Any]:
    """Проверка состояния кэша анализаторов."""
    import threading

    # Инициализируем lock, если он еще не инициализирован
    if _AnalyzerHolder._lock is None:
        _AnalyzerHolder._lock = threading.Lock()

    with _AnalyzerHolder._lock:
        cached_keys = list(_AnalyzerHolder._instances.keys())
        cache_info = []
        for key in cached_keys:
            size, lang = key
            analyzer = _AnalyzerHolder._instances[key]
            model_loaded = analyzer._model is not None
            cache_info.append(
                {
                    'model_size': size,
                    'language': lang,
                    'model_loaded': model_loaded,
                }
            )
    return {
        'cached_analyzers': len(cached_keys),
        'analyzers': cache_info,
    }


if __name__ == '__main__':
    import uvicorn

    # Отключаем reload в продакшене для предотвращения перезапуска во время загрузки моделей
    # Reload может убить процесс во время инициализации Stanza (15+ секунд)
    enable_reload = os.getenv('UVICORN_RELOAD', 'false').lower() in ('true', '1', 'yes')
    uvicorn.run(
        'main:app',
        host='127.0.0.1',
        port=8000,
        reload=enable_reload,
        app_dir='.',
    )

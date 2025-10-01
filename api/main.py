from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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


class _AnalyzerHolder:
    _instance: AdvancedPronunciationAnalyzer | None = None
    _ffmpeg_checked: bool = False

    @classmethod
    def get(
        cls, model_size: str | None = None, language: str | None = None
    ) -> AdvancedPronunciationAnalyzer:
        if not cls._ffmpeg_checked:
            import shutil

            if shutil.which('ffmpeg') is None:
                raise HTTPException(status_code=503, detail='FFmpeg не найден в PATH')
            cls._ffmpeg_checked = True
        if cls._instance is None:
            size = model_size or os.getenv('MODEL_SIZE', 'base')
            lang = language or os.getenv('LANGUAGE') or None
            cls._instance = AdvancedPronunciationAnalyzer(
                model_size=size, language=lang
            )
        return cls._instance


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


@app.post('/v1/voice/analyze')
async def analyze_voice(
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
    req_started = time.time()
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

        # Сохраняем во временный файл с исходным суффиксом
        import os as _os
        import pathlib as _pathlib
        import tempfile as _tempfile

        suffix = _pathlib.Path(file.filename or 'audio.webm').suffix or '.webm'
        tmp_path = None
        wav_path = None
        try:
            with _tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            logger.debug('temp audio saved: %s', tmp_path)

            # Конвертируем в WAV для стабильной обработки webm/opus/mp3
            use_path = tmp_path
            if suffix.lower() != '.wav':
                import subprocess as _sp

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
                    _sp.run(cmd, check=True, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
                    use_path = wav_path
                except Exception as conv_err:
                    logger.exception('ffmpeg convert failed')
                    raise HTTPException(
                        status_code=500, detail=f'ffmpeg convert failed: {conv_err}'
                    )

            t0 = time.time()
            tr = analyzer.transcribe(
                use_path,
                temperature=temperature,
                beam_size=beam_size,
                initial_prompt=initial_prompt,
            )
            t1 = time.time()
            feats = extract_basic_features(use_path)
            t2 = time.time()
            logger.info(
                'timings_ms transcribe=%.0f features=%.0f total=%.0f',
                (t1 - t0) * 1000.0,
                (t2 - t1) * 1000.0,
                (time.time() - req_started) * 1000.0,
            )
        finally:
            if tmp_path:
                try:
                    _os.remove(tmp_path)
                except Exception:
                    logger.debug('temp cleanup failed: %s', tmp_path)
                    pass
            if wav_path:
                try:
                    _os.remove(wav_path)
                except Exception:
                    logger.debug('temp cleanup failed: %s', wav_path)
                    pass
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

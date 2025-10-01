from __future__ import annotations

import io
import os
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def extract_basic_features(file) -> dict[str, Any]:
    data: bytes | None = None
    if hasattr(file, 'getvalue'):
        data = file.getvalue()
    elif hasattr(file, 'read'):
        data = file.read()
    elif isinstance(file, bytes | bytearray):
        data = bytes(file)
    elif isinstance(file, str | os.PathLike):
        value_y, sr = librosa.load(str(file), sr=None, mono=True)
        value_y = librosa.util.normalize(value_y)
        return _features_from_wave(value_y, sr)
    else:
        raise TypeError(
            'extract_basic_features: ожидаются bytes/UploadedFile/путь к файлу'
        )
    value_y, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=True)
    value_y = np.mean(value_y, axis=1).astype(np.float32)
    value_y = librosa.util.normalize(value_y)
    return _features_from_wave(value_y, int(sr))


def _features_from_wave(value_y: np.ndarray, sr: int) -> dict[str, Any]:
    rms = float(np.mean(librosa.feature.rms(y=value_y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(value_y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=value_y, sr=sr)))
    zcr_dev = abs(0.1 - zcr)
    cent_norm = float(centroid / max(sr, 1))
    cent_dev = abs(0.2 - cent_norm)
    audio_quality = float(np.clip(1.0 - (zcr_dev + cent_dev), 0.0, 1.0))
    centroid_norm = float(centroid / sr)
    return {
        'summary': {
            'rms': rms,
            'zcr': zcr,
            'centroid': centroid,
            'audio_quality': audio_quality,
            'centroid_norm': centroid_norm,
        }
    }

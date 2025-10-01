from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class SemanticResult:
    provider: str
    score: float
    details: dict[str, Any]


class SemanticEvaluator:
    def __init__(self, provider: str = 'local-sbert'):
        self.provider = provider.lower()

    def evaluate(
        self, text: str, reference: str | None = None, criteria: str | None = None
    ) -> SemanticResult:
        if self.provider == 'local-sbert':
            return self._local_sbert(text, reference or criteria or '')
        if self.provider == 'openai':
            return self._openai_llm(text, reference, criteria)
        if self.provider == 'hf-api':
            return self._hf_api(text, reference or criteria or '')
        raise ValueError(f'Unknown provider: {self.provider}')

    def _local_sbert(self, text: str, ref: str) -> SemanticResult:
        try:
            if not ref.strip():
                return SemanticResult(
                    provider='local-sbert',
                    score=0.0,
                    details={
                        'error': 'reference_empty',
                        'hint': "Заполните поле 'Эталон/критерии'.",
                    },
                )
            from sentence_transformers import SentenceTransformer, util

            model_name = os.getenv(
                'SBERT_MODEL', 'paraphrase-multilingual-MiniLM-L12-v2'
            )
            model = SentenceTransformer(model_name)
            emb = model.encode(
                [text, ref], convert_to_tensor=True, normalize_embeddings=True
            )
            cos = float(util.cos_sim(emb[0], emb[1]).item())
            score = max(0.0, min(100.0, (cos + 1.0) * 50.0))
            return SemanticResult(
                provider='local-sbert',
                score=score,
                details={'cosine': cos, 'model': model_name},
            )
        except Exception as error:
            return SemanticResult(
                provider='local-sbert', score=0.0, details={'error': str(error)}
            )

    def _openai_llm(
        self, text: str, reference: str | None, criteria: str | None
    ) -> SemanticResult:
        try:
            import json
            import re

            from openai import OpenAI

            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise RuntimeError('OPENAI_API_KEY is not set')
            client = OpenAI(api_key=api_key)
            rubric = criteria or (
                f'Оцени семантическую близость к эталону (0..100). Эталон: {reference}\nТекст:'
                if reference
                else 'Оцени смысловую связность текста (0..100) и обоснуй:'
            )
            prompt = f"{rubric}\n{text}\nВерни JSON: {{'score': <0..100>, 'explanation': '...'}}"
            resp = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            count_m = re.search('\\{.*\\}', content, re.S)
            data = {}
            if count_m:
                try:
                    data = json.loads(count_m.group(0).replace("'", '"'))
                except Exception:
                    data = {'raw': content}
            score = float(data.get('score', 0.0))
            return SemanticResult(
                provider='openai',
                score=max(0, min(100, score)),
                details=data or {'raw': content},
            )
        except Exception as error:
            return SemanticResult(
                provider='openai', score=0.0, details={'error': str(error)}
            )

    def _hf_api(self, text: str, ref: str) -> SemanticResult:
        try:
            import numpy as np
            import requests

            token = os.getenv('HF_API_TOKEN')
            if not token:
                raise RuntimeError('HF_API_TOKEN is not set')
            model = os.getenv(
                'HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2'
            )
            headers = {'Authorization': f'Bearer {token}'}

            def embed(text: str):
                var_r = requests.post(
                    f'https://api-inference.huggingface.co/pipeline/feature-extraction/{model}',
                    headers=headers,
                    json={'inputs': text},
                )
                var_r.raise_for_status()
                return var_r.json()

            var_a, var_b = (
                np.array(embed(text)).mean(axis=0),
                np.array(embed(ref)).mean(axis=0),
            )
            cos = float(
                np.dot(var_a, var_b)
                / (np.linalg.norm(var_a) * np.linalg.norm(var_b) + 1e-09)
            )
            score = max(0.0, min(100.0, cos * 100.0))
            return SemanticResult(
                provider='hf-api', score=score, details={'cosine': cos, 'model': model}
            )
        except Exception as error:
            return SemanticResult(
                provider='hf-api', score=0.0, details={'error': str(error)}
            )

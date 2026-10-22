from __future__ import annotations

import hashlib

from harness.config import EmbeddingConfig
from harness.factories import register_embedding


@register_embedding("stub")
class StubEmbeddingModel:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        self._model_name = cfg.model_name_or_path
        self._dimension = 128

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        results = []
        for text in texts:
            seed = hashlib.sha256(text.encode()).digest()
            vec = []
            for i in range(self._dimension):
                byte_val = seed[i % len(seed)]
                vec.append((byte_val / 255.0) * 2 - 1)
            norm = sum(x * x for x in vec) ** 0.5
            vec = [x / norm for x in vec]
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model_name

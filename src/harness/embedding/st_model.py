from __future__ import annotations

from harness.config import EmbeddingConfig
from harness.factories import register_embedding


@register_embedding("sentence_transformer")
class SentenceTransformerModel:
    def __init__(self, cfg: EmbeddingConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(cfg.model_name_or_path, device=cfg.device)
        self._model_name = cfg.model_name_or_path
        self._batch_size = cfg.batch_size

    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size or self._batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

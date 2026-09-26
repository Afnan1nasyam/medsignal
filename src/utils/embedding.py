"""Biomedical sentence-embedding model wrapper.

Lazily loads a ``sentence-transformers`` model (``settings.EMBEDDING_MODEL``,
BioLORD-2023-C by default) and exposes helpers to embed single texts or
batches. The model is loaded on first use, so importing this module performs no
network activity — the actual download/load only happens when an embed method
is called (this dev machine's proxy blocks HuggingFace downloads).
"""

from typing import TYPE_CHECKING

from loguru import logger

from src.config import settings

if TYPE_CHECKING:  # import only for type checkers, never at runtime import
    from sentence_transformers import SentenceTransformer


class BiomedEmbeddingModel:
    """Lazy wrapper around a biomedical sentence-transformers model."""

    def __init__(self, model_name: str | None = None) -> None:
        """Initialize without loading the model.

        Args:
            model_name: Model identifier. Defaults to ``settings.EMBEDDING_MODEL``.
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model: "SentenceTransformer | None" = None

    def _load(self) -> "SentenceTransformer":
        """Load the underlying model on first use (triggers download if needed)."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model: {}", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded: {}", self.model_name)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text into a dense vector.

        Args:
            text: Input text.

        Returns:
            The embedding as a list of floats.
        """
        model = self._load()
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: Input texts.
            batch_size: Encoding batch size.

        Returns:
            A list of embedding vectors, one per input text.
        """
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def get_embedding_dim(self) -> int:
        """Return the model's output dimensionality (loads the model if needed)."""
        model = self._load()
        return model.get_sentence_embedding_dimension()


# Module-level singleton.
_embedding_model: BiomedEmbeddingModel | None = None


def get_embedding_model() -> BiomedEmbeddingModel:
    """Return the process-wide singleton :class:`BiomedEmbeddingModel`."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = BiomedEmbeddingModel()
    return _embedding_model

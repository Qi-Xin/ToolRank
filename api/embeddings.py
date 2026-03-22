"""
Embedding providers for semantic search.
- MockEmbeddingProvider: deterministic, no dependencies (for tests)
- SentenceTransformerProvider: local model, no API key needed
- OpenAIEmbeddingProvider: best quality, requires OPENAI_API_KEY
"""

import hashlib
import math
import os
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def dim(self) -> int: ...


class MockEmbeddingProvider:
    """Deterministic embeddings based on text hash. For testing only."""

    DIM = 64

    def embed(self, text: str) -> list[float]:
        text = text.lower().strip()
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        result = []
        for i in range(self.DIM):
            seed = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
            val = (seed & 0xFFFF) / 0xFFFF * 2 - 1
            result.append(val)
        return _normalize(result)

    def dim(self) -> int:
        return self.DIM


class FastEmbedProvider:
    """Uses fastembed (ONNX, no PyTorch, ~130MB model). Install: pip install fastembed"""

    MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self):
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=self.MODEL)
        except ImportError:
            raise ImportError("Run: pip install fastembed")

    def embed(self, text: str) -> list[float]:
        return list(next(self._model.embed([text])))

    def dim(self) -> int:
        return 384


class SentenceTransformerProvider:
    """Uses sentence-transformers (local, no API key). Install: pip install sentence-transformers"""

    MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL)
        except ImportError:
            raise ImportError("Run: pip install sentence-transformers")

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True).tolist()
        return vec

    def dim(self) -> int:
        return 384


class OpenAIEmbeddingProvider:
    """Uses OpenAI text-embedding-3-small. Requires OPENAI_API_KEY."""

    MODEL = "text-embedding-3-small"
    DIM = 1536

    def __init__(self):
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        except ImportError:
            raise ImportError("Run: pip install openai")
        except KeyError:
            raise EnvironmentError("Set OPENAI_API_KEY environment variable")

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(input=text, model=self.MODEL)
        return resp.data[0].embedding

    def dim(self) -> int:
        return self.DIM


def get_default_provider() -> EmbeddingProvider:
    """Auto-select provider based on available dependencies."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIEmbeddingProvider()
        except Exception:
            pass
    try:
        return FastEmbedProvider()
    except Exception:
        pass
    try:
        return SentenceTransformerProvider()
    except Exception:
        pass
    return MockEmbeddingProvider()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]

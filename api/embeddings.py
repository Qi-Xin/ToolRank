"""
Embedding providers for semantic search.
- TFIDFEmbeddingProvider: no downloads, no API key, word-level semantics (default)
- FastEmbedProvider: ONNX model, no API key, real semantic search
- SentenceTransformerProvider: local model, no API key needed
- OpenAIEmbeddingProvider: best quality, requires OPENAI_API_KEY
- MockEmbeddingProvider: deterministic hash-based, for tests only
"""

import hashlib
import math
import os
import re
import json
from collections import Counter
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> list[float]: ...
    def dim(self) -> int: ...


# Common English stop words to ignore in TF-IDF
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","has","have","had",
    "it","its","this","that","these","those","can","will","would","could",
    "should","may","might","do","does","did","not","no","nor","so","yet",
    "both","either","each","any","all","as","if","then","than","when",
    "which","who","what","how","your","you","their","they","we","our",
}


class TFIDFEmbeddingProvider:
    """
    Lightweight TF-IDF based embeddings. No downloads, no API key.
    Vocabulary is built from a fixed seed corpus + updated at runtime.
    Captures keyword-level semantic similarity — good enough for tool search.
    """

    DIM = 256  # vocabulary size

    # Seed vocabulary covering common agent tool concepts
    _SEED_TERMS = [
        "search", "web", "internet", "browse", "scrape", "crawl", "fetch",
        "github", "code", "repository", "pull", "request", "issue", "commit",
        "file", "filesystem", "read", "write", "directory", "path",
        "database", "sql", "query", "postgres", "sqlite", "data",
        "slack", "message", "channel", "notification", "communication",
        "browser", "automation", "click", "form", "screenshot", "puppeteer",
        "memory", "store", "cache", "persist", "knowledge", "retrieve",
        "think", "reason", "plan", "sequential", "step", "chain",
        "map", "location", "place", "geocode", "direction", "navigate",
        "email", "calendar", "document", "spreadsheet", "notion",
        "monitor", "track", "log", "metric", "alert", "dashboard",
        "api", "endpoint", "rest", "http", "request", "response",
        "image", "pdf", "convert", "extract", "parse", "format",
        "deploy", "cloud", "server", "container", "docker", "aws",
        "linear", "issue", "task", "project", "team", "workflow",
        "language", "model", "llm", "agent", "tool", "skill", "mcp",
        "text", "content", "generate", "summarize", "analyze", "classify",
        "url", "link", "page", "html", "markdown", "json", "xml",
        "fast", "slow", "latency", "performance", "efficient", "reliable",
        "embed", "vector", "semantic", "similarity", "rank", "score",
    ]

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._idf: list[float] = []
        self._docs: list[list[str]] = []
        # Build initial vocab from seed terms
        for i, term in enumerate(self._SEED_TERMS[:self.DIM]):
            self._vocab[term] = i
        self._idf = [1.0] * self.DIM

    def _tokenize(self, text: str) -> list[str]:
        tokens = re.findall(r"[a-z]+", text.lower())
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]

    def _tf(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {}
        counts = Counter(tokens)
        total = len(tokens)
        return {t: c / total for t, c in counts.items()}

    def _to_vector(self, tokens: list[str]) -> list[float]:
        tf = self._tf(tokens)
        vec = [0.0] * self.DIM
        for term, weight in tf.items():
            if term in self._vocab:
                idx = self._vocab[term]
                vec[idx] = weight * self._idf[idx]
        return _normalize(vec)

    def fit(self, texts: list[str]):
        """Update IDF weights from a corpus of tool descriptions."""
        if not texts:
            return
        tokenized = [self._tokenize(t) for t in texts]
        N = len(tokenized)
        # Add new terms to vocab
        for tokens in tokenized:
            for t in tokens:
                if t not in self._vocab and len(self._vocab) < self.DIM:
                    self._vocab[t] = len(self._vocab)
        # Recompute IDF
        idf = [1.0] * self.DIM
        for idx, term in enumerate([t for t, i in sorted(self._vocab.items(), key=lambda x: x[1])]):
            df = sum(1 for tokens in tokenized if term in tokens)
            if df > 0:
                idf[idx] = math.log((N + 1) / (df + 1)) + 1
        self._idf = idf
        self._docs = tokenized

    def embed(self, text: str) -> list[float]:
        tokens = self._tokenize(text)
        return self._to_vector(tokens)

    def dim(self) -> int:
        return self.DIM


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
    """Auto-select best available provider."""
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
    # Always-available fallback: TF-IDF (word-level semantics, no downloads)
    return TFIDFEmbeddingProvider()


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

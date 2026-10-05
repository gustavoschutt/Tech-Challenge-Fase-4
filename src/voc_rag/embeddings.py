"""Representação vetorial dos documentos e das perguntas.

Dois backends:
- "sentence-transformers" (padrão): modelo neural multilíngue pré-treinado
  (ex.: intfloat/multilingual-e5-base). Captura sinônimos e paráfrases
  ("não chegou" ~ "nunca recebi").
- "lsa": TF-IDF + SVD truncado (Latent Semantic Analysis) treinado no próprio
  corpus. 100% offline; serve como BASELINE clássico na avaliação e como
  alternativa quando não há acesso ao repositório de modelos.

Todos os vetores são normalizados (norma L2 = 1): produto interno = cosseno.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .config import Settings


def quiet_hf() -> None:
    """Silencia avisos informativos do transformers (best effort: se a API
    mudar, nada quebra). As barras de progresso continuam, para quem baixa os
    modelos pela primeira vez; os notebooks as desligam na primeira célula."""
    try:
        from transformers.utils import logging as tl

        tl.set_verbosity_error()
    except Exception:  # pragma: no cover
        pass


class Embedder:
    def __init__(self, settings: Settings):
        self.s = settings
        self.backend = settings.embedding_backend
        self._model = None      # sentence-transformers
        self._lsa = None        # (vectorizer, svd)

    # ------------------------------------------------------------------ fit
    def fit(self, texts: list[str]) -> "Embedder":
        """Necessário apenas para o backend LSA (aprende o vocabulário)."""
        if self.backend == "lsa":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            from .text_utils import tokenize_lexical

            vec = TfidfVectorizer(tokenizer=tokenize_lexical, lowercase=False, token_pattern=None,
                                  ngram_range=(1, 2), min_df=2 if len(texts) > 1000 else 1,
                                  max_df=0.5 if len(texts) > 1000 else 1.0, sublinear_tf=True)
            x = vec.fit_transform(texts)
            n_comp = max(2, min(256, x.shape[1] - 1, x.shape[0] - 1))
            svd = TruncatedSVD(n_components=n_comp, random_state=self.s.seed)
            svd.fit(x)
            self._lsa = (vec, svd)
        return self

    def save(self, directory: Path) -> None:
        if self.backend == "lsa" and self._lsa is not None:
            with open(directory / "lsa.pkl", "wb") as f:
                pickle.dump(self._lsa, f)

    def load(self, directory: Path) -> "Embedder":
        if self.backend == "lsa":
            with open(directory / "lsa.pkl", "rb") as f:
                self._lsa = pickle.load(f)
        return self

    # ---------------------------------------------------------------- encode
    def _st_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            quiet_hf()
            self._model = SentenceTransformer(self.s.embedding_model, device=None)
        return self._model

    def _encode(self, texts: list[str], prefix: str, show_progress: bool = False) -> np.ndarray:
        if self.backend == "lsa":
            vec, svd = self._lsa
            v = svd.transform(vec.transform(texts)).astype(np.float32)
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return v / norms
        model = self._st_model()
        v = model.encode([prefix + t for t in texts], batch_size=self.s.embedding_batch_size,
                         normalize_embeddings=True, show_progress_bar=show_progress,
                         convert_to_numpy=True)
        return v.astype(np.float32)

    def encode_documents(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        return self._encode(texts, self.s.passage_prefix, show_progress)

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.s.query_prefix)[0]

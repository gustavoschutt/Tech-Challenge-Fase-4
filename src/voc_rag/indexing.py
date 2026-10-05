"""Indexação: vetores (FAISS) + índice léxico (BM25).

- FAISS IndexFlatIP: busca EXATA por produto interno (= cosseno, vetores
  normalizados). Com ~42 mil documentos a busca exata custa milissegundos,
  então não há motivo para aceitar o erro de um índice aproximado
  (HNSW/IVF só compensaria na casa dos milhões de vetores).
- Filtros por metadados são aplicados ANTES da busca vetorial, via
  IDSelector do FAISS (pré-filtragem): o top-k já sai do subconjunto certo.
- BM25 (rank_bm25) sobre tokens normalizados + stemming Snowball (português). É reconstruído
  ao carregar (leva segundos), o que evita artefatos binários frágeis.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Settings
from .data_prep import load_documents
from .embeddings import Embedder
from .text_utils import tokenize_lexical


def build_index(settings: Settings, verbose: bool = True) -> dict:
    docs = load_documents(settings)
    out = settings.index_dir
    out.mkdir(parents=True, exist_ok=True)
    texts = docs["texto"].tolist()

    t0 = time.time()
    embedder = Embedder(settings).fit(texts)
    vectors = embedder.encode_documents(texts, show_progress=verbose)
    embedder.save(out)
    t_embed = time.time() - t0

    np.save(out / "embeddings.npy", vectors)
    try:
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        faiss.write_index(index, str(out / "faiss.index"))
        backend = "faiss.IndexFlatIP"
    except ImportError:  # pragma: no cover
        backend = "numpy (FAISS indisponível)"

    # Os ids do índice são as posições em documentos.parquet; gravamos o
    # review_id na mesma ordem para validar a correspondência ao carregar.
    docs[["review_id"]].to_parquet(out / "ids.parquet", index=False)
    meta = {
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.embedding_model if settings.embedding_backend != "lsa" else "tfidf+svd(256)",
        "n_documentos": int(len(docs)),
        "dimensao": int(vectors.shape[1]),
        "indice_vetorial": backend,
        "segundos_embeddings": round(t_embed, 1),
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


@dataclass
class KnowledgeIndex:
    settings: Settings
    docs: pd.DataFrame
    vectors: np.ndarray
    embedder: Embedder
    faiss_index: object | None
    bm25: object
    meta: dict
    doc_freq: dict          # termo (stem) -> nº de documentos que o contêm

    @classmethod
    def load(cls, settings: Settings, verbose: bool = False) -> "KnowledgeIndex":
        d = settings.index_dir
        if not (d / "embeddings.npy").exists():
            raise FileNotFoundError(f"Índice não encontrado em {d}. Rode: python scripts/02_construir_indice.py")
        docs = load_documents(settings)
        ids = pd.read_parquet(d / "ids.parquet")["review_id"]
        if len(ids) != len(docs) or not (ids.values == docs["review_id"].values).all():
            raise RuntimeError("Índice desatualizado em relação a documentos.parquet. Reconstrua o índice.")
        vectors = np.load(d / "embeddings.npy")
        embedder = Embedder(settings).load(d)
        faiss_index = None
        try:
            import faiss

            faiss_index = faiss.read_index(str(d / "faiss.index"))
        except Exception:  # pragma: no cover
            pass
        from rank_bm25 import BM25Okapi

        t0 = time.time()
        tokenized = [tokenize_lexical(t) for t in docs["texto"]]
        bm25 = BM25Okapi(tokenized)
        doc_freq: dict[str, int] = {}
        for toks in tokenized:
            for tok in set(toks):
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        if verbose:
            print(f"BM25 reconstruído em {time.time() - t0:.1f}s")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        return cls(settings, docs, vectors, embedder, faiss_index, bm25, meta, doc_freq)

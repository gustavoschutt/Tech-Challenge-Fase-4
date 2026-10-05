"""Etapa 2 — Gera embeddings e o índice vetorial (artifacts/indice/<modelo>/).

Uso:
  python scripts/02_construir_indice.py                       # modelo do .env
  python scripts/02_construir_indice.py --modelo intfloat/multilingual-e5-small
  python scripts/02_construir_indice.py --backend lsa         # baseline offline
"""
import argparse

import _bootstrap  # noqa: F401

from voc_rag.config import get_settings
from voc_rag.indexing import build_index

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sentence-transformers", "lsa"])
    ap.add_argument("--modelo")
    a = ap.parse_args()
    over = {}
    if a.backend:
        over["embedding_backend"] = a.backend
    if a.modelo:
        over["embedding_model"] = a.modelo
    build_index(get_settings(**over))

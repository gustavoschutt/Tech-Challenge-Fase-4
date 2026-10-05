"""Etapa 1 — Constrói a base de conhecimento (artifacts/base/).

Uso:  python scripts/01_preparar_base.py
"""
import _bootstrap  # noqa: F401

from voc_rag.config import get_settings
from voc_rag.data_prep import build_knowledge_base

if __name__ == "__main__":
    build_knowledge_base(get_settings())

"""Etapa 7 (opcional) — Compara representações vetoriais no conjunto de
teste: baseline clássico (LSA = TF-IDF + SVD) x modelos neurais.

Para cada modelo: constrói o índice (se ainda não existir) e mede a
precisão@k "silver" da recuperação densa pura e da híbrida, além do tempo
de indexação. Grava eval/resultados/comparacao_embeddings.csv.

Uso:
  python scripts/07_comparar_embeddings.py
  python scripts/07_comparar_embeddings.py --modelos lsa sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 intfloat/multilingual-e5-base
"""
import argparse
import json

import _bootstrap  # noqa: F401
import pandas as pd

from voc_rag.config import ROOT, get_settings
from voc_rag.evaluation import load_questions, retrieval_comparison
from voc_rag.indexing import build_index
from voc_rag.pipeline import VoCRAG

PADRAO = ["lsa",
          "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
          "intfloat/multilingual-e5-small",
          "intfloat/multilingual-e5-base"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=PADRAO)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    qs = load_questions(ROOT / "eval" / "perguntas_teste.json")
    rows = []
    for m in a.modelos:
        over = {"embedding_backend": "lsa"} if m == "lsa" else {"embedding_backend": "sentence-transformers",
                                                                 "embedding_model": m}
        s = get_settings(reranker_model="", **over)  # sem reranker: isola o efeito do embedding
        if not (s.index_dir / "embeddings.npy").exists():
            build_index(s)
        meta = json.loads((s.index_dir / "meta.json").read_text(encoding="utf-8"))
        rag = VoCRAG(s, load_llm=False)
        rec = retrieval_comparison(rag, qs, k=a.k)
        for modo, g in rec.groupby("modo"):
            if modo == "bm25":
                continue
            rows.append({"modelo": meta["embedding_model"], "dimensao": meta["dimensao"],
                         "segundos_indexacao": meta["segundos_embeddings"], "modo": modo,
                         f"precisao@{a.k}": round(g[f"precisao@{a.k}"].mean(), 3), "mrr": round(g["rr"].mean(), 3)})
        print(pd.DataFrame(rows).tail(2).to_string(index=False))
    out = ROOT / "eval" / "resultados" / "comparacao_embeddings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\n{pd.DataFrame(rows).to_string(index=False)}\n\nGravado em {out}")

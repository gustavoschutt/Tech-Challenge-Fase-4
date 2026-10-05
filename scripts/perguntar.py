"""Faz uma pergunta ao sistema pela linha de comando.

Uso:
  python scripts/perguntar.py "O que os clientes relatam sobre atrasos na entrega?"
  python scripts/perguntar.py "..." --modo rag      # força o RAG puro
  python scripts/perguntar.py "..." --json          # saída estruturada
"""
import argparse
import json

import _bootstrap  # noqa: F401

from voc_rag.pipeline import VoCRAG

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pergunta")
    ap.add_argument("--modo", default="auto", choices=["auto", "rag", "analitico"])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = VoCRAG().ask(a.pergunta, modo=a.modo)
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2, default=str) if a.json else r.to_markdown())

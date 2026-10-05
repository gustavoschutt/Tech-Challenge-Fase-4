"""Etapa 4 — Calibra os limiares do gate de abstenção no conjunto de
CALIBRAÇÃO (eval/perguntas_calibracao.json) e grava
artifacts/limiares__<modelo>.json. Não usa LLM.

Uso:  python scripts/04_calibrar_limiares.py
"""
import json

import _bootstrap  # noqa: F401

from voc_rag.config import ROOT, get_settings
from voc_rag.evaluation import calibrate, load_questions, verify_absence
from voc_rag.pipeline import VoCRAG

if __name__ == "__main__":
    s = get_settings()
    rag = VoCRAG(s, load_llm=False)
    qs = load_questions(ROOT / "eval" / "perguntas_calibracao.json")
    print(verify_absence(qs, rag.index.docs).to_string(index=False))
    res = calibrate(rag, qs)
    s.thresholds_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "features"}, ensure_ascii=False, indent=2))
    print(f"Limiares gravados em {s.thresholds_path}")

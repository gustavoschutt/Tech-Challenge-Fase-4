"""Etapa 3 (camada analítica) — Rotula cada avaliação com os temas da
taxonomia (themes.py) e grava artifacts/base/temas.parquet + um resumo.

Também exporta eval/auditoria_temas.csv: amostra estratificada (até 15 por
tema, semente fixa) para auditoria manual da precisão das regras.

Uso:  python scripts/03_construir_temas.py
"""
import json

import _bootstrap  # noqa: F401
import pandas as pd

from voc_rag.config import ROOT, get_settings
from voc_rag.data_prep import load_documents
from voc_rag.evaluation import write_annotation_csv
from voc_rag.themes import NOTAS_ELOGIO, NOTAS_PROBLEMA, TAXONOMIA, label_texts

if __name__ == "__main__":
    s = get_settings()
    docs = load_documents(s)
    lab = label_texts(docs["texto"])
    out = pd.concat([docs[["review_id"]], lab], axis=1)
    out.to_parquet(s.base_dir / "temas.parquet", index=False)

    sc = docs["review_score"]
    resumo = {}
    for pol, (lo, hi) in (("problema", NOTAS_PROBLEMA), ("elogio", NOTAS_ELOGIO)):
        base = (sc >= lo) & (sc <= hi)
        ids = [t.id for t in TAXONOMIA if t.polaridade == pol]
        resumo[pol] = {
            "base_avaliacoes_com_texto": int(base.sum()),
            "cobertura": round(float(lab.loc[base, ids].any(axis=1).mean()), 4),
            "por_tema": {t: int(lab.loc[base, t].sum()) for t in ids},
        }
    (s.base_dir / "temas_resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resumo, ensure_ascii=False, indent=2))

    # Amostra para auditoria (não sobrescreve anotações já feitas).
    audit_path = ROOT / "eval" / "auditoria_temas.csv"
    if not audit_path.exists():
        rows = []
        for t in TAXONOMIA:
            lo, hi = NOTAS_PROBLEMA if t.polaridade == "problema" else NOTAS_ELOGIO
            pool = docs[lab[t.id] & (sc >= lo) & (sc <= hi)]
            for _, r in pool.sample(min(15, len(pool)), random_state=s.seed).iterrows():
                rows.append({"tema": t.id, "nome_do_tema": t.nome, "review_id": r["review_id"],
                             "nota": r["review_score"], "texto": r["texto"], "correto": "", "anotador": ""})
        write_annotation_csv(pd.DataFrame(rows), audit_path)
        print(f"Amostra de auditoria gravada em {audit_path} (preencha a coluna 'correto' com 1/0).")

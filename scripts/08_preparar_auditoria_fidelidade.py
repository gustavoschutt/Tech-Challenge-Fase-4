"""Etapa 8 (auditoria humana) — Gera eval/auditoria_fidelidade.csv a partir
das respostas da avaliação (eval/resultados/<modelo>/respostas.json).

A verificação automática confere se cada citação [E#] existe. Ela não confere
se a evidência citada SUSTENTA a frase. Esta planilha lista cada frase com
citação das respostas de status "ok", ao lado do texto das avaliações citadas,
para uma pessoa anotar a coluna `sustentada`:
  1 = a evidência citada sustenta a frase
  P = sustenta em parte (exagera, generaliza ou junta coisas diferentes)
  0 = não sustenta
O resultado é calculado no notebook 03.

Não sobrescreve uma planilha que já tenha anotações (use --forcar para isso).

Uso:  python scripts/08_preparar_auditoria_fidelidade.py
"""
import argparse
import json
import re

import _bootstrap  # noqa: F401
import pandas as pd

from voc_rag.config import ROOT, get_settings
from voc_rag.evaluation import load_questions, read_annotation_csv, write_annotation_csv
from voc_rag.generation import is_header

CIT = re.compile(r"\[(E\d+)\]")
# Marcador de lista no início ("- ...") faz o Excel tratar a célula como fórmula
# (#NOME?) e perder o texto; o marcador não faz parte da afirmação.
BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--forcar", action="store_true", help="sobrescreve mesmo com anotações")
    a = ap.parse_args()
    s = get_settings()
    out = ROOT / "eval" / "auditoria_fidelidade.csv"
    if out.exists() and not a.forcar and (read_annotation_csv(out)["sustentada"].str.strip() != "").any():
        raise SystemExit(f"{out} já tem anotações; nada foi alterado (use --forcar para recriar).")

    respostas = json.loads((ROOT / "eval" / "resultados" / s.model_slug / "respostas.json").read_text(encoding="utf-8"))
    ids = {q["pergunta"]: q["id"] for q in load_questions(ROOT / "eval" / "perguntas_teste.json")}
    rows, sem_citacao = [], 0
    for r in respostas:
        if r["status"] != "ok" or not r["resposta"]:
            continue
        textos = {e.get("rotulo"): e["texto"] for e in r["evidencias"] if e.get("rotulo")}
        for linha in r["resposta"].splitlines():
            linha = linha.strip()
            if not linha or is_header(linha):
                continue
            cits = list(dict.fromkeys(CIT.findall(linha)))
            if not cits:
                sem_citacao += 1  # no modo analítico, a linha de resumo traz só números da tabela
                continue
            rows.append({"id": ids.get(r["pergunta"], ""), "modo": r["modo"], "pergunta": r["pergunta"],
                         "frase": BULLET.sub("", CIT.sub("", linha)).strip(), "citacoes": " ".join(cits),
                         "evidencias_citadas": " || ".join(f"[{c}] {textos.get(c, '(rótulo não encontrado)')}"
                                                           for c in cits),
                         "sustentada": "", "observacao": "", "anotador": ""})
    write_annotation_csv(pd.DataFrame(rows), out)
    print(f"{len(rows)} frases com citação gravadas em {out} "
          f"({sem_citacao} linhas sem citação ficaram de fora). Preencha 'sustentada' com 1, P ou 0.")

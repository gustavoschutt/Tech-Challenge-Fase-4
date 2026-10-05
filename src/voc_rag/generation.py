"""Geração da resposta + verificação pós-geração (grounding).

Depois que o LLM responde, o texto é auditado:
1. Se o LLM declarou SEM_EVIDENCIA_SUFICIENTE -> status sem_evidencias.
2. Citações [E#] que não existem no contexto são removidas e registradas.
3. Toda linha de conteúdo precisa de ao menos uma citação válida; linhas sem
   citação são REMOVIDAS (e registradas no diagnóstico).
4. Se nada citado sobrar -> status resposta_nao_fundamentada (a resposta é
   descartada e só as evidências são exibidas).
5. Expressões quantitativas ("a maioria", "%") são sinalizadas: o MVP não
   tem base estatística para elas (a camada analítica tem).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .prompts import SEM_EVIDENCIA_TOKEN, SYSTEM_RAG, USER_RAG

_BRACKET = re.compile(r"\[([^\]]*E\d+[^\]]*)\]")
_LABEL = re.compile(r"E(\d+)")
_QUANT = re.compile(r"(\d+\s?%|\ba maioria\b|\bmaior parte\b|\bmuitos clientes\b|\bquase todos\b|\bfrequentemente\b|\bgrande parte\b)", re.I)


@dataclass
class GroundingReport:
    citacoes_validas: list[str] = field(default_factory=list)
    citacoes_invalidas: list[str] = field(default_factory=list)
    linhas_removidas_sem_citacao: list[str] = field(default_factory=list)
    expressoes_quantitativas: list[str] = field(default_factory=list)
    llm_declarou_sem_evidencia: bool = False

    @property
    def fundamentada(self) -> bool:
        return bool(self.citacoes_validas) and not self.llm_declarou_sem_evidencia


def build_messages(pergunta: str, filtros: str, contexto: str) -> tuple[str, str]:
    return SYSTEM_RAG, USER_RAG.format(pergunta=pergunta, filtros=filtros, contexto=contexto)


def fix_citations(line: str, valid_labels: set[str]) -> tuple[str, list[str], list[str]]:
    """Normaliza citações da linha ([E1, E3] -> [E1][E3]) e remove as inválidas."""
    good_all, bad_all = [], []

    def _fix(m: re.Match) -> str:
        labels = [f"E{n}" for n in _LABEL.findall(m.group(1))]
        good = [lab for lab in labels if lab in valid_labels]
        bad_all.extend(lab for lab in labels if lab not in valid_labels)
        good_all.extend(good)
        return "".join(f"[{lab}]" for lab in good)

    return _BRACKET.sub(_fix, line).rstrip(), good_all, bad_all


def is_header(line: str) -> bool:
    return line.strip().endswith(":") and len(line) < 60


def verify_answer(raw: str, valid_labels: set[str]) -> tuple[str, GroundingReport]:
    rep = GroundingReport()
    if SEM_EVIDENCIA_TOKEN in raw:
        rep.llm_declarou_sem_evidencia = True
        return "", rep

    kept_lines = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        fixed, cited_here, bad = fix_citations(line, valid_labels)
        rep.citacoes_invalidas.extend(bad)
        content = re.sub(r"[\s\-•*:]+", "", re.sub(r"\[E\d+\]", "", fixed))
        if not content:
            continue
        if not cited_here and is_header(fixed):
            continue  # cabeçalho ("Padrões observados:"), não é afirmação
        if cited_here:
            kept_lines.append(fixed)
            rep.citacoes_validas.extend(cited_here)
            rep.expressoes_quantitativas.extend(m.group(0) for m in _QUANT.finditer(fixed))
        else:
            rep.linhas_removidas_sem_citacao.append(line.strip())

    # Ordem de primeira citação, sem repetição.
    rep.citacoes_validas = list(dict.fromkeys(rep.citacoes_validas))
    return "\n".join(kept_lines), rep

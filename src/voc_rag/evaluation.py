"""Avaliação do pipeline: recuperação, gate de abstenção e geração.

Métricas
- Recuperação (perguntas com `relevancia_regex`): precisão@k "silver" =
  fração das k evidências do contexto cujo texto casa com a regex do tema;
  posição da 1ª evidência relevante (MRR). Comparação entre modos: BM25,
  densa, híbrida e híbrida + reranker.
- Filtros: acerto da extração de filtros (`filtros_esperados`).
- Gate e ponta a ponta, com DOIS critérios, ambos publicados:
  tolerante ("recusou quando devia"; qualquer abstenção serve):
    respondivel   -> ok
    sem_evidencia -> sem_evidencias | fora_do_escopo  (não pode ser ok)
    insuficiente  -> evidencias_insuficientes | sem_evidencias (não pode ser ok)
    fora_escopo   -> fora_do_escopo | sem_evidencias (não pode ser ok)
  estrito ("recusou pelo motivo certo"): exatamente ok | sem_evidencias |
    evidencias_insuficientes | fora_do_escopo, respectivamente.
- Geração (com LLM): % de respostas fundamentadas, citações inválidas,
  linhas removidas por falta de citação, expressões quantitativas, e se o
  LLM recusou nas perguntas sem evidência que passaram pelo gate.

Limitação declarada: a relevância "silver" por regex favorece casamento
léxico. Em 7 das 8 perguntas de teste a regex casa termos da própria
pergunta, o que favorece o BM25 e a busca híbrida; a métrica é lida como
comparação indicativa entre modos, não como precisão absoluta.
- Filtros (desafio adicional): fração das k evidências dentro do recorte
  pedido, com e sem o pré-filtro de metadados (filter_effect).
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .evidence import STATUS_FORA_ESCOPO, STATUS_INSUFICIENTE, STATUS_OK, STATUS_SEM_EVIDENCIAS, assess
from .query import plan_query
from .text_utils import normalize_for_rules

ACEITOS = {
    "respondivel": {STATUS_OK},
    "sem_evidencia": {STATUS_SEM_EVIDENCIAS, STATUS_FORA_ESCOPO},
    "insuficiente": {STATUS_INSUFICIENTE, STATUS_SEM_EVIDENCIAS},
    "fora_escopo": {STATUS_FORA_ESCOPO, STATUS_SEM_EVIDENCIAS},
}
ACEITOS_ESTRITO = {
    "respondivel": {STATUS_OK},
    "sem_evidencia": {STATUS_SEM_EVIDENCIAS},
    "insuficiente": {STATUS_INSUFICIENTE},
    "fora_escopo": {STATUS_FORA_ESCOPO},
}


def read_annotation_csv(path: str | Path) -> pd.DataFrame:
    """Lê uma planilha de anotação (auditorias) gravada com ';' (Excel em
    português) ou ',' (editores de texto, Google Sheets), em UTF-8 (com ou sem
    BOM) ou em cp1252 (Excel salvando como "CSV" simples no Windows)."""
    path = Path(path)
    raw = path.read_bytes()
    try:
        text, enc = raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        text, enc = raw.decode("cp1252"), "cp1252"
    header = text.splitlines()[0]
    return pd.read_csv(path, sep=";" if ";" in header else ",", dtype=str, encoding=enc).fillna("")


def write_annotation_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Grava em formato que o Excel em português abre direto (';' + BOM UTF-8)."""
    df.to_csv(path, sep=";", index=False, encoding="utf-8-sig")


def load_questions(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))["perguntas"]


def verify_absence(questions: list[dict], docs: pd.DataFrame) -> pd.DataFrame:
    """Confere que os temas das perguntas 'sem_evidencia' de fato NÃO existem
    na base. Se existirem, a pergunta está mal rotulada (falha explícita)."""
    norm = docs["texto"].map(normalize_for_rules)
    warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression")
    rows = []
    for q in questions:
        if q.get("verificacao_ausencia"):
            n = int(norm.str.contains(q["verificacao_ausencia"], regex=True).sum())
            rows.append({"id": q["id"], "regex": q["verificacao_ausencia"], "ocorrencias_na_base": n})
    df = pd.DataFrame(rows)
    if len(df) and (df["ocorrencias_na_base"] > 0).any():
        bad = df[df["ocorrencias_na_base"] > 0]["id"].tolist()
        raise AssertionError(f"Perguntas rotuladas 'sem_evidencia' com ocorrências na base: {bad}")
    return df


def check_filters(rag, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        if "filtros_esperados" not in q:
            continue
        plan = plan_query(q["pergunta"], rag.known_categories)
        got = plan.filtros.to_dict()
        exp = q["filtros_esperados"]
        ok = all((sorted(got[k]) if isinstance(got[k], list) else got[k]) ==
                 (sorted(v) if isinstance(v, list) else v) for k, v in exp.items())
        # Nenhum filtro além dos esperados pode ter sido inferido.
        extras = [k for k, v in got.items() if v not in (None, []) and k not in exp]
        rows.append({"id": q["id"], "pergunta": q["pergunta"], "esperado": exp,
                     "obtido": {k: v for k, v in got.items() if v not in (None, [])},
                     "acerto": ok and not extras})
    return pd.DataFrame(rows)


def filter_effect(rag, questions: list[dict], k: int = 8) -> pd.DataFrame:
    """Efeito dos filtros de metadados na recuperação (desafio adicional).

    Para cada pergunta com `filtros_esperados`: fração das k primeiras
    evidências que pertencem ao recorte pedido (ex.: UF = SP e entrega
    atrasada), com o pré-filtro do sistema e sem ele (mesma pergunta, busca
    na base inteira). Mede o que a busca semântica sozinha não enxerga:
    metadados que não estão no texto da avaliação."""
    docs = rag.index.docs
    rows = []
    for q in questions:
        if "filtros_esperados" not in q:
            continue
        plan = plan_query(q["pergunta"], rag.known_categories)
        if plan.filtros.is_empty():
            continue
        recorte = plan.filtros.mask(docs)
        sem = plan_query(q["pergunta"], rag.known_categories, infer_filters=False)
        for modo, p in (("com filtro", plan), ("sem filtro", sem)):
            top = rag.retriever.retrieve(p).candidatos[:k]
            rows.append({"id": q["id"], "filtros": plan.filtros.describe(), "modo": modo,
                         f"dentro_do_recorte@{k}": round(float(np.mean([recorte[c.pos] for c in top])), 3) if top else 0.0,
                         "pct_da_base_no_recorte": round(float(recorte.mean()), 4)})
    return pd.DataFrame(rows)


def _precision_at_k(texts: list[str], regex: str, k: int) -> tuple[float, int | None]:
    rx = re.compile(regex)
    hits = [bool(rx.search(normalize_for_rules(t))) for t in texts[:k]]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return (float(np.mean(hits)) if hits else 0.0), first


def retrieval_comparison(rag, questions: list[dict], k: int = 8) -> pd.DataFrame:
    """Compara modos de recuperação nas perguntas com relevancia_regex."""
    modes = [("bm25", "bm25", False), ("densa", "densa", False), ("hibrida", "hibrida", False)]
    if rag.s.reranker_model:
        modes.append(("hibrida+reranker", "hibrida", True))
    rows = []
    for q in questions:
        if not q.get("relevancia_regex"):
            continue
        plan = plan_query(q["pergunta"], rag.known_categories)
        for name, mode, rr in modes:
            ret = rag.retriever.retrieve(plan, use_reranker=rr, mode=mode)
            texts = [c.texto for c in ret.candidatos[:k]]
            p, first = _precision_at_k(texts, q["relevancia_regex"], k)
            rows.append({"id": q["id"], "modo": name, f"precisao@{k}": p,
                         "rr": 1.0 / first if first else 0.0})
    return pd.DataFrame(rows)


def gate_features(rag, questions: list[dict]) -> pd.DataFrame:
    """Para cada pergunta: score de escopo e relevância do n-ésimo melhor
    candidato (n = MIN_EVIDENCES). Base da calibração dos limiares."""
    rows = []
    n = rag.s.min_evidences
    for q in questions:
        plan = plan_query(q["pergunta"], rag.known_categories)
        ret = rag.retriever.retrieve(plan)
        rel = sorted((c.relevancia for c in ret.candidatos), reverse=True)
        rows.append({"id": q["id"], "tipo": q["tipo"], "pergunta": q["pergunta"],
                     "score_escopo": ret.score_escopo,
                     "relevancia_top1": rel[0] if rel else 0.0,
                     f"relevancia_top{n}": rel[n - 1] if len(rel) >= n else 0.0})
    return pd.DataFrame(rows)


def _best_threshold(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float]:
    """Limiar t que maximiza a acurácia balanceada da regra 'score >= t'.
    Em empate, escolhe o ponto médio do intervalo ótimo (margem máxima)."""
    cand = np.unique(np.concatenate([pos, neg]))
    if len(cand) == 0:
        return 0.0, 0.0
    mids = np.concatenate([[cand[0] - 1e-6], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1e-6]])
    best, best_ts = -1.0, []
    for t in mids:
        bacc = 0.5 * ((pos >= t).mean() + (neg < t).mean())
        if bacc > best + 1e-12:
            best, best_ts = bacc, [t]
        elif abs(bacc - best) <= 1e-12:
            best_ts.append(t)
    return float(np.median(best_ts)), float(best)


def calibrate(rag, questions: list[dict]) -> dict:
    feats = gate_features(rag, questions)
    n = rag.s.min_evidences
    in_scope = feats[feats["tipo"] != "fora_escopo"]["score_escopo"].values
    out_scope = feats[feats["tipo"] == "fora_escopo"]["score_escopo"].values
    t_scope, bacc_scope = _best_threshold(in_scope, out_scope)
    pos = feats[feats["tipo"] == "respondivel"][f"relevancia_top{n}"].values
    neg = feats[feats["tipo"].isin(["sem_evidencia", "fora_escopo"])][f"relevancia_top{n}"].values
    t_rel, bacc_rel = _best_threshold(pos, neg)
    return {
        "scope_threshold": round(t_scope, 4),
        "relevance_threshold": round(t_rel, 4),
        "min_evidences": n,
        "acuracia_balanceada_calibracao": {"escopo": round(bacc_scope, 3), "relevancia": round(bacc_rel, 3)},
        "n_perguntas": int(len(feats)),
        "modelo_embeddings": rag.index.meta.get("embedding_model"),
        "reranker": rag.s.reranker_model or None,
        "features": feats.round(4).to_dict(orient="records"),
    }


def gate_eval(rag, questions: list[dict]) -> pd.DataFrame:
    rows = []
    for q in questions:
        plan = plan_query(q["pergunta"], rag.known_categories)
        ret = rag.retriever.retrieve(plan)
        dec = assess(ret, rag.s, plan.filtros.describe(), bool(plan.termos_conteudo), not plan.filtros.is_empty())
        rows.append({"id": q["id"], "tipo": q["tipo"], "status_gate": dec.status,
                     "acerto": dec.status in ACEITOS[q["tipo"]],
                     "acerto_estrito": dec.status in ACEITOS_ESTRITO[q["tipo"]],
                     "score_escopo": round(ret.score_escopo, 4), "n_relevantes": dec.n_relevantes,
                     "pergunta": q["pergunta"]})
    return pd.DataFrame(rows)


def end_to_end(rag, questions: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    rows, responses = [], []
    for q in questions:
        r = rag.ask(q["pergunta"])
        v = r.diagnostico.get("verificacao", {})
        rows.append({
            "id": q["id"], "tipo": q["tipo"], "modo": r.modo, "status": r.status,
            "acerto_status": r.status in ACEITOS[q["tipo"]],
            "acerto_estrito": r.status in ACEITOS_ESTRITO[q["tipo"]],
            "n_evidencias": len(r.evidencias),
            "n_citadas": sum(1 for e in r.evidencias if e.get("citada")),
            "citacoes_invalidas": len(v.get("citacoes_invalidas", [])),
            "linhas_sem_citacao_removidas": len(v.get("linhas_removidas_sem_citacao", [])),
            "expressoes_quantitativas": len(v.get("expressoes_quantitativas", [])),
            "llm_recusou": bool(v.get("llm_declarou_sem_evidencia", False)),
            "tempo_total_ms": r.diagnostico.get("tempos_ms", {}).get("total") or r.diagnostico.get("tempo_total_ms"),
        })
        responses.append(r.to_dict())
    return pd.DataFrame(rows), responses

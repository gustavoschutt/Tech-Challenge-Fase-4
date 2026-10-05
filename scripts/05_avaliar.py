"""Etapa 5 — Avalia o pipeline no conjunto de TESTE (eval/perguntas_teste.json).

Gera em eval/resultados/<modelo>/:
  ausencia_verificada.csv, filtros.csv, efeito_filtros.csv, recuperacao.csv,
  recuperacao_resumo.csv, gate.csv, ponta_a_ponta.csv (se houver LLM),
  respostas.json, resumo.json
e o relatório docs/resultados_avaliacao.md (nada escrito à mão).

Uso:
  python scripts/05_avaliar.py            # completo (usa o LLM configurado)
  python scripts/05_avaliar.py --sem-llm  # só recuperação, filtros e gate
"""
import argparse
import json

import _bootstrap  # noqa: F401

from voc_rag.config import ROOT, get_settings
from voc_rag.evaluation import (check_filters, end_to_end, filter_effect, gate_eval, load_questions,
                                retrieval_comparison, verify_absence)
from voc_rag.pipeline import VoCRAG


def _por_tipo(df, col_tol: str) -> "pd.DataFrame":
    g = df.groupby("tipo").agg(perguntas=(col_tol, "count"), acerto_tolerante=(col_tol, "mean"),
                               acerto_estrito=("acerto_estrito", "mean"))
    return g.round(3).reset_index()


def write_report(resumo, rec_sum, filt, efeito, gate, e2e, out, k):
    """Gera docs/resultados_avaliacao.md a partir dos resultados (nada escrito à mão)."""
    from datetime import datetime

    lim = resumo["limiares"]
    L = ["# Resultados da avaliação", "",
         f"> Gerado por `scripts/05_avaliar.py` em {datetime.now():%Y-%m-%d %H:%M}. Arquivos brutos: "
         f"`{out.relative_to(ROOT).as_posix()}/`.", "",
         "## Configuração", "",
         f"- Embeddings: `{resumo['modelo_embeddings']}`",
         f"- Reranker: `{resumo['reranker'] or 'desligado'}`",
         f"- LLM: `{resumo.get('llm', 'não executado (--sem-llm)')}`",
         f"- Limiares do gate: escopo = {lim['escopo']} e relevância = {lim['relevancia']} "
         f"(calibrados no conjunto de calibração: {lim['calibrados']}); mínimo de evidências = "
         f"{lim['min_evidencias']} e base mínima da camada analítica = {lim['min_base_analitico']} (fixados, não calibrados)",
         "",
         "## Recuperação (perguntas com relevância *silver*)", "", rec_sum.reset_index().to_markdown(index=False), "",
         "> A relevância *silver* é uma regex do tema. Em 7 das 8 perguntas ela casa termos da própria pergunta, "
         "o que favorece o BM25 e a busca híbrida: leia a tabela como comparação indicativa entre modos.", "",
         "## Filtros de metadados", "",
         f"Extração dos filtros: {filt['acerto'].mean():.0%} de acerto ({int(filt['acerto'].sum())}/{len(filt)})"
         if len(filt) else "-", ""]
    if len(efeito):
        piv = efeito.pivot(index=["id", "filtros"], columns="modo", values=f"dentro_do_recorte@{k}").reset_index()
        L += [f"Efeito do pré-filtro: fração das {k} evidências que pertencem ao recorte pedido, com e sem o filtro "
              "(mesma pergunta).", "", piv.to_markdown(index=False), "",
              f"Média: com filtro {efeito[efeito.modo == 'com filtro'][f'dentro_do_recorte@{k}'].mean():.3f}; "
              f"sem filtro {efeito[efeito.modo == 'sem filtro'][f'dentro_do_recorte@{k}'].mean():.3f}.", ""]
    L += ["## Gate de abstenção (status antes do LLM)", "",
          "Critério tolerante: qualquer abstenção conta como acerto quando a pergunta não é respondível. "
          "Critério estrito: o status precisa ser exatamente o do tipo da pergunta.", "",
          _por_tipo(gate, "acerto").to_markdown(index=False), "",
          f"Acerto geral: tolerante {gate['acerto'].mean():.1%}; estrito {gate['acerto_estrito'].mean():.1%}", ""]
    erros = gate[~gate["acerto_estrito"]]
    if len(erros):
        L += ["Divergências do gate (critério estrito):", "",
              erros[["id", "tipo", "status_gate", "acerto", "pergunta"]].rename(
                  columns={"acerto": "aceito_no_tolerante"}).to_markdown(index=False), ""]
    if e2e is not None:
        ok = e2e[e2e["status"] == "ok"]
        L += ["## Ponta a ponta (com LLM, modo automático)", "",
              _por_tipo(e2e, "acerto_status").to_markdown(index=False), "",
              f"- Acerto geral de status: tolerante {e2e['acerto_status'].mean():.1%}; "
              f"estrito {e2e['acerto_estrito'].mean():.1%}",
              f"- Respostas com status ok: {len(ok)}; destas, com ≥ 1 evidência citada: "
              f"{(ok['n_citadas'] > 0).mean():.0%}" if len(ok) else "- Nenhuma resposta ok",
              f"- Citações inválidas removidas: {int(e2e['citacoes_invalidas'].sum())}",
              f"- Linhas sem citação removidas: {int(e2e['linhas_sem_citacao_removidas'].sum())}",
              f"- Recusas do próprio LLM: {int(e2e['llm_recusou'].sum())}",
              f"- Tempo mediano por pergunta: {e2e['tempo_total_ms'].median() / 1000:.1f} s"
              if e2e['tempo_total_ms'].notna().any() else "", ""]
    (ROOT / "docs" / "resultados_avaliacao.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-llm", action="store_true")
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()

    s = get_settings()
    rag = VoCRAG(s, load_llm=not a.sem_llm)
    if not rag.thresholds_calibrados:
        print("AVISO: limiares NÃO calibrados. Rode antes: python scripts/04_calibrar_limiares.py")
    if rag.analytics is None:
        print("AVISO: camada analítica indisponível (falta artifacts/base/temas.parquet). "
              "As perguntas agregadas serão respondidas pelo RAG puro. Rode antes: python scripts/03_construir_temas.py")
    qs = load_questions(ROOT / "eval" / "perguntas_teste.json")
    out = ROOT / "eval" / "resultados" / s.model_slug
    out.mkdir(parents=True, exist_ok=True)

    verify_absence(qs, rag.index.docs).to_csv(out / "ausencia_verificada.csv", index=False)
    filt = check_filters(rag, qs); filt.to_csv(out / "filtros.csv", index=False)
    efeito = filter_effect(rag, qs, k=a.k); efeito.to_csv(out / "efeito_filtros.csv", index=False)
    rec = retrieval_comparison(rag, qs, k=a.k); rec.to_csv(out / "recuperacao.csv", index=False)
    rec_sum = rec.groupby("modo", sort=False).mean(numeric_only=True).round(3)
    rec_sum.to_csv(out / "recuperacao_resumo.csv")
    gate = gate_eval(rag, qs); gate.to_csv(out / "gate.csv", index=False)

    col = f"dentro_do_recorte@{a.k}"
    resumo = {
        "modelo_embeddings": rag.index.meta.get("embedding_model"),
        "reranker": s.reranker_model or None,
        "limiares": {"escopo": s.scope_threshold, "relevancia": s.relevance_threshold,
                     "min_evidencias": s.min_evidences, "min_base_analitico": s.min_base_analitico,
                     "calibrados": rag.thresholds_calibrados},
        "filtros_acerto": float(filt["acerto"].mean()) if len(filt) else None,
        "efeito_filtros_medio": ({m: round(float(g[col].mean()), 3) for m, g in efeito.groupby("modo")}
                                 if len(efeito) else None),
        "recuperacao": rec_sum.reset_index().to_dict(orient="records"),
        "gate_acerto_por_tipo": gate.groupby("tipo")["acerto"].mean().round(3).to_dict(),
        "gate_acerto_geral": round(float(gate["acerto"].mean()), 3),
        "gate_acerto_estrito_geral": round(float(gate["acerto_estrito"].mean()), 3),
    }
    e2e = None
    if not a.sem_llm:
        e2e, responses = end_to_end(rag, qs)
        e2e.to_csv(out / "ponta_a_ponta.csv", index=False)
        (out / "respostas.json").write_text(json.dumps(responses, ensure_ascii=False, indent=1, default=str),
                                            encoding="utf-8")
        resumo["llm"] = rag.llm.name
        resumo["ponta_a_ponta_acerto_por_tipo"] = e2e.groupby("tipo")["acerto_status"].mean().round(3).to_dict()
        resumo["ponta_a_ponta_acerto_geral"] = round(float(e2e["acerto_status"].mean()), 3)
        resumo["ponta_a_ponta_acerto_estrito_geral"] = round(float(e2e["acerto_estrito"].mean()), 3)
        ok = e2e[e2e["status"] == "ok"]
        resumo["respostas_ok"] = int(len(ok))
        resumo["citacoes_invalidas_total"] = int(e2e["citacoes_invalidas"].sum())
        resumo["linhas_sem_citacao_removidas_total"] = int(e2e["linhas_sem_citacao_removidas"].sum())
    (out / "resumo.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_report(resumo, rec_sum, filt, efeito, gate, None if a.sem_llm else e2e, out, a.k)
    print(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))
    print(f"\nResultados em {out}")

"""Teste de ponta a ponta numa base sintética: preparação -> índice (LSA)
-> temas -> pipeline (RAG e analítico) com o modo extrativo (sem LLM)."""
import pandas as pd
import pytest

from voc_rag.data_prep import build_knowledge_base
from voc_rag.indexing import build_index


@pytest.fixture(scope="module")
def rag(settings_sinteticas):
    from voc_rag.pipeline import VoCRAG
    from voc_rag.themes import label_texts

    s = settings_sinteticas
    rep = build_knowledge_base(s, verbose=False)
    assert rep["avaliacoes_brutas"] == 60 and rep["documentos_indexados"] == 60
    build_index(s, verbose=False)
    docs = pd.read_parquet(s.base_dir / "documentos.parquet")
    pd.concat([docs[["review_id"]], label_texts(docs["texto"])], axis=1).to_parquet(s.base_dir / "temas.parquet")
    return VoCRAG(s)


def test_preparacao_mascara_pii_e_metadados(rag):
    d = rag.index.docs
    assert d["pii_mascarada"].sum() == 3                  # 1 texto x 3 repetições
    assert set(d["situacao_entrega"]) <= {"atrasada", "no_prazo", "nao_entregue"}
    assert d["texto"].str.contains("Não recebi o produto. Comprei").any()   # título + comentário


def test_rag_responde_com_citacoes(rag):
    r = rag.ask("O que os clientes contam sobre produtos quebrados?", modo="rag")
    assert r.status == "ok" and r.modo == "rag"
    assert any(e["citada"] for e in r.evidencias)
    assert all(e["review_id"] in set(rag.index.docs["review_id"]) for e in r.evidencias)


def test_termo_ausente_gera_abstencao(rag):
    r = rag.ask("O que os clientes acham do pagamento com Pix?", modo="rag")
    assert r.status == "sem_evidencias" and r.resposta == ""
    assert "pix" in r.avisos[0]


def test_filtro_sem_documentos(rag):
    r = rag.ask("O que dizem os clientes do AM?", modo="rag")
    assert r.status == "sem_evidencias"


def test_analitico_conta_sobre_a_base(rag):
    r = rag.ask("Quais são os principais problemas relatados pelos clientes?")
    assert r.modo == "analitico" and r.status == "ok"
    tab = {row["tema_id"]: row["avaliacoes"] for row in r.estatisticas["tabela"]}
    assert tab.get("nao_recebido") == 6                    # 2 textos x 3 repetições
    assert r.evidencias and all("tema" in e for e in r.evidencias)


def test_hibrida_sem_reranker_mantem_ordem_rrf(rag):
    """Sem reranker, a híbrida deve sair na ordem do RRF (e não reordenada
    pelo cosseno, o que a tornaria idêntica à densa)."""
    from voc_rag.query import plan_query

    plan = plan_query("O que os clientes contam sobre produtos quebrados?", rag.known_categories)
    hib = rag.retriever.retrieve(plan, use_reranker=False, mode="hibrida")
    rrf = [c.score_rrf for c in hib.candidatos]
    assert rrf == sorted(rrf, reverse=True)
    densa = rag.retriever.retrieve(plan, use_reranker=False, mode="densa")
    rank = [c.rank_denso for c in densa.candidatos]
    assert rank == sorted(rank)


def _rag_com(rag, **mudancas):
    """Mesmo índice e artefatos, com outros parâmetros do gate."""
    from dataclasses import replace

    from voc_rag.pipeline import VoCRAG

    return VoCRAG(replace(rag.s, **mudancas))


def test_analitico_sem_base_minima_nao_calcula_porcentagem(rag):
    r = _rag_com(rag, min_base_analitico=1000).ask("Quais são os principais problemas relatados pelos clientes?")
    assert r.modo == "analitico" and r.status == "evidencias_insuficientes" and r.resposta == ""
    assert "não calculo porcentagens" in r.avisos[0]


def test_analitico_fora_do_escopo_vai_para_o_gate(rag):
    alto = _rag_com(rag, scope_threshold=0.99)
    r = alto.ask("Quais são os principais problemas com a entrega?")       # tem termo de conteúdo
    assert r.status == "fora_do_escopo" and "desvio_analitico" in r.diagnostico
    r = alto.ask("Quais são os principais problemas relatados pelos clientes?")  # só a base inteira
    assert r.modo == "analitico" and r.status == "ok"


def test_analitico_termo_raro_vai_para_o_gate(rag):
    # Cada texto sintético aparece 3 vezes; com mínimo 4, "falsificado" vira termo raro.
    r = _rag_com(rag, min_evidences=4).ask("Quais são os principais problemas com produtos falsificados?")
    assert r.status == "evidencias_insuficientes" and "desvio_analitico" in r.diagnostico
    assert r.evidencias and all("falsificado" in e["texto"].lower() for e in r.evidencias)

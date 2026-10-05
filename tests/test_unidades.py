"""Testes de unidade das peças determinísticas do pipeline."""
import numpy as np
import pandas as pd

from voc_rag.evaluation import _best_threshold
from voc_rag.generation import verify_answer
from voc_rag.query import extract_filters, plan_query
from voc_rag.retrieval import rrf_fuse
from voc_rag.text_utils import clean_for_embedding, dedup_key, tokenize_lexical
from voc_rag.themes import THEMES_BY_ID, label_texts


def test_rrf_premia_consenso():
    fused = dict(rrf_fuse([[1, 2, 3], [3, 1, 4]], k=60))
    assert max(fused, key=fused.get) == 1          # 1º e 2º lugares
    assert fused[3] > fused[2] and fused[4] > 0


def test_pii_mascarada_e_quebra_de_linha():
    t = clean_for_embedding("Me liga\r\n(11) 98888-7777 ou fulano@x.com")
    assert "[TELEFONE]" in t and "[EMAIL]" in t and "\n" not in t


def test_dedup_key_ignora_caixa_acento_pontuacao():
    assert dedup_key("Ótimo!") == dedup_key("otimo") == "otimo"


def test_bm25_mantem_negacao():
    toks = tokenize_lexical("Não recebi o produto")
    assert "nao" in toks and "o" not in toks


def test_filtros_nota_periodo_categoria_uf():
    f, why, _ = extract_filters("Problemas de clientes insatisfeitos com móveis em SP em 2018")
    assert f.nota_max == 2 and f.data_inicio == "2018-01-01" and f.data_fim == "2019-01-01"
    assert "moveis_decoracao" in f.categorias and f.ufs == ["SP"]


def test_filtro_mes_e_entrega_atrasada():
    f, _, _ = extract_filters("O que dizem dos pedidos entregues com atraso em março de 2018?")
    assert (f.data_inicio, f.data_fim) == ("2018-03-01", "2018-04-01")
    assert f.situacao_entrega == ["atrasada"]


def test_sem_falso_filtro_em_tema():
    # "atraso" como TEMA não vira filtro de pedidos atrasados
    f, _, _ = extract_filters("O que os clientes relatam sobre atraso na entrega?")
    assert f.is_empty()


def test_roupa_de_cama_nao_e_moda():
    f, _, _ = extract_filters("reclamações sobre roupa de cama")
    assert f.categorias == ["cama_mesa_banho"]


def test_intencao_agregada():
    assert plan_query("Quais são os principais problemas?").agregada
    assert not plan_query("O que os clientes relatam sobre a embalagem?").agregada


def test_termos_de_conteudo_ignoram_moldura_e_filtros():
    p = plan_query("Quais são os principais problemas relatados nas avaliações negativas de móveis em 2018?")
    assert p.termos_conteudo == {}
    p = plan_query("O que os clientes acham de pagar com Pix?")
    assert "pix" in p.termos_conteudo.values()


def test_verificacao_remove_citacao_invalida_e_linha_sem_citacao():
    raw = "Resumo: entregas atrasam [E1][E9]\n- Produto quebrado [E2]\n- Afirmação solta sem fonte"
    txt, rep = verify_answer(raw, {"E1", "E2"})
    assert "[E9]" not in txt and "Afirmação solta" not in txt
    assert rep.citacoes_invalidas == ["E9"] and rep.citacoes_validas == ["E1", "E2"]
    assert rep.linhas_removidas_sem_citacao == ["- Afirmação solta sem fonte"]


def test_verificacao_recusa_do_llm():
    txt, rep = verify_answer("SEM_EVIDENCIA_SUFICIENTE", {"E1"})
    assert txt == "" and rep.llm_declarou_sem_evidencia and not rep.fundamentada


def test_verificacao_numeros_analitico():
    from voc_rag.analytics import AnalyticsEngine

    tabela = "PROBLEMAS — base: 100 avaliações\n- Atraso: 30 avaliações (30,0%)"
    raw = "Resumo: Atraso lidera com 30 avaliações (30,0%).\n- Atraso: 45% reclamam [E1]\n- Atraso: clientes relatam demora [E1]"
    txt, rep, inval = AnalyticsEngine._verify(raw, {"E1"}, tabela)
    assert "45%" not in txt and len(inval) == 1
    assert "30 avaliações" in txt and "demora [E1]" in txt


def test_temas_negacao_e_regras():
    s = pd.Series(["Não recomendo, chegou quebrado", "Recomendo! Chegou antes do prazo",
                   "Entrega rápida e sem nenhuma avaria", "Faltou a nota fiscal"])
    lab = label_texts(s)
    assert not lab.loc[0, "recomenda"] and lab.loc[0, "insatisfacao_geral"] and lab.loc[0, "danificado"]
    assert lab.loc[1, "recomenda"] and lab.loc[1, "entrega_rapida"]
    assert not lab.loc[2, "danificado"]                      # "sem avaria" é negação
    assert lab.loc[3, "nota_fiscal"] and not lab.loc[3, "incompleto"]
    assert THEMES_BY_ID["atraso"].polaridade == "problema"


def test_limiar_otimo_separa_classes():
    t, bacc = _best_threshold(np.array([0.8, 0.9, 0.7]), np.array([0.2, 0.3]))
    assert 0.3 < t < 0.7 and bacc == 1.0


def test_leitura_de_csv_compactado(tmp_path):
    import gzip

    import pytest

    from voc_rag.data_prep import raw_path, read_raw

    with gzip.open(tmp_path / "x.csv.gz", "wt", encoding="utf-8") as f:
        f.write("a,b\n1,ótimo\n")
    assert raw_path(tmp_path, "x.csv").name == "x.csv.gz"
    df = read_raw(tmp_path, "x.csv")
    assert df.shape == (1, 2) and df.loc[0, "b"] == "ótimo"
    (tmp_path / "x.csv").write_text("a,b\n2,bom\n", encoding="utf-8")
    assert raw_path(tmp_path, "x.csv").name == "x.csv"          # .csv tem preferência
    with pytest.raises(FileNotFoundError):
        raw_path(tmp_path, "inexistente.csv")


def test_categoria_com_dois_termos_nao_vira_termo_de_conteudo():
    # "beleza e saúde" -> filtro de categoria; nenhum dos dois termos pode
    # sobrar como termo de conteúdo (isso estreitaria o recorte analítico)
    p = plan_query("O que os clientes mais elogiam em produtos de beleza e saúde?")
    assert p.filtros.categorias == ["beleza_saude"]
    assert "saude" not in p.termos_conteudo.values() and "beleza" not in p.termos_conteudo.values()


def test_mensagem_fora_do_escopo_concorda_em_numero():
    from voc_rag.evidence import STATUS_FORA_ESCOPO, GateDecision, gate_message

    um = GateDecision(STATUS_FORA_ESCOPO, "score de escopo 0.780 < limiar 0.832; "
                                          "termos sem nenhuma ocorrência na base: 'pix'")
    assert "o termo 'pix' não aparece" in gate_message(um)
    dois = GateDecision(STATUS_FORA_ESCOPO, "score de escopo 0.780 < limiar 0.832; "
                                            "termos sem nenhuma ocorrência na base: 'pix', 'cashback'")
    assert "os termos 'pix', 'cashback' não aparecem" in gate_message(dois)
    assert "Além disso" not in gate_message(GateDecision(STATUS_FORA_ESCOPO, "score de escopo 0.780 < limiar 0.832"))


def test_termo_raro_mostra_mencao_mesmo_abaixo_do_limiar():
    """Termo que aparece 1 vez na base: a menção é mostrada como evidência
    insuficiente, mesmo que o reranker a pontue abaixo do limiar."""
    from voc_rag.config import Settings
    from voc_rag.evidence import STATUS_INSUFICIENTE, STATUS_SEM_EVIDENCIAS, assess
    from voc_rag.retrieval import Evidence, RetrievalResult

    def ev(pos, texto, rel):
        return Evidence(pos=pos, review_id=f"r{pos}", order_id=f"o{pos}", texto=texto, nota=1,
                        data_compra="2018-01-01", categoria="x", uf="SP", situacao_entrega="no_prazo",
                        atraso_dias=None, relevancia=rel)

    s = Settings(relevance_threshold=0.05, scope_threshold=0.0, min_evidences=3, lexical_gate=True)
    cands = [ev(1, "Comprei um patinete e recebi outro produto", 0.02), ev(2, "Entrega rápida", 0.01)]
    dec = assess(RetrievalResult(cands, 100, 0.9, True, termos_raros={"patinetes": 1}), s, "nenhum", True, False)
    assert dec.status == STATUS_INSUFICIENTE and [e.review_id for e in dec.evidencias] == ["r1"]
    # Sem menção entre os candidatos: segue o fluxo normal (relevância).
    dec = assess(RetrievalResult(cands[1:], 100, 0.9, True, termos_raros={"patinetes": 1}), s, "nenhum", True, False)
    assert dec.status == STATUS_SEM_EVIDENCIAS


def test_planilha_de_anotacao_aceita_excel_e_texto(tmp_path):
    from voc_rag.evaluation import read_annotation_csv, write_annotation_csv

    df = pd.DataFrame({"texto": ["Não chegou; péssimo, demorou"], "correto": ["1"]})
    write_annotation_csv(df, tmp_path / "a.csv")                                 # ';' + BOM (Excel pt-BR)
    df.to_csv(tmp_path / "b.csv", index=False, encoding="utf-8")                 # ',' (editor de texto)
    df.to_csv(tmp_path / "c.csv", sep=";", index=False, encoding="cp1252")      # Excel "CSV" simples
    for f in ("a.csv", "b.csv", "c.csv"):
        assert read_annotation_csv(tmp_path / f).equals(df), f


def test_experiencias_insatisfatorias_viram_nota_baixa_e_pergunta_agregada():
    p = plan_query("Quais temas aparecem nas avaliações relacionadas a experiências de entrega insatisfatórias?")
    assert p.filtros.nota_max == 2 and p.agregada
    assert "insatisfatorias" not in p.termos_conteudo.values()   # não dispara a checagem de termo ausente
    assert plan_query("Que experiências satisfatórias os clientes relatam?").filtros.nota_min == 4

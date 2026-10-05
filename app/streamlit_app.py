"""Interface web (Streamlit) do Voice of Customer Intelligence.

Uso:  streamlit run app/streamlit_app.py
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voc_rag.pipeline import VoCRAG  # noqa: E402
from voc_rag.query import SITUACOES_ENTREGA, UFS, Filters  # noqa: E402

st.set_page_config(page_title="Voz do Cliente · Olist", page_icon="💬", layout="wide")

EXEMPLOS = [
    "Quais são os principais problemas relatados pelos clientes em suas avaliações?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados pelos clientes?",
    "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?",
    "O que os clientes contam sobre mercadorias que chegaram quebradas ou danificadas?",
    "O que os clientes acham de pagar com Pix?",
    "Qual é a capital da Austrália?",
]
STATUS_UI = {
    "ok": ("Resposta fundamentada", "success"),
    "evidencias_insuficientes": ("Evidências insuficientes", "warning"),
    "sem_evidencias": ("Sem evidências na base", "error"),
    "fora_do_escopo": ("Fora do escopo da base", "error"),
    "resposta_nao_fundamentada": ("Resposta descartada (sem fundamentação)", "error"),
}


@st.cache_resource(show_spinner="Carregando índice e modelos...")
def carregar() -> VoCRAG:
    return VoCRAG()


rag = carregar()
docs = rag.index.docs

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Configuração")
    modo = st.radio("Modo", ["auto", "rag", "analitico"], index=0,
                    help="auto: perguntas de frequência/padrões vão para o modo analítico (contagens); "
                         "as demais, para o RAG.")
    inferir = st.checkbox("Inferir filtros a partir da pergunta", value=True)
    st.subheader("Filtros explícitos")
    nota = st.slider("Nota da avaliação", 1, 5, (1, 5))
    usar_periodo = st.checkbox("Filtrar período da compra")
    periodo = st.date_input("Período", (date(2016, 9, 1), date(2018, 10, 31)), disabled=not usar_periodo)
    categorias = st.multiselect("Categorias", sorted(docs["categoria"].unique()))
    ufs = st.multiselect("UF do cliente", UFS)
    entrega = st.multiselect("Situação da entrega", list(SITUACOES_ENTREGA))
    st.divider()
    st.caption(f"Base: {len(docs):,} avaliações com comentário · embeddings: "
               f"{rag.index.meta.get('embedding_model')} · LLM: {rag.s.llm_provider}:{rag.s.llm_model}".replace(",", "."))
    if not rag.thresholds_calibrados:
        st.warning("Limiares do gate NÃO calibrados. Rode scripts/04_calibrar_limiares.py.")

filtros = Filters(
    nota_min=nota[0] if nota != (1, 5) else None,
    nota_max=nota[1] if nota != (1, 5) else None,
    data_inicio=str(periodo[0]) if usar_periodo and len(periodo) == 2 else None,
    data_fim=str(pd.Timestamp(periodo[1]) + pd.Timedelta(days=1))[:10] if usar_periodo and len(periodo) == 2 else None,
    categorias=categorias, ufs=ufs, situacao_entrega=entrega,
)

# --------------------------------------------------------------------- main
st.title("💬 Voz do Cliente — Olist")
st.caption("Pergunte em linguagem natural. As respostas são geradas apenas a partir de avaliações reais "
           "e mostram as evidências usadas. Quando não há evidência suficiente, o sistema diz isso.")

ex = st.selectbox("Exemplos", ["(escreva sua pergunta abaixo)"] + EXEMPLOS)
pergunta = st.text_area("Pergunta", value="" if ex.startswith("(") else ex, height=80)

if st.button("Perguntar", type="primary") and pergunta.strip():
    with st.spinner("Recuperando avaliações e gerando a resposta..."):
        r = rag.ask(pergunta, filtros=filtros, modo=modo, inferir_filtros=inferir)

    titulo, tipo = STATUS_UI.get(r.status, (r.status, "info"))
    getattr(st, tipo)(f"**{titulo}** · modo `{r.modo}` · filtros: {r.plano.get('filtros_descricao')}")

    if r.resposta:
        st.markdown(r.resposta)
    for a in r.avisos:
        st.info(a)

    if r.estatisticas:
        st.subheader("Contagens por tema (calculadas sobre a base)")
        tab = pd.DataFrame(r.estatisticas["tabela"])
        st.dataframe(tab[["tema", "polaridade", "avaliacoes", "pct_base", "nota_media"]], hide_index=True,
                     width="stretch")
        st.bar_chart(tab.set_index("tema")["avaliacoes"], horizontal=True)

    if r.evidencias:
        st.subheader("Evidências (avaliações usadas)")
        ev = pd.DataFrame(r.evidencias)
        cols = [c for c in ["rotulo", "citada", "tema", "nota", "data_compra", "categoria", "uf", "situacao_entrega",
                            "relevancia", "n_textos_identicos", "texto", "review_id", "order_id"] if c in ev.columns]
        st.dataframe(ev[cols], hide_index=True, width="stretch")

    with st.expander("Diagnóstico do pipeline (plano, gate, tempos, verificação)"):
        st.json({"plano": r.plano, "diagnostico": r.diagnostico})

"""Processamento da pergunta.

Etapas:
1. Normalização (unicode NFC, espaços).
2. Extração de FILTROS DE METADADOS por regras explícitas ("self-query"
   determinístico): nota, período da compra, categoria, UF e situação da
   entrega. Regras em vez de um LLM porque (a) são auditáveis, (b) não
   alucinam filtros e (c) funcionam com qualquer LLM, inclusive pequenos.
   Tudo o que for inferido é DEVOLVIDO ao usuário, que pode corrigir.
3. Detecção de intenção: pergunta AGREGADA ("principais", "mais frequentes",
   "padrões") x ESPECÍFICA. Perguntas agregadas pedem contagem sobre a base
   inteira, o que um top-k de documentos não sustenta (ver camada analítica).
4. Montagem do texto de consulta para os embeddings (prefixo do modelo).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .text_utils import normalize_for_rules, normalize_unicode, tokenize_lexical

# Palavras de "moldura" da pergunta (como se pergunta), não do conteúdo
# perguntado. Ficam fora da checagem de termos ausentes da base.
META_TERMS = set("""
quais qual que quem como onde quando porque por existem existe existiu ha houve sao foram foi seria podem pode
identificar identificados identificadas identificado padrao padroes principais principal maiores maior frequentes
frequencia frequente recorrentes recorrente comuns comum mais menos relatam relatados relatadas relatado relatos
relato relatar mencionam mencionados citam citados dizem diz falam fala comentam comentarios comentario contam
contaram avaliacoes avaliacao avaliam avaliaram opiniao opinioes clientes cliente consumidores consumidor
compradores comprador usuarios pessoas experiencia experiencias aspectos aspecto sustentam sustenta conclusao
evidencias evidencia mostram mostra apontam indicam tema temas sobre relacionados relacionadas relacionado
relacionada respeito acham acha pensam percebem percepcao queixas queixa reclamacoes reclamacao reclamam elogios
elogio elogiam elogiados elogiadas valorizam valorizados destacam tipo tipos exemplos exemplo segundo base dados
problemas problema notas nota baixa baixas alta altas negativas positivas negativa positiva insatisfeitos
satisfeitos insatisfeito satisfeito aparecem aparece aparecer aparecendo podemos possivel
""".split())

# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
SITUACOES_ENTREGA = ("atrasada", "no_prazo", "nao_entregue")


@dataclass
class Filters:
    nota_min: int | None = None
    nota_max: int | None = None
    data_inicio: str | None = None   # AAAA-MM-DD (data da compra, inclusiva)
    data_fim: str | None = None      # AAAA-MM-DD (exclusiva)
    categorias: list[str] = field(default_factory=list)
    ufs: list[str] = field(default_factory=list)
    situacao_entrega: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any([self.nota_min, self.nota_max, self.data_inicio, self.data_fim,
                        self.categorias, self.ufs, self.situacao_entrega])

    def mask(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        if self.nota_min is not None:
            m &= df["review_score"].values >= self.nota_min
        if self.nota_max is not None:
            m &= df["review_score"].values <= self.nota_max
        if self.data_inicio:
            m &= (df["order_purchase_timestamp"] >= pd.Timestamp(self.data_inicio)).values
        if self.data_fim:
            m &= (df["order_purchase_timestamp"] < pd.Timestamp(self.data_fim)).values
        if self.categorias:
            wanted = set(self.categorias)
            m &= df["categorias"].map(lambda cats: bool(wanted.intersection(cats))).values
        if self.ufs:
            m &= df["customer_state"].isin(self.ufs).values
        if self.situacao_entrega:
            m &= df["situacao_entrega"].isin(self.situacao_entrega).values
        return m

    def describe(self) -> str:
        parts = []
        if self.nota_min is not None or self.nota_max is not None:
            lo, hi = self.nota_min or 1, self.nota_max or 5
            parts.append(f"nota {lo}" if lo == hi else f"nota {lo}–{hi}")
        if self.data_inicio or self.data_fim:
            parts.append(f"compra de {self.data_inicio or 'início'} até {self.data_fim or 'fim'} (exclusivo)")
        if self.categorias:
            parts.append("categoria ∈ {" + ", ".join(self.categorias) + "}")
        if self.ufs:
            parts.append("UF ∈ {" + ", ".join(self.ufs) + "}")
        if self.situacao_entrega:
            parts.append("entrega ∈ {" + ", ".join(self.situacao_entrega) + "}")
        return "; ".join(parts) if parts else "nenhum"

    def to_dict(self) -> dict:
        return asdict(self)

    def merged_with(self, other: "Filters | None") -> "Filters":
        """Filtros explícitos (interface) têm precedência sobre os inferidos."""
        if other is None:
            return self
        out = Filters(**self.to_dict())
        for k, v in other.to_dict().items():
            if v not in (None, [], ""):
                setattr(out, k, v)
        return out


# --- regras de nota -----------------------------------------------------------
_NOTA_NEG = re.compile(r"\b(notas? baixas?|avaliac\w+ negativas?|negativas|insatisfeit\w*|insatisfatori\w*|detrator\w*|nota (1|um) (e|ou) (2|dois))\b")
_NOTA_POS = re.compile(r"\b(notas? altas?|avaliac\w+ positivas?|positivas|(?<!in)satisfeit\w*|(?<!in)satisfatori\w*|promotor\w*|nota (4|quatro) (e|ou) (5|cinco))\b")
_NOTA_NEU = re.compile(r"\b(notas? (3|tres)|neutr\w+)\b")
_NOTA_EXATA = re.compile(r"\bnotas? (?:igual a |de )?([1-5])\b(?! (?:e|ou|a) [1-5])")

# --- regras de período ---------------------------------------------------------
_MESES = {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5, "junho": 6, "julho": 7,
          "agosto": 8, "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12}
_MES_ANO = re.compile(r"\b(" + "|".join(_MESES) + r")(?: de)? (2016|2017|2018)\b")
_SEMESTRE = re.compile(r"\b(primeiro|1o|segundo|2o) semestre de (2016|2017|2018)\b")
_ANO = re.compile(r"\b(2016|2017|2018)\b")

# --- regras de categoria (dicionário explícito e auditável) ----------------------
CATEGORY_TERMS: list[tuple[str, list[str]]] = [
    (r"cama,? mesa e banho|cama mesa banho|roupas? de cama|toalhas?|lencol|lencois", ["cama_mesa_banho"]),
    (r"beleza|saude|cosmetic\w*", ["beleza_saude"]),
    (r"esportes?|lazer|fitness", ["esporte_lazer", "fashion_esporte"]),
    (r"informatica|computador\w*|\bpcs?\b", ["informatica_acessorios", "pcs", "pc_gamer"]),
    (r"relogios?", ["relogios_presentes"]),
    (r"move(?:is|l)|decoracao", ["moveis_decoracao", "moveis_escritorio", "moveis_sala", "moveis_quarto",
                                 "moveis_cozinha_area_de_servico_jantar_e_jardim", "moveis_colchao_e_estofado"]),
    (r"utilidades domesticas|utensilios?", ["utilidades_domesticas"]),
    (r"telefonia|celular\w*|smartphones?", ["telefonia", "telefonia_fixa"]),
    (r"automotiv\w*|automove(?:is|l)|carros?|veiculos?", ["automotivo"]),
    (r"ferramentas?|jardinagem|jardim", ["ferramentas_jardim", "construcao_ferramentas_ferramentas",
                                         "construcao_ferramentas_jardim", "construcao_ferramentas_construcao"]),
    (r"brinquedos?", ["brinquedos"]),
    (r"perfum\w*", ["perfumaria"]),
    (r"eletronicos?", ["eletronicos"]),
    (r"bebes?|infantis?|fraldas?", ["bebes", "fraldas_higiene"]),
    (r"papelaria", ["papelaria"]),
    (r"bolsas?", ["fashion_bolsas_e_acessorios"]),
    (r"malas?|bagagens?", ["malas_acessorios"]),
    (r"pet ?shop|\bpets?\b|cachorros?|gatos?|racao", ["pet_shop"]),
    (r"games?|videogames?|consoles?", ["consoles_games"]),
    (r"eletrodomesticos?", ["eletrodomesticos", "eletrodomesticos_2"]),
    (r"eletroportate\w*", ["eletroportateis", "portateis_casa_forno_e_cafe", "portateis_cozinha_e_preparadores_de_alimentos"]),
    (r"construcao", ["casa_construcao", "construcao_ferramentas_construcao", "construcao_ferramentas_iluminacao",
                     "construcao_ferramentas_jardim", "construcao_ferramentas_seguranca", "construcao_ferramentas_ferramentas"]),
    (r"instrumentos? musica\w*", ["instrumentos_musicais"]),
    (r"livros?", ["livros_interesse_geral", "livros_tecnicos", "livros_importados"]),
    (r"alimentos?|comidas?", ["alimentos", "alimentos_bebidas"]),
    (r"bebidas?", ["bebidas", "alimentos_bebidas"]),
    (r"fones? de ouvido|caixas? de som|\baudio\b", ["audio"]),
    (r"calcados?|sapatos?", ["fashion_calcados"]),
    (r"roupas?(?! de cama)|vestuario|moda", ["fashion_roupa_masculina", "fashion_roupa_feminina",
                                            "fashion_underwear_e_moda_praia", "fashion_roupa_infanto_juvenil"]),
    (r"climatizacao|ar condicionado|ventiladores?", ["climatizacao"]),
    (r"natal(?:inos?)?", ["artigos_de_natal"]),
    (r"artigos? de festas?", ["artigos_de_festas"]),
    (r"flores", ["flores"]),
    (r"colch(?:ao|oes)", ["moveis_colchao_e_estofado"]),
    (r"tablets?", ["tablets_impressao_imagem"]),
    (r"iluminacao|lampadas?|luminarias?", ["construcao_ferramentas_iluminacao"]),
]
_CATEGORY_RULES = [(re.compile(r"\b(?:" + pat + r")\b"), cats) for pat, cats in CATEGORY_TERMS]
_CATEGORY_LITERAL = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")  # nome técnico, ex.: cama_mesa_banho

# --- UF ----------------------------------------------------------------------------
UFS = ["AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS", "MG", "PA", "PB", "PR",
       "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO"]
_UF_SIGLA = re.compile(r"\b(" + "|".join(UFS) + r")\b")  # aplicado ao texto ORIGINAL (maiúsculas)
_UF_NOMES = {
    "sao paulo": "SP", "rio de janeiro": "RJ", "minas gerais": "MG", "rio grande do sul": "RS",
    "parana": "PR", "bahia": "BA", "santa catarina": "SC", "espirito santo": "ES", "goias": "GO",
    "pernambuco": "PE", "ceara": "CE", "distrito federal": "DF", "mato grosso do sul": "MS",
    "maranhao": "MA", "paraiba": "PB", "rio grande do norte": "RN", "alagoas": "AL", "piaui": "PI",
    "sergipe": "SE", "rondonia": "RO", "tocantins": "TO", "amazonas": "AM", "acre": "AC",
    "amapa": "AP", "roraima": "RR",
}  # "Pará" foi omitido de propósito: sem acento colide com a preposição "para".
_UF_NOME_RE = re.compile(r"\b(" + "|".join(sorted(_UF_NOMES, key=len, reverse=True)) + r")\b")

# --- situação da entrega (só quando a pergunta delimita PEDIDOS, não temas) ---------
_ENTREGA_ATRASADA = re.compile(r"\b(pedidos|entregas|compras|encomendas)( que foram)? (entregues? )?(com atraso|atrasad[oa]s?)\b")
_ENTREGA_NO_PRAZO = re.compile(r"\b(pedidos|entregas|compras|encomendas)( que foram)? (entregues? )?(no|dentro do) prazo\b")
_ENTREGA_NAO = re.compile(r"\b(pedidos|compras|encomendas)( que)? (nao foram entregues|nao entregues)\b")

# --- intenção agregada ---------------------------------------------------------------
_AGREGADA = re.compile(
    r"\b(principa(?:l|is)|mais (?:frequentes?|comuns?|citad\w*|relatad\w*|mencionad\w*|elogiad\w*|reclamad\w*|valoriz\w*)"
    r"|maior frequencia|frequen\w*|recorrent\w*|padr(?:ao|oes)|quant[oa]s?|proporcao|percentu\w*|porcentagem"
    r"|maioria|ranking|tendencias?|distribuicao|o que (?:os clientes |os consumidores )?mais|mais (?:elogiam|reclamam|valorizam)"
    r"|quais (?:sao )?(?:os )?(?:temas|assuntos|motivos))\b"
)


@dataclass
class QueryPlan:
    pergunta: str
    pergunta_normalizada: str
    filtros: Filters
    filtros_inferidos: list[str]
    agregada: bool
    gatilhos_agregada: list[str]
    termos_conteudo: dict[str, str]  # stem -> palavra original (checagem de cobertura léxica)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["filtros_descricao"] = self.filtros.describe()
        return d


def _add_months(year: int, month: int, n: int) -> tuple[int, int]:
    m = month - 1 + n
    return year + m // 12, m % 12 + 1


def extract_filters(question: str, known_categories: set[str] | None = None) -> tuple[Filters, list[str], list[str]]:
    """Devolve (filtros, justificativas, trechos da pergunta que geraram filtros)."""
    q = normalize_for_rules(question)
    f, why, spans = Filters(), [], []

    # Nota
    if (m := _NOTA_NEG.search(q)):
        f.nota_max = 2; why.append(f"nota ≤ 2 ('{m.group(0)}')"); spans.append(m.group(0))
    elif (m := _NOTA_POS.search(q)):
        f.nota_min = 4; why.append(f"nota ≥ 4 ('{m.group(0)}')"); spans.append(m.group(0))
    elif _NOTA_NEU.search(q):
        f.nota_min = f.nota_max = 3; why.append("nota = 3")
    elif _NOTA_EXATA.search(q):
        n = int(_NOTA_EXATA.search(q).group(1)); f.nota_min = f.nota_max = n; why.append(f"nota = {n}")

    # Período (mais específico primeiro)
    if (m := _MES_ANO.search(q)):
        y, mo = int(m.group(2)), _MESES[m.group(1)]
        y2, mo2 = _add_months(y, mo, 1)
        f.data_inicio, f.data_fim = f"{y}-{mo:02d}-01", f"{y2}-{mo2:02d}-01"
        why.append(f"período = {m.group(0)}"); spans.append(m.group(0))
    elif (m := _SEMESTRE.search(q)):
        y = int(m.group(2)); first = m.group(1) in ("primeiro", "1o")
        f.data_inicio = f"{y}-01-01" if first else f"{y}-07-01"
        f.data_fim = f"{y}-07-01" if first else f"{y + 1}-01-01"
        why.append(f"período = {m.group(0)}"); spans.append(m.group(0))
    else:
        years = sorted({int(y) for y in _ANO.findall(q)})
        if years:
            f.data_inicio, f.data_fim = f"{years[0]}-01-01", f"{years[-1] + 1}-01-01"
            why.append(f"período = {years[0]}" + (f"–{years[-1]}" if len(years) > 1 else ""))
            spans.extend(str(y) for y in years)

    # Categoria
    cats: list[str] = []
    for rx, cs in _CATEGORY_RULES:
        found = [m.group(0) for m in rx.finditer(q)]
        if found:
            # Todos os trechos que casaram viram "spans" (ex.: "beleza" E "saude"),
            # para nenhum deles sobrar como termo de conteúdo e estreitar o recorte.
            cats.extend(cs); why.append("categoria ('" + "', '".join(found) + "')"); spans.extend(found)
    if known_categories:
        for lit in _CATEGORY_LITERAL.findall(q):
            if lit in known_categories:
                cats.append(lit); why.append(f"categoria ('{lit}')")
    if known_categories:
        cats = [c for c in cats if c in known_categories]
    f.categorias = sorted(set(cats))

    # UF
    ufs = set(_UF_SIGLA.findall(normalize_unicode(question)))
    ufs |= {_UF_NOMES[n] for n in _UF_NOME_RE.findall(q)}
    spans.extend(_UF_NOME_RE.findall(q))
    if ufs:
        f.ufs = sorted(ufs); why.append("UF = " + ", ".join(f.ufs))

    # Situação da entrega
    sit = []
    for rx, nome in ((_ENTREGA_ATRASADA, "atrasada"), (_ENTREGA_NO_PRAZO, "no_prazo"), (_ENTREGA_NAO, "nao_entregue")):
        if (m := rx.search(q)):
            sit.append(nome); spans.append(m.group(0))
    if sit:
        f.situacao_entrega = sit; why.append("entrega = " + ", ".join(sit))
    return f, why, spans


def plan_query(question: str, known_categories: set[str] | None = None,
               explicit_filters: Filters | None = None, infer_filters: bool = True) -> QueryPlan:
    question = normalize_unicode(question or "").strip()
    q = normalize_for_rules(question)
    if infer_filters:
        inferred, why, spans = extract_filters(question, known_categories)
    else:
        inferred, why, spans = Filters(), [], []
    filters = inferred.merged_with(explicit_filters)
    if explicit_filters is not None and not explicit_filters.is_empty():
        why.append("filtros explícitos da interface aplicados")
    triggers = sorted({m.group(0) for m in _AGREGADA.finditer(q)})
    return QueryPlan(question, q, filters, why, bool(triggers), triggers, content_terms(q, spans))


def content_terms(q_norm: str, spans: list[str] | None = None) -> dict[str, str]:
    """Termos de CONTEÚDO da pergunta: tira os trechos que viraram filtro
    (categoria, período, UF), as palavras de moldura (META_TERMS) e as
    UFs em sigla; devolve os stems na forma usada pelo índice BM25.
    Anos (ex.: 2023) são mantidos: ano fora de 2016–2018 é ausência real."""
    text = q_norm
    for sp in spans or []:
        text = text.replace(sp, " ")
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in META_TERMS and w.upper() not in UFS]
    out: dict[str, str] = {}
    for w in words:
        for stem in tokenize_lexical(w):
            out.setdefault(stem, w)
    return out

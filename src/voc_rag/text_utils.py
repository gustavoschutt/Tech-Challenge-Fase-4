"""Funções de texto: normalização, mascaramento de dados pessoais e
tokenização léxica (BM25).

Decisão central: o texto que vai para os EMBEDDINGS passa por limpeza mínima
(modelos Transformer usam acentos, caixa e palavras funcionais como sinal).
A limpeza pesada (minúsculas, sem acento, sem stopwords, stemming Snowball) é
aplicada apenas ao ramo LÉXICO (BM25), onde ela melhora o casamento de termos.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# ---------------------------------------------------------------------------
# Normalização e limpeza mínima (ramo semântico)
# ---------------------------------------------------------------------------
_WS = re.compile(r"\s+")
_ALPHA = re.compile(r"[A-Za-zÀ-ÿ]")

# Padrões de dados pessoais. Telefones BR: (DD) 9XXXX-XXXX, com variações.
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    (re.compile(r"https?://\S+|www\.\S+"), "[URL]"),
    (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), "[CPF]"),
    (re.compile(r"\(?\b\d{2}\)?\s?9?\s?\d{4}[-\s]?\d{4}\b"), "[TELEFONE]"),
]


def normalize_unicode(text: str) -> str:
    """NFC: une formas compostas/decompostas ('ótimo' vs 'ótimo')."""
    return unicodedata.normalize("NFC", text)


def mask_pii(text: str) -> str:
    for pattern, token in _PII_PATTERNS:
        text = pattern.sub(token, text)
    return text


def clean_for_embedding(text: str) -> str:
    """Limpeza mínima: unicode NFC, quebras de linha, espaços, PII."""
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = mask_pii(text)
    return _WS.sub(" ", text).strip()


def has_alpha(text: str) -> bool:
    return bool(_ALPHA.search(text or ""))


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def dedup_key(text: str) -> str:
    """Chave para detectar textos idênticos ('Muito bom!' == 'muito bom')."""
    t = strip_accents(text.lower())
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    return _WS.sub(" ", t).strip()


# ---------------------------------------------------------------------------
# Ramo léxico (BM25)
# ---------------------------------------------------------------------------
# Lista própria (evita download em tempo de execução). Negações ("não",
# "nunca", "nem", "sem") são MANTIDAS de propósito: em avaliações elas
# invertem o sentido ("não chegou").
STOPWORDS_PT = set(
    """
    a o as os um uma uns umas de do da dos das no na nos nas em ao aos à às
    por pelo pela pelos pelas para pra pro com e ou que se me te lhe nos vos
    eu tu ele ela eles elas voce voces meu minha meus minhas seu sua seus suas
    esse essa esses essas este esta estes estas isso isto aquele aquela aquilo
    foi era ser sao e esta estao estava tem ter tinha ja muito muita mais
    mas como quando onde qual quais quem ate apos entre sobre tambem so
    ai la aqui ne pois porque entao todo toda todos todas
    """.split()
)


@lru_cache(maxsize=1)
def _stemmer():
    """Stemmer Snowball para português (pacote `snowballstemmer`, puro
    Python, sem download de dados em tempo de execução)."""
    try:
        import snowballstemmer

        return snowballstemmer.stemmer("portuguese")
    except ImportError:  # sem stemming se o pacote não estiver instalado
        return None


_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize_lexical(text: str, use_stemming: bool = True) -> list[str]:
    t = strip_accents(normalize_unicode(text or "").lower())
    tokens = [tok for tok in _TOKEN.findall(t) if tok not in STOPWORDS_PT and len(tok) > 1]
    stem = _stemmer() if use_stemming else None
    if stem is not None:
        tokens = stem.stemWords(tokens)
    return tokens


def normalize_for_rules(text: str) -> str:
    """Forma usada pelas regras (filtros de consulta, temas): minúsculas,
    sem acento, espaços simples."""
    t = strip_accents(normalize_unicode(text or "").lower())
    return _WS.sub(" ", t).strip()

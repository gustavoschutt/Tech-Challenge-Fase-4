"""Configuração central do projeto.

Todos os parâmetros podem ser sobrescritos por variáveis de ambiente ou pelo
arquivo `.env` na raiz do repositório (veja `.env.example`).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # python-dotenv é opcional em tempo de import (testes)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(ROOT / ".env", override=False)

# Menos ruído do Hugging Face nos terminais e nos notebooks publicados (avisos
# informativos). As barras de progresso continuam visíveis para quem baixa os
# modelos pela primeira vez. Cada variável pode ser sobrescrita no .env.
for _var, _val in (("HF_HUB_VERBOSITY", "error"), ("TRANSFORMERS_VERBOSITY", "error"),
                   ("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")):
    os.environ.setdefault(_var, _val)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (ROOT / p).resolve()


# Prefixos exigidos por alguns modelos de embedding assimétricos (E5).
# Modelos fora da lista usam texto sem prefixo.
EMBEDDING_PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}


@dataclass
class Settings:
    # --- caminhos -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _resolve(_env("DATA_DIR", "data/raw")))
    artifacts_dir: Path = field(default_factory=lambda: _resolve(_env("ARTIFACTS_DIR", "artifacts")))

    # --- representação vetorial ---------------------------------------------
    # "sentence-transformers" (padrão) ou "lsa" (TF-IDF + SVD, 100% offline)
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "sentence-transformers"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    embedding_batch_size: int = field(default_factory=lambda: int(_env("EMBEDDING_BATCH_SIZE", "64")))

    # --- re-ranking -----------------------------------------------------------
    # Vazio desliga o reranker (a relevância passa a ser o cosseno denso).
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"))

    # --- recuperação ------------------------------------------------------------
    n_candidates_dense: int = field(default_factory=lambda: int(_env("N_CANDIDATES_DENSE", "100")))
    n_candidates_bm25: int = field(default_factory=lambda: int(_env("N_CANDIDATES_BM25", "100")))
    n_rerank: int = field(default_factory=lambda: int(_env("N_RERANK", "40")))
    top_k_context: int = field(default_factory=lambda: int(_env("TOP_K_CONTEXT", "8")))
    rrf_k: int = field(default_factory=lambda: int(_env("RRF_K", "60")))
    max_context_chars: int = field(default_factory=lambda: int(_env("MAX_CONTEXT_CHARS", "6000")))

    # --- gate de evidências (abstenção) -----------------------------------------
    # Valores padrão; são substituídos por artifacts/limiares__<modelo>.json quando o
    # script de calibração (scripts/04_calibrar_limiares.py) é executado. O mínimo
    # de evidências também é lido desse arquivo, mas não é calibrado (é copiado daqui).
    min_evidences: int = field(default_factory=lambda: int(_env("MIN_EVIDENCES", "3")))
    # Camada analítica: abaixo deste número de avaliações com comentário na base
    # de contagem, porcentagens são instáveis e o sistema não as calcula.
    # Valor fixado (regra prática), NÃO calibrado.
    min_base_analitico: int = field(default_factory=lambda: int(_env("MIN_BASE_ANALITICO", "30")))
    relevance_threshold: float = field(default_factory=lambda: float(_env("RELEVANCE_THRESHOLD", "0.5")))
    scope_threshold: float = field(default_factory=lambda: float(_env("SCOPE_THRESHOLD", "0.0")))
    # Checagem de cobertura léxica: termo de conteúdo da pergunta que não
    # aparece em NENHUMA avaliação (df = 0) -> sem evidências; que aparece em
    # menos de MIN_EVIDENCES avaliações -> evidências insuficientes.
    lexical_gate: bool = field(default_factory=lambda: _env("LEXICAL_GATE", "true").lower() in ("1", "true", "sim"))

    # --- LLM ----------------------------------------------------------------------
    # hf | ollama | openai | gemini | anthropic | openai_compatible | extractive
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "hf"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    llm_base_url: str = field(default_factory=lambda: _env("LLM_BASE_URL", ""))
    llm_max_new_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_NEW_TOKENS", "450")))
    llm_temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0.0")))
    # Só para LLM_PROVIDER=hf: auto (float16 na GPU, float32 na CPU) | float32 | bfloat16
    # bfloat16 na CPU usa metade da memória (útil com pouca RAM), com geração mais lenta.
    llm_dtype: str = field(default_factory=lambda: _env("LLM_DTYPE", "auto"))

    seed: int = field(default_factory=lambda: int(_env("SEED", "42")))

    # ---------------------------------------------------------------------------
    @property
    def base_dir(self) -> Path:
        return self.artifacts_dir / "base"

    @property
    def model_slug(self) -> str:
        name = self.embedding_model if self.embedding_backend != "lsa" else "lsa"
        return name.replace("/", "__")

    @property
    def index_dir(self) -> Path:
        return self.artifacts_dir / "indice" / self.model_slug

    @property
    def thresholds_path(self) -> Path:
        return self.artifacts_dir / f"limiares__{self.model_slug}.json"

    @property
    def query_prefix(self) -> str:
        return EMBEDDING_PREFIXES.get(self.embedding_model, ("", ""))[0] if self.embedding_backend != "lsa" else ""

    @property
    def passage_prefix(self) -> str:
        return EMBEDDING_PREFIXES.get(self.embedding_model, ("", ""))[1] if self.embedding_backend != "lsa" else ""

    def load_calibrated_thresholds(self) -> bool:
        """Carrega limiares calibrados, se existirem. Retorna True se carregou."""
        if self.thresholds_path.exists():
            data = json.loads(self.thresholds_path.read_text(encoding="utf-8"))
            self.relevance_threshold = float(data.get("relevance_threshold", self.relevance_threshold))
            self.scope_threshold = float(data.get("scope_threshold", self.scope_threshold))
            self.min_evidences = int(data.get("min_evidences", self.min_evidences))
            return True
        return False


def get_settings(**overrides) -> Settings:
    s = Settings()
    for k, v in overrides.items():
        if not hasattr(s, k):
            raise AttributeError(f"Parâmetro desconhecido: {k}")
        setattr(s, k, v)
    return s

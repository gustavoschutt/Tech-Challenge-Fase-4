"""Recuperação híbrida: densa (embeddings) + léxica (BM25), fusão por
Reciprocal Rank Fusion (RRF), deduplicação de textos idênticos e
re-ranking com cross-encoder.

Por que híbrida: a busca densa acha paráfrases ("não chegou" ~ "nunca
recebi"); o BM25 acha termos exatos e raros (marcas, "nota fiscal",
"Correios") que os embeddings diluem. O RRF combina os dois RANKINGS sem
precisar calibrar escalas de score diferentes.

Por que deduplicar: ~17% dos textos se repetem literalmente ("muito bom",
"recomendo"). Sem deduplicação, o contexto do LLM seria ocupado por cópias
da mesma frase e a diversidade de evidências cairia.

Por que re-ranking: o bi-encoder compara vetores calculados separadamente;
o cross-encoder lê pergunta e avaliação JUNTAS e produz um score de
relevância mais preciso, que também é usado pelo gate de abstenção.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .config import Settings
from .indexing import KnowledgeIndex
from .query import QueryPlan
from .text_utils import tokenize_lexical


@dataclass
class Evidence:
    pos: int                      # posição em documentos.parquet / no índice
    review_id: str
    order_id: str
    texto: str
    nota: int
    data_compra: str
    categoria: str
    uf: str
    situacao_entrega: str
    atraso_dias: float | None
    rank_denso: int | None = None
    rank_bm25: int | None = None
    score_denso: float = 0.0
    score_bm25: float = 0.0
    score_rrf: float = 0.0
    relevancia: float = 0.0       # score final usado no gate (0–1 com reranker)
    n_textos_identicos: int = 1   # quantas avaliações do subconjunto têm o mesmo texto
    rotulo: str = ""              # [E1], [E2]... atribuído na montagem do contexto

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievalResult:
    candidatos: list[Evidence]
    n_documentos_filtrados: int
    score_escopo: float
    usou_reranker: bool
    tempos: dict = field(default_factory=dict)
    termos_ausentes: dict = field(default_factory=dict)   # palavra -> 0
    termos_raros: dict = field(default_factory=dict)      # palavra -> df


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_d)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        from .embeddings import quiet_hf

        quiet_hf()
        self.model = CrossEncoder(model_name, max_length=256)

    def scores(self, query: str, texts: list[str]) -> np.ndarray:
        raw = np.asarray(self.model.predict([(query, t) for t in texts], convert_to_numpy=True,
                                            show_progress_bar=False), dtype=np.float64)
        # Alguns modelos/versões já aplicam sigmoide; outros devolvem logits.
        if raw.min() < 0.0 or raw.max() > 1.0:
            raw = np.array([_sigmoid(x) for x in raw])
        return raw


class HybridRetriever:
    def __init__(self, kindex: KnowledgeIndex, settings: Settings):
        self.k = kindex
        self.s = settings
        self.docs = kindex.docs
        self._reranker: Reranker | None = None
        self._dedup_keys = self.docs["chave_dedup"].values

    # ------------------------------------------------------------------
    @property
    def reranker(self) -> Reranker | None:
        if self._reranker is None and self.s.reranker_model:
            self._reranker = Reranker(self.s.reranker_model)
        return self._reranker

    def _dense(self, qv: np.ndarray, ids: np.ndarray, n: int) -> tuple[list[int], dict[int, float]]:
        n = min(n, len(ids))
        if n == 0:
            return [], {}
        if self.k.faiss_index is not None and len(ids) < len(self.docs):
            import faiss

            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(ids.astype(np.int64)))
            dist, pos = self.k.faiss_index.search(qv[None, :], n, params=params)
        elif self.k.faiss_index is not None:
            dist, pos = self.k.faiss_index.search(qv[None, :], n)
        else:
            sims = self.k.vectors[ids] @ qv
            order = np.argsort(-sims)[:n]
            pos, dist = ids[order][None, :], sims[order][None, :]
        pairs = [(int(p), float(d)) for p, d in zip(pos[0], dist[0]) if p >= 0]
        return [p for p, _ in pairs], dict(pairs)

    def _bm25(self, question: str, mask: np.ndarray, n: int) -> tuple[list[int], dict[int, float]]:
        tokens = tokenize_lexical(question)
        if not tokens:
            return [], {}
        scores = np.asarray(self.k.bm25.get_scores(tokens))
        scores = np.where(mask, scores, 0.0)
        top = np.argsort(-scores)[:n]
        top = [int(i) for i in top if scores[i] > 0]
        return top, {i: float(scores[i]) for i in top}

    def scope_score(self, qv: np.ndarray, top: int = 5) -> float:
        """Média do cosseno dos `top` documentos mais próximos em TODA a base
        (sem filtros). Pergunta fora do domínio -> nada parecido na base."""
        if self.k.faiss_index is not None:
            dist, _ = self.k.faiss_index.search(qv[None, :], top)
            return float(np.mean(dist[0]))
        sims = self.k.vectors @ qv
        return float(np.mean(np.sort(sims)[-top:]))

    # ------------------------------------------------------------------
    def lexical_coverage(self, plan: QueryPlan) -> tuple[dict, dict]:
        """Termos de conteúdo ausentes da base (df=0) e raros (df < MIN_EVIDENCES)."""
        df = self.k.doc_freq
        ausentes = {w: 0 for stem, w in plan.termos_conteudo.items() if df.get(stem, 0) == 0}
        raros = {w: df[stem] for stem, w in plan.termos_conteudo.items()
                 if 0 < df.get(stem, 0) < self.s.min_evidences}
        return ausentes, raros

    def retrieve(self, plan: QueryPlan, use_reranker: bool = True, mode: str = "hibrida",
                 extra_mask: np.ndarray | None = None, query_text: str | None = None,
                 n_rerank: int | None = None) -> RetrievalResult:
        """mode: 'hibrida' | 'densa' | 'bm25' (as duas últimas para avaliação).
        extra_mask: restrição adicional (ex.: só avaliações de um tema).
        query_text: texto de busca diferente da pergunta (ex.: pergunta + descrição do tema)."""
        t = {}
        ausentes, raros = self.lexical_coverage(plan)
        query = query_text or plan.pergunta
        t0 = time.perf_counter()
        mask = plan.filtros.mask(self.docs)
        if extra_mask is not None:
            mask = mask & extra_mask
        ids = np.flatnonzero(mask)
        t["filtro_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        qv = self.k.embedder.encode_query(query)
        scope = self.scope_score(qv)
        t["embedding_pergunta_ms"] = (time.perf_counter() - t0) * 1000
        if len(ids) == 0:
            return RetrievalResult([], 0, scope, False, t, ausentes, raros)

        t0 = time.perf_counter()
        dense_rank, dense_scores = ([], {}) if mode == "bm25" else self._dense(qv, ids, self.s.n_candidates_dense)
        t["busca_densa_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        bm25_rank, bm25_scores = ([], {}) if mode == "densa" else self._bm25(query, mask, self.s.n_candidates_bm25)
        t["busca_bm25_ms"] = (time.perf_counter() - t0) * 1000

        fused = rrf_fuse([r for r in (dense_rank, bm25_rank) if r], k=self.s.rrf_k)
        dense_pos = {p: i + 1 for i, p in enumerate(dense_rank)}
        bm25_pos = {p: i + 1 for i, p in enumerate(bm25_rank)}

        # Deduplicação de textos idênticos (mantém o de melhor posição).
        subset_counts: dict[str, int] = {}
        for key in self._dedup_keys[ids]:
            subset_counts[key] = subset_counts.get(key, 0) + 1
        seen, cands = set(), []
        for pos, rrf in fused:
            key = self._dedup_keys[pos]
            if key in seen:
                continue
            seen.add(key)
            row = self.docs.iloc[pos]
            cands.append(Evidence(
                pos=pos, review_id=row["review_id"], order_id=row["order_id"], texto=row["texto"],
                nota=int(row["review_score"]), data_compra=str(row["order_purchase_timestamp"])[:10],
                categoria=str(row["categoria"]), uf=str(row["customer_state"]),
                situacao_entrega=str(row["situacao_entrega"]),
                atraso_dias=None if row["atraso_dias"] != row["atraso_dias"] else float(row["atraso_dias"]),
                rank_denso=dense_pos.get(pos), rank_bm25=bm25_pos.get(pos),
                score_denso=float(self.k.vectors[pos] @ qv), score_bm25=bm25_scores.get(pos, 0.0),
                score_rrf=rrf, n_textos_identicos=subset_counts.get(key, 1),
            ))
            if len(cands) >= (n_rerank or self.s.n_rerank):
                break

        t0 = time.perf_counter()
        used_rr = False
        if use_reranker and cands and self.reranker is not None:
            rel = self.reranker.scores(query, [c.texto for c in cands])
            for c, r in zip(cands, rel):
                c.relevancia = float(r)
            used_rr = True
        else:
            # Sem reranker, o cosseno denso serve só como medida de relevância
            # para o gate; a ORDEM continua a da fusão (RRF na híbrida, ranking
            # nativo na densa e no BM25). Reordenar pelo cosseno aqui faria a
            # híbrida virar uma cópia da densa.
            for c in cands:
                c.relevancia = c.score_denso
        t["rerank_ms"] = (time.perf_counter() - t0) * 1000
        if used_rr:
            cands.sort(key=lambda c: c.relevancia, reverse=True)
        return RetrievalResult(cands, int(len(ids)), scope, used_rr, t, ausentes, raros)

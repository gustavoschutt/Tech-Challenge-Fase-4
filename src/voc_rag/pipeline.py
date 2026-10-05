"""Orquestração do pipeline RAG.

pergunta -> plano (filtros + intenção) -> recuperação híbrida -> re-ranking
-> gate de evidências -> contexto -> LLM -> verificação de citações -> resposta

A camada analítica (analytics.py) é acionada para perguntas AGREGADAS quando
o modo é "auto" e o arquivo de temas existe.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .config import Settings, get_settings
from .evidence import (STATUS_NAO_FUNDAMENTADA, STATUS_OK, STATUS_SEM_EVIDENCIAS, assess, build_context,
                       gate_message)
from .generation import build_messages, verify_answer
from .indexing import KnowledgeIndex
from .llm import get_llm
from .prompts import PROMPT_VERSION
from .query import Filters, plan_query
from .retrieval import HybridRetriever

AVISO_AMOSTRA = ("Base da resposta: {n} avaliações recuperadas por similaridade semântica e léxica. "
                 "É uma amostra das avaliações mais relevantes, não uma contagem da base.")
AVISO_AGREGADA = ("Atenção: a pergunta pede frequência/padrões ({gatilhos}). No modo RAG a resposta descreve a "
                  "amostra recuperada e NÃO mede frequência; use o modo analítico para contagens sobre a base.")


@dataclass
class RAGResponse:
    pergunta: str
    status: str
    modo: str
    resposta: str
    avisos: list[str]
    evidencias: list[dict]
    plano: dict
    diagnostico: dict = field(default_factory=dict)
    estatisticas: dict | None = None   # preenchido pelo modo analítico

    def to_dict(self) -> dict:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [f"**Pergunta:** {self.pergunta}", "", f"**Status:** `{self.status}` · **modo:** `{self.modo}`",
                 f"**Filtros:** {self.plano.get('filtros_descricao', 'nenhum')}", ""]
        if self.resposta:
            lines += [self.resposta, ""]
        for a in self.avisos:
            lines.append(f"> {a}")
        if self.evidencias:
            lines += ["", "| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for e in self.evidencias:
                txt = e["texto"].replace("|", "/")
                lines.append(f"| {e.get('rotulo') or '-'} | `{e['review_id']}` | {e['nota']} | {e['data_compra'][:7]} | "
                             f"{e['categoria']} | {e['situacao_entrega']} | {e['relevancia']:.3f} | "
                             f"{'sim' if e.get('citada') else 'não'} | {txt} |")
        return "\n".join(lines)


class VoCRAG:
    def __init__(self, settings: Settings | None = None, load_llm: bool = True):
        self.s = settings or get_settings()
        self.thresholds_calibrados = self.s.load_calibrated_thresholds()
        self.index = KnowledgeIndex.load(self.s)
        self.retriever = HybridRetriever(self.index, self.s)
        self.known_categories = set(self.index.docs["categoria"].unique())
        self._llm = None
        self._load_llm = load_llm
        self._analytics = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(self.s)
        return self._llm

    @property
    def analytics(self):
        if self._analytics is None:
            try:
                from .analytics import AnalyticsEngine

                self._analytics = AnalyticsEngine.load(self.s, self.index, self.retriever)
            except (FileNotFoundError, ImportError):
                self._analytics = False
        return self._analytics or None

    # ------------------------------------------------------------------
    def ask(self, pergunta: str, filtros: Filters | None = None, modo: str = "auto",
            inferir_filtros: bool = True, gerar: bool = True) -> RAGResponse:
        """modo: 'auto' (agregada -> analítico, se disponível) | 'rag' | 'analitico'.
        gerar=False executa tudo até o gate (útil para avaliação sem LLM)."""
        t_total = time.perf_counter()
        plan = plan_query(pergunta, self.known_categories, filtros, inferir_filtros)
        use_analytics = modo == "analitico" or (modo == "auto" and plan.agregada)
        if use_analytics and self.analytics is not None:
            return self.analytics.answer(plan, self, gerar=gerar)
        return self._rag(plan, gerar, t_total)

    def _rag(self, plan, gerar: bool, t_total: float) -> RAGResponse:
        ret = self.retriever.retrieve(plan)
        desc = plan.filtros.describe()
        decision = assess(ret, self.s, desc, bool(plan.termos_conteudo), not plan.filtros.is_empty())
        diag = {
            "motivo_gate": decision.motivo,
            "score_escopo": round(ret.score_escopo, 4),
            "n_documentos_filtrados": ret.n_documentos_filtrados,
            "n_candidatos_unicos": len(ret.candidatos),
            "usou_reranker": ret.usou_reranker,
            "limiares": {"relevancia": self.s.relevance_threshold, "escopo": self.s.scope_threshold,
                         "min_evidencias": self.s.min_evidences, "calibrados": self.thresholds_calibrados},
            "modelo_embeddings": self.index.meta.get("embedding_model"),
            "reranker": self.s.reranker_model or None,
            "versao_prompt": PROMPT_VERSION,
            "tempos_ms": {k: round(v, 1) for k, v in ret.tempos.items()},
        }
        avisos: list[str] = []
        if decision.status != STATUS_OK:
            evs = [e.to_dict() for e in decision.evidencias]
            for e in evs:
                e["citada"] = False
            diag["tempos_ms"]["total"] = round((time.perf_counter() - t_total) * 1000, 1)
            return RAGResponse(plan.pergunta, decision.status, "rag", "", [gate_message(decision, desc)],
                               evs, plan.to_dict(), diag)

        contexto, used = build_context(decision.evidencias, self.s)
        if not gerar:
            evs = [dict(e.to_dict(), citada=False) for e in used]
            return RAGResponse(plan.pergunta, STATUS_OK, "rag", "", ["geração desativada (gerar=False)"],
                               evs, plan.to_dict(), diag)

        system, user = build_messages(plan.pergunta, desc, contexto)
        t0 = time.perf_counter()
        raw = self.llm.generate(system, user)
        diag["tempos_ms"]["geracao"] = round((time.perf_counter() - t0) * 1000, 1)
        diag["llm"] = self.llm.name
        diag["resposta_bruta_llm"] = raw
        answer, rep = verify_answer(raw, {e.rotulo for e in used})
        diag["verificacao"] = asdict(rep)
        cited = set(rep.citacoes_validas)
        evs = [dict(e.to_dict(), citada=e.rotulo in cited) for e in used]

        if rep.llm_declarou_sem_evidencia:
            status, answer = STATUS_SEM_EVIDENCIAS, ""
            avisos.append("O modelo avaliou que as avaliações recuperadas não respondem à pergunta. "
                          "Não vou responder para não gerar informação sem suporte nos dados.")
        elif not rep.fundamentada:
            status, answer = STATUS_NAO_FUNDAMENTADA, ""
            from .evidence import MENSAGENS
            avisos.append(MENSAGENS[STATUS_NAO_FUNDAMENTADA])
        else:
            status = STATUS_OK
            avisos.append(AVISO_AMOSTRA.format(n=len(used)))
            if plan.agregada:
                avisos.append(AVISO_AGREGADA.format(gatilhos=", ".join(plan.gatilhos_agregada)))
            if rep.expressoes_quantitativas:
                avisos.append("O texto gerado contém expressões quantitativas (" +
                              ", ".join(sorted(set(rep.expressoes_quantitativas))) +
                              ") que a amostra não sustenta; leia-as como descrição da amostra.")
            if rep.linhas_removidas_sem_citacao:
                avisos.append(f"{len(rep.linhas_removidas_sem_citacao)} linha(s) sem citação foram removidas da resposta.")
        diag["tempos_ms"]["total"] = round((time.perf_counter() - t_total) * 1000, 1)
        return RAGResponse(plan.pergunta, status, "rag", answer, avisos, evs, plan.to_dict(), diag)

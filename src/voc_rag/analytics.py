"""Camada analítica: responde perguntas AGREGADAS ("principais problemas",
"mais frequentes", "padrões") com CONTAGENS sobre a base inteira +
EVIDÊNCIAS exemplares recuperadas pelo RAG.

Por que existe: num RAG puro o LLM vê ~8 avaliações de ~42 mil. Ele não tem
como saber o que é frequente, e qualquer "a maioria reclama de X" seria
inventado. Aqui os números vêm do código (taxonomia de temas auditável,
themes.py) e o LLM só redige a síntese, obrigado a copiar os números da
tabela e a citar exemplos. Uma verificação pós-geração descarta linhas com
números que não estão na tabela.

Fluxo:
 plano da pergunta -> mesmas barreiras do gate RAG (termo ausente, escopo,
 termo raro) -> foco (problemas | elogios | ambos) e grupo (entrega,
 produto, atendimento...) -> recorte (filtros + termos de foco) -> base
 mínima para porcentagens -> contagem
 por tema -> exemplares por tema (busca híbrida + reranker restritos ao tema)
 -> tabela + exemplos -> LLM -> verificação de números e citações.
"""
from __future__ import annotations

import re
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import Settings
from .evidence import STATUS_INSUFICIENTE, STATUS_OK, STATUS_SEM_EVIDENCIAS
from .generation import GroundingReport, fix_citations, is_header
from .prompts import PROMPT_VERSION_ANALITICO, SEM_EVIDENCIA_TOKEN, SYSTEM_ANALITICO, USER_ANALITICO
from .query import QueryPlan
from .text_utils import tokenize_lexical
from .themes import NOTAS_ELOGIO, NOTAS_PROBLEMA, TAXONOMIA, THEMES_BY_ID

_FOCO_ELOGIO = re.compile(r"\b(elogi\w*|valoriz\w*|positiv\w*|gost\w*|pontos? fortes?|agrad\w*|bem avaliad\w*|destaque\w*|satisfeit\w*)\b")
_FOCO_PROBLEMA = re.compile(r"\b(problema\w*|reclam\w*|insatisf\w*|negativ\w*|queix\w*|ruim|ruins|pior\w*|critic\w*|dificuldade\w*|falha\w*|nota baixa|notas baixas|detrator\w*)\b")
GRUPOS = {
    "entrega": re.compile(r"\b(entreg\w*|prazo\w*|atras\w*|frete\w*|transport\w*|correio\w*|chega\w*|logistic\w*|demor\w*)\b"),
    "produto": re.compile(r"\b(qualidade|defeito\w*|quebr\w*|danific\w*|falsific\w*|original\w*|material\w*)\b"),
    "atendimento": re.compile(r"\b(atendiment\w*|vendedor\w*|suporte|contato\w*|comunicac\w*|sac|resposta\w*)\b"),
    "pos_venda": re.compile(r"\b(troca\w*|devolu\w*|reembols\w*|cancel\w*|estorn\w*)\b"),
    "documentacao": re.compile(r"\b(nota fiscal|nf|nfe)\b"),
}
# Palavras genéricas que NÃO restringem o recorte (aparecem em quase tudo).
GENERICOS = set("""compra compras comprar comprei compraram produto produtos pedido pedidos loja lojas site item itens
coisa coisas vez vezes geral relacao relacionad experiencia lado parte qualquer algum alguma todo toda
""".split())


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


class AnalyticsEngine:
    def __init__(self, settings: Settings, kindex, retriever, labels: pd.DataFrame):
        self.s = settings
        self.k = kindex
        self.r = retriever
        self.docs = kindex.docs
        self.labels = labels  # alinhado a documentos.parquet (mesma ordem)
        self.all_reviews = pd.read_parquet(settings.base_dir / "avaliacoes_todas.parquet")

    @classmethod
    def load(cls, settings: Settings, kindex, retriever) -> "AnalyticsEngine":
        path = settings.base_dir / "temas.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path} não existe. Rode: python scripts/03_construir_temas.py")
        lab = pd.read_parquet(path)
        if len(lab) != len(kindex.docs) or not (lab["review_id"].values == kindex.docs["review_id"].values).all():
            raise RuntimeError("temas.parquet desalinhado de documentos.parquet. Rode 03_construir_temas.py.")
        return cls(settings, kindex, retriever, lab.drop(columns=["review_id"]).astype(bool))

    # ------------------------------------------------------------------
    def _focus(self, plan: QueryPlan) -> tuple[list[str], list[str], str]:
        q = plan.pergunta_normalizada
        f = plan.filtros
        pol = []
        if f.nota_max is not None and f.nota_max <= 3:
            pol = ["problema"]
        elif f.nota_min is not None and f.nota_min >= 4:
            pol = ["elogio"]
        else:
            if _FOCO_PROBLEMA.search(q):
                pol.append("problema")
            if _FOCO_ELOGIO.search(q) and not re.search(r"\binsatisf", q):
                pol.append("elogio")
            if not pol:
                pol = ["problema", "elogio"]
        grupos = [g for g, rx in GRUPOS.items() if rx.search(q)]
        why = f"foco: {' + '.join(pol)}" + (f"; grupos: {', '.join(grupos)}" if grupos else "")
        return pol, grupos, why

    def _focus_terms(self, plan: QueryPlan) -> dict[str, str]:
        """Termos de conteúdo que restringem o recorte (ex.: 'cartucho')."""
        group_stems = set()
        for g, rx in GRUPOS.items():
            for m in rx.finditer(plan.pergunta_normalizada):
                group_stems.update(tokenize_lexical(m.group(0)))
        generic = set(tokenize_lexical(" ".join(GENERICOS)))
        # temas e palavras de foco também não restringem
        foco = set()
        for rx in (_FOCO_ELOGIO, _FOCO_PROBLEMA):
            for m in rx.finditer(plan.pergunta_normalizada):
                foco.update(tokenize_lexical(m.group(0)))
        return {st: w for st, w in plan.termos_conteudo.items() if st not in group_stems | generic | foco}

    # ------------------------------------------------------------------
    def answer(self, plan: QueryPlan, rag, gerar: bool = True):
        from .pipeline import RAGResponse

        t_total = time.perf_counter()
        diag: dict = {"versao_prompt": PROMPT_VERSION_ANALITICO, "modelo_embeddings": self.k.meta.get("embedding_model"),
                      "reranker": self.s.reranker_model or None}
        ausentes, raros = self.r.lexical_coverage(plan)
        if self.s.lexical_gate and ausentes:
            termos = ", ".join(f"'{w}'" for w in ausentes)
            msg = (f"Não encontrei avaliações relevantes: os termos {termos} não aparecem em nenhuma avaliação da base. "
                   "Se houve erro de digitação ou você usou um sinônimo, reformule. Não vou responder para não gerar "
                   "informação sem suporte nos dados.")
            return RAGResponse(plan.pergunta, STATUS_SEM_EVIDENCIAS, "analitico", "", [msg], [], plan.to_dict(),
                               dict(diag, motivo_gate=f"termos ausentes: {termos}"))

        # Mesmas barreiras do gate do caminho RAG (evidence.assess): uma pergunta
        # agregada também pode estar fora do escopo ou depender de um termo raro.
        # Nesses casos a decisão é delegada ao gate RAG, que explica a abstenção
        # e mostra as poucas menções existentes, sem contar frequências.
        motivo_desvio = None
        if plan.termos_conteudo:  # sem termos de conteúdo = pergunta sobre a base/recorte inteiro
            escopo = self.r.scope_score(self.k.embedder.encode_query(plan.pergunta))
            diag["score_escopo"] = round(escopo, 4)
            if escopo < self.s.scope_threshold:
                motivo_desvio = f"score de escopo {escopo:.3f} < limiar {self.s.scope_threshold:.3f}"
        if motivo_desvio is None and self.s.lexical_gate and raros:
            motivo_desvio = "termos raros na base: " + ", ".join(f"'{w}' ({n})" for w, n in raros.items())
        if motivo_desvio is not None:
            resp = rag._rag(plan, gerar, t_total)
            resp.avisos.append(f"Pergunta agregada não passou nas checagens da camada analítica ({motivo_desvio}); "
                               "a decisão foi tomada pelo gate do modo RAG, sem contagens.")
            resp.diagnostico["desvio_analitico"] = motivo_desvio
            return resp

        polaridades, grupos, why = self._focus(plan)
        temas = [t for t in TAXONOMIA if t.polaridade in polaridades and (not grupos or t.grupo in grupos)]
        if not temas:  # grupo sem temas daquela polaridade (ex.: elogios de nota fiscal)
            temas = [t for t in TAXONOMIA if t.polaridade in polaridades]
        foco_terms = self._focus_terms(plan)

        # Recorte = filtros explícitos/inferidos + termos de foco (qualquer um).
        mask = plan.filtros.mask(self.docs)
        if foco_terms:
            stems = set(foco_terms)
            has = np.array([bool(stems & set(tokenize_lexical(t))) for t in self.docs["texto"]])
            mask &= has
        all_mask = plan.filtros.mask(self.all_reviews)
        diag.update({"foco": why, "termos_de_foco": list(foco_terms.values()),
                     "temas_considerados": [t.id for t in temas]})

        score = self.docs["review_score"].values
        user_score_filter = plan.filtros.nota_min is not None or plan.filtros.nota_max is not None
        rows, bases = [], {}
        for pol in polaridades:
            lo, hi = NOTAS_PROBLEMA if pol == "problema" else NOTAS_ELOGIO
            base = mask if user_score_filter else mask & (score >= lo) & (score <= hi)
            n_base = int(base.sum())
            pol_temas = [t for t in temas if t.polaridade == pol]
            cobertura = float(self.labels.loc[base, [t.id for t in pol_temas]].any(axis=1).mean()) if n_base else 0.0
            bases[pol] = {"n_avaliacoes_com_texto": n_base, "notas": "filtro do usuário" if user_score_filter else f"{lo}–{hi}",
                          "cobertura_taxonomia": round(cobertura, 4)}
            for t in pol_temas:
                n = int((self.labels[t.id].values & base).sum())
                if n == 0:
                    continue
                nota_media = float(score[self.labels[t.id].values & base].mean())
                rows.append({"tema_id": t.id, "tema": t.nome, "polaridade": pol, "avaliacoes": n,
                             "pct_base": n / n_base if n_base else 0.0, "nota_media": round(nota_media, 2)})
        stats = pd.DataFrame(rows)
        n_recorte_total = int(all_mask.sum())
        n_recorte_texto = int(mask.sum())
        diag.update({"bases": bases, "n_recorte_total_avaliacoes": n_recorte_total,
                     "n_recorte_com_texto": n_recorte_texto})

        if n_recorte_texto == 0:
            return RAGResponse(plan.pergunta, STATUS_SEM_EVIDENCIAS, "analitico", "",
                               ["Nenhuma avaliação com comentário atende ao recorte pedido "
                                f"(filtros: {plan.filtros.describe()}; termos de foco: {', '.join(foco_terms.values()) or 'nenhum'})."],
                               [], plan.to_dict(), diag)
        if stats.empty:
            # Nenhum tema da taxonomia no recorte -> volta ao RAG, avisando.
            resp = rag._rag(plan, gerar, t_total)
            resp.avisos.insert(0, "Nenhum tema da taxonomia aparece neste recorte; a pergunta foi respondida pelo "
                                  "modo RAG (amostra recuperada, sem contagens).")
            return resp
        n_base_max = max(b["n_avaliacoes_com_texto"] for b in bases.values())
        if n_base_max < self.s.min_base_analitico:
            evs = self._examples(plan, stats.head(3), mask)[0]
            diag["motivo_gate"] = f"base de contagem com {n_base_max} avaliação(ões) < mínimo {self.s.min_base_analitico}"
            return RAGResponse(plan.pergunta, STATUS_INSUFICIENTE, "analitico", "",
                               [f"A base de contagem deste recorte tem apenas {n_base_max} avaliação(ões) com comentário "
                                f"(mínimo: {self.s.min_base_analitico}). É pouco para medir frequência, então não calculo "
                                "porcentagens. Alguns exemplos estão listados abaixo para consulta direta."],
                               [e for e in evs], plan.to_dict(), diag)

        stats = stats.sort_values(["polaridade", "avaliacoes"], ascending=[True, False])
        top = pd.concat([g.head(5) for _, g in stats.groupby("polaridade", sort=False)])
        evidencias, contexto = self._examples(plan, top, mask)
        tabela = self._table_text(top, bases)
        recorte = (f"{n_recorte_total} avaliações no recorte ({n_recorte_texto} com comentário); "
                   f"filtros: {plan.filtros.describe()}" +
                   (f"; termos de foco: {', '.join(foco_terms.values())}" if foco_terms else ""))
        estat = {"tabela": top.assign(pct_base=top["pct_base"].map(_fmt_pct)).to_dict(orient="records"),
                 "bases": bases, "recorte": recorte, "metodo": "contagem por regras léxicas (themes.py); "
                 "problemas contados em notas 1–3 e elogios em notas 4–5, salvo filtro de nota do usuário"}
        avisos = [f"Contagens calculadas sobre a base ({recorte}).",
                  "Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. "
                  + " ".join(f"Cobertura da taxonomia ({p}): {_fmt_pct(b['cobertura_taxonomia'])} das "
                             f"{b['n_avaliacoes_com_texto']} avaliações da base de contagem."
                             for p, b in bases.items())]

        if not gerar:
            return RAGResponse(plan.pergunta, STATUS_OK, "analitico", self._deterministic_summary(top, bases),
                               avisos + ["geração desativada (gerar=False): resumo determinístico."],
                               evidencias, plan.to_dict(), diag, estat)

        system = SYSTEM_ANALITICO
        user = USER_ANALITICO.format(pergunta=plan.pergunta, recorte=recorte, tabela=tabela, contexto=contexto)
        t0 = time.perf_counter()
        raw = rag.llm.generate(system, user)
        diag["tempo_geracao_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        diag["llm"] = rag.llm.name
        diag["resposta_bruta_llm"] = raw
        answer, rep, numeros_invalidos = self._verify(raw, {e["rotulo"] for e in evidencias}, tabela)
        diag["verificacao"] = dict(asdict(rep), numeros_fora_da_tabela=numeros_invalidos)
        cited = set(rep.citacoes_validas)
        for e in evidencias:
            e["citada"] = e["rotulo"] in cited
        diag["tempo_total_ms"] = round((time.perf_counter() - t_total) * 1000, 1)
        if rep.llm_declarou_sem_evidencia or not answer:
            # O LLM falhou ou recusou: a tabela determinística continua válida.
            avisos.append("A síntese do modelo de linguagem foi descartada (recusa, números fora da tabela ou "
                          "ausência de citações). Exibindo o resumo determinístico calculado pelo código.")
            return RAGResponse(plan.pergunta, STATUS_OK, "analitico", self._deterministic_summary(top, bases),
                               avisos, evidencias, plan.to_dict(), diag, estat)
        if numeros_invalidos:
            avisos.append(f"{len(numeros_invalidos)} linha(s) com números fora da tabela foram removidas.")
        return RAGResponse(plan.pergunta, STATUS_OK, "analitico", answer, avisos, evidencias, plan.to_dict(), diag, estat)

    # ------------------------------------------------------------------
    def _examples(self, plan: QueryPlan, top: pd.DataFrame, mask: np.ndarray, per_theme: int = 2):
        evidencias, blocks, used_ids = [], [], set()
        for _, row in top.iterrows():
            t = THEMES_BY_ID[row["tema_id"]]
            theme_mask = mask & self.labels[t.id].values
            lo, hi = NOTAS_PROBLEMA if t.polaridade == "problema" else NOTAS_ELOGIO
            if plan.filtros.nota_min is None and plan.filtros.nota_max is None:
                sc = self.docs["review_score"].values
                theme_mask = theme_mask & (sc >= lo) & (sc <= hi)
            ret = self.r.retrieve(plan, extra_mask=theme_mask, query_text=f"{plan.pergunta} {t.descricao}", n_rerank=12)
            got = 0
            lines = []
            for c in ret.candidatos:
                if c.review_id in used_ids:
                    continue
                used_ids.add(c.review_id)
                c.rotulo = f"E{len(evidencias) + 1}"
                d = c.to_dict()
                d["tema"] = t.nome
                d["citada"] = False
                evidencias.append(d)
                lines.append(f"[{c.rotulo}] nota {c.nota}/5 | compra {c.data_compra[:7]} | categoria {c.categoria}\n\"{c.texto}\"")
                got += 1
                if got >= per_theme:
                    break
            if lines:
                blocks.append(f"Tema: {t.nome}\n" + "\n".join(lines))
        return evidencias, "\n\n".join(blocks)

    @staticmethod
    def _table_text(top: pd.DataFrame, bases: dict) -> str:
        lines = []
        for pol, g in top.groupby("polaridade", sort=False):
            b = bases[pol]
            lines.append(f"{'PROBLEMAS' if pol == 'problema' else 'ELOGIOS'} — base: {b['n_avaliacoes_com_texto']} "
                         f"avaliações com comentário (notas {b['notas']})")
            for _, r in g.iterrows():
                lines.append(f"- {r['tema']}: {r['avaliacoes']} avaliações ({_fmt_pct(r['pct_base'])})")
        return "\n".join(lines)

    @staticmethod
    def _deterministic_summary(top: pd.DataFrame, bases: dict) -> str:
        out = []
        for pol, g in top.groupby("polaridade", sort=False):
            b = bases[pol]
            label = "Problemas mais citados" if pol == "problema" else "Elogios mais citados"
            items = "; ".join(f"{r['tema']}: {r['avaliacoes']} ({_fmt_pct(r['pct_base'])})" for _, r in g.iterrows())
            out.append(f"{label} entre {b['n_avaliacoes_com_texto']} avaliações com comentário (notas {b['notas']}): {items}.")
        return "\n".join(out)

    @staticmethod
    def _verify(raw: str, valid_labels: set[str], tabela: str):
        """Verificação da síntese analítica:
        - todo número da linha precisa existir na tabela (senão a linha sai);
        - cada linha precisa de citação válida [E#] OU de número da tabela;
        - citações inexistentes são removidas."""
        rep = GroundingReport()
        if SEM_EVIDENCIA_TOKEN in raw:
            rep.llm_declarou_sem_evidencia = True
            return "", rep, []
        allowed = set(re.findall(r"\d+(?:[.,]\d+)?", tabela))
        allowed |= {n.replace(",", ".") for n in allowed} | {n.replace(".", ",") for n in allowed}
        kept, invalid = [], []
        for line in raw.splitlines():
            if not line.strip():
                continue
            fixed, good, bad = fix_citations(line, valid_labels)
            rep.citacoes_invalidas.extend(bad)
            nums = re.findall(r"\d+(?:[.,]\d+)?", re.sub(r"\[E\d+\]", "", fixed))
            bad_nums = [n for n in nums if n not in allowed]
            if bad_nums:
                invalid.append({"linha": line.strip(), "numeros": bad_nums})
                continue
            if is_header(fixed) and not good:
                continue
            if good or nums:
                kept.append(fixed)
                rep.citacoes_validas.extend(good)
            else:
                rep.linhas_removidas_sem_citacao.append(line.strip())
        rep.citacoes_validas = list(dict.fromkeys(rep.citacoes_validas))
        return "\n".join(kept), rep, invalid

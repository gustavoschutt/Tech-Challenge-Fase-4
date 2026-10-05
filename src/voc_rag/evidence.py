"""Gate de evidências: decide SE o sistema pode responder, ANTES de chamar
o LLM, e monta o contexto.

Estados possíveis:
- ok                       : há evidências relevantes suficientes -> gerar.
- fora_do_escopo           : a pergunta não se parece com nada da base
                             (score de escopo abaixo do limiar).
- sem_evidencias           : nenhum documento passou nos filtros, ou
                             nenhum candidato atingiu o limiar de relevância.
- evidencias_insuficientes : o termo central da pergunta aparece em
                             1..(MIN_EVIDENCES-1) avaliações da base, ou há
                             1..(MIN_EVIDENCES-1) avaliações relevantes.
                             Mostramos o que foi achado, mas NÃO sintetizamos
                             padrão (n pequeno não sustenta generalização).

O gate é a primeira de três camadas contra respostas sem suporte; as outras
duas são a instrução no prompt (o LLM pode declarar SEM_EVIDENCIA) e a
verificação de citações depois da geração (generation.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .retrieval import Evidence, RetrievalResult

STATUS_OK = "ok"
STATUS_FORA_ESCOPO = "fora_do_escopo"
STATUS_SEM_EVIDENCIAS = "sem_evidencias"
STATUS_INSUFICIENTE = "evidencias_insuficientes"
STATUS_NAO_FUNDAMENTADA = "resposta_nao_fundamentada"

MENSAGENS = {
    STATUS_FORA_ESCOPO: (
        "A pergunta parece estar fora do escopo da base de conhecimento, que contém apenas "
        "avaliações de clientes da Olist (pedidos de 2016 a 2018). Não vou responder para não "
        "gerar informação sem suporte nos dados."
    ),
    STATUS_SEM_EVIDENCIAS: (
        "Não encontrei avaliações relevantes para responder a esta pergunta na base de "
        "conhecimento{filtros}{detalhe}. Não vou responder para não gerar informação sem suporte nos dados."
    ),
    STATUS_INSUFICIENTE: (
        "Encontrei apenas {n} avaliação(ões) relevante(s){filtros}. É pouco para identificar um "
        "padrão, então não vou generalizar. As evidências encontradas estão listadas abaixo para "
        "consulta direta."
    ),
    STATUS_NAO_FUNDAMENTADA: (
        "O modelo de linguagem produziu um texto que não pôde ser vinculado às evidências "
        "recuperadas (citações ausentes ou inválidas). A resposta foi descartada; as evidências "
        "recuperadas estão listadas abaixo."
    ),
}


@dataclass
class GateDecision:
    status: str
    motivo: str
    evidencias: list[Evidence] = field(default_factory=list)
    n_relevantes: int = 0


def assess(ret: RetrievalResult, settings: Settings, filtros_desc: str = "nenhum",
           plan_terms_present: bool = True, filters_active: bool = False) -> GateDecision:
    """Ordem das checagens: filtros -> escopo -> cobertura léxica -> relevância -> quantidade."""
    if ret.n_documentos_filtrados == 0:
        return GateDecision(STATUS_SEM_EVIDENCIAS, "nenhum documento atende aos filtros")
    filter_defined = not plan_terms_present and filters_active
    if not filter_defined and ret.score_escopo < settings.scope_threshold:
        motivo = f"score de escopo {ret.score_escopo:.3f} < limiar {settings.scope_threshold:.3f}"
        if ret.termos_ausentes:
            motivo += "; termos sem nenhuma ocorrência na base: " + ", ".join(f"'{w}'" for w in ret.termos_ausentes)
        return GateDecision(STATUS_FORA_ESCOPO, motivo)
    if settings.lexical_gate and ret.termos_ausentes:
        termos = ", ".join(f"'{w}'" for w in ret.termos_ausentes)
        return GateDecision(STATUS_SEM_EVIDENCIAS, f"termos sem nenhuma ocorrência na base: {termos}")
    if filter_defined:
        # Pergunta definida só pelos filtros ("o que dizem as avaliações
        # negativas de móveis em 2018?"): todo documento do recorte é
        # pertinente por construção; o ranking ordena, o limiar não corta.
        sel = ret.candidatos[: settings.top_k_context]
        if len(ret.candidatos) < settings.min_evidences:
            return GateDecision(STATUS_INSUFICIENTE, f"recorte com {len(ret.candidatos)} avaliação(ões) com texto",
                                sel, len(sel))
        return GateDecision(STATUS_OK, "pergunta definida pelos filtros; evidências = recorte filtrado",
                            sel, len(sel))
    if settings.lexical_gate and ret.termos_raros:
        # Termo central raro (aparece em 1..MIN_EVIDENCES-1 avaliações da base):
        # mostramos as menções recuperadas, sem generalizar. Vem ANTES do limiar
        # de relevância, como na ordem documentada acima: se a única menção
        # existente ficasse abaixo do limiar, o usuário receberia "nada
        # encontrado" quando há, sim, algo a mostrar.
        from .text_utils import tokenize_lexical

        rare_stems = set(tokenize_lexical(" ".join(ret.termos_raros)))
        mention = [c for c in ret.candidatos if rare_stems & set(tokenize_lexical(c.texto))]
        if mention:
            termos = ", ".join(f"'{w}' ({n} avaliação(ões))" for w, n in ret.termos_raros.items())
            return GateDecision(STATUS_INSUFICIENTE, f"termos raros na base: {termos}",
                                mention, len(mention))
    relevant = [c for c in ret.candidatos if c.relevancia >= settings.relevance_threshold]
    if not relevant:
        best = ret.candidatos[0].relevancia if ret.candidatos else float("nan")
        return GateDecision(STATUS_SEM_EVIDENCIAS,
                            f"melhor relevância {best:.3f} < limiar {settings.relevance_threshold:.3f}")
    if len(relevant) < settings.min_evidences:
        return GateDecision(STATUS_INSUFICIENTE,
                            f"{len(relevant)} relevante(s) < mínimo {settings.min_evidences}",
                            relevant, len(relevant))
    return GateDecision(STATUS_OK, f"{len(relevant)} evidências relevantes", relevant, len(relevant))


def gate_message(decision: GateDecision, filtros_desc: str = "nenhum") -> str:
    ftxt = "" if filtros_desc == "nenhum" else f" com os filtros aplicados ({filtros_desc})"
    detalhe = ""
    if decision.motivo.startswith("termos sem nenhuma ocorrência"):
        detalhe = (f" ({decision.motivo}; se houve erro de digitação ou você usou um sinônimo, "
                   "reformule a pergunta)")
    elif decision.motivo.startswith("termos raros"):
        detalhe = f" ({decision.motivo})"
    tmpl = MENSAGENS.get(decision.status, "")
    msg = tmpl.format(n=decision.n_relevantes, filtros=ftxt, detalhe=detalhe)
    if decision.status == STATUS_FORA_ESCOPO and "termos sem nenhuma ocorrência" in decision.motivo:
        termos = decision.motivo.split("termos sem nenhuma ocorrência na base: ", 1)[1]
        if termos.count("'") == 2:
            msg += f" Além disso, o termo {termos} não aparece em nenhuma avaliação da base."
        else:
            msg += f" Além disso, os termos {termos} não aparecem em nenhuma avaliação da base."
    if decision.status == STATUS_INSUFICIENTE and detalhe:
        msg = msg.replace(". É pouco", f"{detalhe}. É pouco", 1)
    return msg


def build_context(evidences: list[Evidence], settings: Settings) -> tuple[str, list[Evidence]]:
    """Formata as evidências numeradas [E1..En] com metadados, respeitando
    o orçamento de caracteres. Devolve o texto e a lista efetivamente usada."""
    blocks, used, total = [], [], 0
    for ev in evidences[: settings.top_k_context]:
        ev.rotulo = f"E{len(used) + 1}"
        entrega = {"atrasada": f"atrasada {int(ev.atraso_dias)} dia(s)" if ev.atraso_dias else "atrasada",
                   "no_prazo": "no prazo", "nao_entregue": "não entregue"}.get(ev.situacao_entrega, ev.situacao_entrega)
        rep = f" | texto idêntico em {ev.n_textos_identicos} avaliações" if ev.n_textos_identicos > 1 else ""
        block = (f"[{ev.rotulo}] nota {ev.nota}/5 | compra {ev.data_compra[:7]} | categoria {ev.categoria} | "
                 f"entrega {entrega} | UF {ev.uf}{rep}\n\"{ev.texto}\"")
        if total + len(block) > settings.max_context_chars and used:
            break
        blocks.append(block)
        used.append(ev)
        total += len(block)
    return "\n\n".join(blocks), used

"""Taxonomia de temas da Voz do Cliente (camada analítica).

Cada avaliação com texto recebe zero ou mais temas (multirrótulo) por regras
léxicas explícitas sobre o texto normalizado (minúsculas, sem acento).

Por que regras e não um classificador/LLM:
- AUDITÁVEL: qualquer contagem pode ser explicada pela regra e conferida
  abrindo as avaliações rotuladas (rastreabilidade do insight até o dado).
- DETERMINÍSTICO e barato: roda nas ~42 mil avaliações em segundos, sem GPU
  e sem custo de API.
- A taxonomia foi definida e refinada a partir da exploração dos dados
  (leitura de amostras, inclusive das avaliações negativas que ficavam sem
  tema; clusters de embeddings no notebook 04). A precisão de cada tema é
  medida por anotação humana na amostra eval/auditoria_temas.csv (15 por
  tema); o notebook 04 calcula o resultado.
Custo assumido: recall incompleto (paráfrases não previstas ficam "sem
tema"). A cobertura é sempre informada junto com as contagens.

Negação: para temas de ELOGIO, uma ocorrência precedida de negação
("não recomendo", "nada rápido") é descartada.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from .text_utils import normalize_for_rules


@dataclass(frozen=True)
class Theme:
    id: str
    nome: str
    polaridade: str          # "problema" | "elogio"
    grupo: str               # entrega | produto | atendimento | pos_venda | documentacao | geral
    padrao: str
    descricao: str           # usada como consulta para escolher exemplares
    checar_negacao: bool = False
    excluir: str | None = None
    _rx: re.Pattern = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "_rx", re.compile(self.padrao))
        object.__setattr__(self, "_ex", re.compile(self.excluir) if self.excluir else None)

    def matches(self, text_norm: str) -> bool:
        if self._ex is not None and self._ex.search(text_norm):
            return False
        for m in self._rx.finditer(text_norm):
            if not self.checar_negacao:
                return True
            before = text_norm[max(0, m.start() - 14): m.start()]
            if not _NEG.search(before):
                return True
        return False


_NEG = re.compile(r"\b(nao|nem|nunca|jamais|nada|sem)\b[^.!?]{0,10}$")

# Base de contagem: temas de PROBLEMA são contados nas avaliações com nota
# 1–3; temas de ELOGIO nas avaliações com nota 4–5 (a não ser que o usuário
# filtre a nota explicitamente). Isso evita contar "com nota fiscal, tudo
# certo" (nota 5) como problema de nota fiscal, ao custo de ignorar
# ressalvas em avaliações positivas ("ótimo, mas atrasou") — documentado.
NOTAS_PROBLEMA = (1, 3)
NOTAS_ELOGIO = (4, 5)

TAXONOMIA: list[Theme] = [
    # ------------------------------------------------------------- problemas
    Theme("nao_recebido", "Pedido não recebido", "problema", "entrega",
          r"\b(nao|ainda nao|nunca) (recebi|recebo|recebemos|chegou|chegaram|foi entregue|foram entregues|me entregaram|entregaram)\b"
          r"(?! (a |o )?(nota|nf|boleto|e-?mail|resposta|retorno|contato|manual))"
          r"|\bcade (o|meu|minha) (produto|pedido|encomenda)"
          r"|\b(aguardando|esperando|aguardo) (o |a |meu |minha )?(produto|pedido|encomenda|entrega|mercadoria)",
          "Cliente diz que o pedido ainda não chegou ou nunca foi entregue."),
    Theme("atraso", "Atraso na entrega", "problema", "entrega",
          r"\batras|\bdemor|\bfora do prazo|\b(passou|ultrapass\w*|excede\w*|estour\w*) (do |o )?prazo|\bdepois do prazo"
          r"|\bprazo (de entrega )?(nao (foi )?cumprido|vencido|expirado|esgotado)|\bnao (cumpr\w+|respeit\w+) (o )?prazo"
          r"|\bvenc\w* o prazo|\bnao (ter )?(chegou|chegado|chega\w*) (a tempo|antes d)",
          "Entrega atrasada, demorou mais que o prazo prometido."),
    Theme("incompleto", "Pedido incompleto / faltando itens", "problema", "entrega",
          r"\bfalt(ou|ando|aram|a) (?!(a |o )?(nota|nf|resposta|informac|respeito|compromisso|atencao|cuidado|consideracao|comunicacao))(um|uma|o|a|os|as|\d|item|itens|produto|pecas?|parte|unidade)"
          r"|\b(veio|vieram|chegou|chegaram|recebi|recebemos|entregaram|enviaram|mandaram) (so|apenas|somente) "
          r"|\b(so|apenas|somente) (recebi|chegou|veio|vieram) |\bincomplet"
          r"|\b(recebi|chegou|veio) (um|uma|1|2|dois|duas) (dos|das|de) \d"
          r"|\bnao (recebi|chegou|veio) (o|a|os|as) (outro|outra|outros|outras|restante|segundo|segunda|demais)",
          "Cliente recebeu só parte dos itens comprados; faltaram produtos no pedido."),
    Theme("produto_errado", "Produto errado / diferente do anunciado", "problema", "produto",
          r"\b(produto|item|cor|modelo|tamanho|voltagem|numero|mercadoria)s? (errad|diferente|trocad)"
          r"|\b(veio|vieram|chegou|enviaram|mandaram|recebi|entregaram) (o |a |um |uma )?(produto |item |modelo |cor )?(errad|outr[oa]s? (produto|cor|modelo|item)|diferente)"
          r"|\bnao (e|era|eh|corresponde\w*|condiz\w*) (com )?(o|ao|a|as|aos)? ?(que|produto|anunciad|pedid|compr|descri|foto|imagem)"
          r"|\bnao (e|era|eh) (como|igual|o mesmo|a mesma) ?(na|a|ao|o|as|da|do)? ?(foto|imagem|anuncio|descri|site|que)"
          r"|\b(entregue|entregaram|enviado|enviaram|recebido) (foi )?(completamente |totalmente |bem )?diferente|\bentreg\w* outro\b"
          r"|\bdiferente d[oa]s? (anunciad|foto|imagem|pedid|descri|site|que (eu )?(comprei|pedi|solicitei))",
          "Produto entregue diferente do anunciado ou do pedido: modelo, cor ou item errado."),
    Theme("danificado", "Produto danificado / avariado", "problema", "produto",
          r"\bquebr|\bdanific|\bavari|\bamassad|\btrincad|\brachad|\barranhad|\briscad|\bestragad|\bvazand|\bvazament|\bfurad|\brasgad|\bamassou",
          "Produto chegou quebrado, danificado, amassado ou avariado.", checar_negacao=True),
    Theme("defeito", "Defeito / não funciona", "problema", "produto",
          r"\bdefeit|\bnao (funciona|funcionou|funcionam|liga|ligou|carrega|reconhec\w*)\b|\bparou de funcionar"
          r"|\bdeu (problema|defeito)|\bcom problema",
          "Produto com defeito, não funciona ou não é reconhecido."),
    Theme("qualidade_ruim", "Qualidade abaixo do esperado", "problema", "produto",
          r"\b(baixa|pessima|ma|ruim|horrivel|pouca) qualidade|\bqualidade (ruim|baixa|pessima|inferior|horrivel|duvidosa|deixa)"
          r"|\bmaterial (ruim|fraco|fino|fragil|vagabundo|pessimo)|\bvagabund|\bmal acabad|\bacabamento (ruim|pessimo|mal)"
          r"|\b(produto|material|qualidade|acabamento)[^.!?]{0,25}(deix\w*|ficou) (muito )?a desejar|\bprecari|\bfragil|\bmuito (fino|fraco|frageis?)\b"
          r"|\bmaterial (muito )?(ruim|fraco|fino|mole|pessimo)|\bplastico (horrivel|ruim|fraco|mole|vagabundo)",
          "Qualidade do produto ruim, material fraco, abaixo da expectativa."),
    Theme("falsificado", "Produto falsificado / não original", "problema", "produto",
          r"\bfalsific|\bpirata|\breplica\b|\bnao (e|era|eh) original|\bnao original|\bparalelo\b|\bproduto falso|\bfalso\b"
          r"|\bimitac|\bnao (e|era) (o )?verdadeiro",
          "Produto falsificado, pirata, réplica ou não original."),
    Theme("atendimento_ruim", "Atendimento / falta de resposta", "problema", "atendimento",
          r"\b(sem|nenhum|nenhuma|nao (tive|tenho|obtive|recebi|recebo|tivemos)) (nenhum |nenhuma |qualquer )?(resposta|retorno|contato|posicionamento|satisfacao|informac\w*|suporte)"
          r"|\bnao (me )?(respond\w*|retorn\w*)\b|\bnao atend\w* (o telefone|as ligac|ligac|telefone)|\bninguem (respond\w*|atend\w*|resolv\w*|retorn\w*|da)"
          r"|\batendimento (ruim|pessimo|horrivel|demorado|precario|fraco|lamentavel)|\bdescaso"
          r"|\bfalta de (respeito|compromisso|comunicacao|informac\w*|atencao|consideracao)|\bdificil (contato|comunicacao|falar)"
          r"|\bnao consigo (contato|falar|contatar)|\bsac (e|eh|foi) (uma )?(porcaria|pessimo|ruim|horrivel)",
          "Vendedor ou loja não responde, falta de retorno e atendimento ruim."),
    Theme("sem_estoque", "Venda sem estoque / produto indisponível", "problema", "pos_venda",
          r"\b(sem|nao (tinha|tinham|tem|havia|possui\w*)( o produto| a mercadoria)?( em)?) estoque|\bindisponiv|\bfalta de estoque|\besgotad",
          "Loja vendeu produto sem estoque; produto indisponível e pedido cancelado."),
    Theme("insatisfacao_geral", "Não recomenda / insatisfação geral", "problema", "geral",
          r"\bnao recomend|\bpessim|\bhorrivel|\bdecepcion|\bdeix\w* a desejar|\bficou a desejar|\bnao gostei|\bjamais compr"
          r"|\bnunca mais|\bnao compr\w* mais|\binsatisfeit|\bporcaria|\blamentavel|\barrependid|\bpior (compra|loja|experiencia)",
          "Cliente insatisfeito, não recomenda, experiência péssima."),
    Theme("troca_reembolso", "Troca, devolução, cancelamento ou reembolso", "problema", "pos_venda",
          r"\btroca\b|\btrocar\b|\bdevolu|\bdevolv|\breembols|\bestorn|\bdinheiro de volta|\bcancel",
          "Cliente pede ou relata troca, devolução, cancelamento ou reembolso."),
    Theme("nota_fiscal", "Nota fiscal", "problema", "documentacao",
          r"\b(sem|nao (recebi|veio|vieram|enviaram|emitiram|foi emitida|chegou|mandaram)|falt\w*|nem) (a |o )?(nota|nf|nfe|danfe|cupom fiscal)"
          r"|\b(nota fiscal|nf|nfe)\w* (veio |foi |esta |emitida )?(errad|incorret|com erro|divergent|nao (veio|foi|chegou))",
          "Problemas com a nota fiscal: não enviada, errada ou não emitida."),
    Theme("transportadora", "Transportadora / Correios", "problema", "entrega",
          r"\bcorreio|\btransportador|\bentregador|\bextravi|\bretirar (na|no|nos) (agencia|correio)|\bdevolvid[oa] (ao|para o) remetente",
          "Problemas com Correios, transportadora ou entregador; extravio; retirada na agência.",
          checar_negacao=False),
    # --------------------------------------------------------------- elogios
    Theme("entrega_rapida", "Entrega rápida / no prazo", "elogio", "entrega",
          r"\bantes do (prazo|previsto|combinado|esperado|tempo)|\bchegou (rapid|cedo|antes|super rapid|muito rapid|bem rapid|no prazo|dentro do prazo)"
          r"|\bentrega (rapida|super rapida|muito rapida|antecipada|eficiente|otima|excelente|perfeita|no prazo|dentro do prazo|pontual|em dia)"
          r"|\brapidez|\b(no|dentro do) prazo|\bbem antes|\bentregue (rapid|antes|no prazo)|\bentrega foi rapida",
          "Entrega rápida, chegou antes do prazo ou dentro do prazo.", checar_negacao=True,
          excluir=r"\b(fora|passou|depois|alem) do prazo|\batras"),
    Theme("qualidade_boa", "Boa qualidade do produto", "elogio", "produto",
          r"\b(otima|boa|excelente|alta|muito boa|exelente|perfeita) qualidade|\bqualidade (otima|boa|excelente|superior|muito boa|impecavel)"
          r"|\bbem feit|\bmaterial (bom|otimo|excelente|de qualidade|resistente)|\b(otimo|excelente|lindo|maravilhoso|perfeito) produto"
          r"|\bproduto (otimo|excelente|de qualidade|muito bom|perfeito|maravilhoso|lindo|bom|show)",
          "Produto de boa qualidade, bem feito, excelente.", checar_negacao=True),
    Theme("conforme_anunciado", "Conforme o anunciado", "elogio", "produto",
          r"\bconforme (o )?(anunciad|descri|anuncio|a foto|esperad|pedid|combinad|solicitad)|\bigual (a|ao|as) (foto|anuncio|descri|imagem)"
          r"|\bexatamente (como|o que|igual)|\bcomo (descrito|anunciado|esperado)|\btudo (certo|ok|perfeito|correto|certinho|conforme)"
          r"|\bcertinho|\bcorrespondeu",
          "Produto conforme o anunciado, tudo certo, exatamente como descrito.", checar_negacao=True),
    Theme("atendimento_bom", "Bom atendimento", "elogio", "atendimento",
          r"\b(otimo|bom|excelente|exelente|atencioso|educado|rapido) atendimento|\batendimento (otimo|bom|excelente|nota 10|rapido|atencioso|eficiente|perfeito)"
          r"|\batencios|\bprestativ|\bvendedor (otimo|excelente|atencioso|honesto|confiavel)",
          "Atendimento bom, vendedor atencioso e prestativo.", checar_negacao=True),
    Theme("embalagem_boa", "Embalagem bem feita", "elogio", "entrega",
          r"\bbem embalad|\bembalagem (otima|boa|excelente|perfeita|caprichada|segura|impecavel|resistente)|\bbem (protegid|acondicionad)|\bcaprich",
          "Produto bem embalado, embalagem caprichada e segura.", checar_negacao=True),
    Theme("preco_bom", "Preço / custo-benefício", "elogio", "produto",
          r"\b(bom|otimo|excelente|justo|melhor) preco|\bpreco (bom|otimo|justo|baixo|acessivel|em conta|excelente)|\bcusto.?beneficio|\bbarat(o|a|os|as|issimo|inho)\b|\bvale a pena",
          "Bom preço, ótimo custo-benefício, vale a pena.", checar_negacao=True),
    Theme("recomenda", "Recomenda / satisfação geral", "elogio", "geral",
          r"\brecomend|\bsatisfeit|\badorei|\bamei\b|\bgostei|\bparabens|\bnota 10|\bperfeito|\bexcelente|\botim[oa]|\bmuito bo[am]",
          "Cliente satisfeito, recomenda a loja e o produto.", checar_negacao=True,
          excluir=r"\binsatisfeit"),
]

THEMES_BY_ID = {t.id: t for t in TAXONOMIA}


def label_texts(texts: pd.Series) -> pd.DataFrame:
    """Matriz booleana (avaliações x temas)."""
    norm = texts.fillna("").map(normalize_for_rules)
    return pd.DataFrame({t.id: norm.map(t.matches) for t in TAXONOMIA}, index=texts.index)

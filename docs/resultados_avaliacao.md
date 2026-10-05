# Resultados da avaliação

> Gerado por `scripts/05_avaliar.py` em 2026-10-04 19:58. Arquivos brutos: `eval/resultados/intfloat__multilingual-e5-base/`.

## Configuração

- Embeddings: `intfloat/multilingual-e5-base`
- Reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- LLM: `gemini:gemini-3.8-flash`
- Limiares do gate: escopo = 0.8316 e relevância = 0.053 (calibrados no conjunto de calibração: True); mínimo de evidências = 3 e base mínima da camada analítica = 30 (fixados, não calibrados)

## Recuperação (perguntas com relevância *silver*)

| modo             |   precisao@8 |    rr |
|:-----------------|-------------:|------:|
| bm25             |        0.672 | 0.938 |
| densa            |        0.797 | 0.938 |
| hibrida          |        0.875 | 1     |
| hibrida+reranker |        0.922 | 1     |

> A relevância *silver* é uma regex do tema. Em 7 das 8 perguntas ela casa termos da própria pergunta, o que favorece o BM25 e a busca híbrida: leia a tabela como comparação indicativa entre modos.

## Filtros de metadados

Extração dos filtros: 100% de acerto (5/5)

Efeito do pré-filtro: fração das 8 evidências que pertencem ao recorte pedido, com e sem o filtro (mesma pergunta).

| id   | filtros                                                                                                                                                                                                             |   com filtro |   sem filtro |
|:-----|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------:|-------------:|
| T04  | nota 1–2                                                                                                                                                                                                            |            1 |        0.5   |
| T06  | nota 1–2                                                                                                                                                                                                            |            1 |        0.5   |
| T07  | categoria ∈ {beleza_saude}                                                                                                                                                                                          |            1 |        0.125 |
| T15  | nota 1–2; compra de 2018-01-01 até 2019-01-01 (exclusivo); categoria ∈ {moveis_colchao_e_estofado, moveis_cozinha_area_de_servico_jantar_e_jardim, moveis_decoracao, moveis_escritorio, moveis_quarto, moveis_sala} |            1 |        0.125 |
| T16  | UF ∈ {SP}; entrega ∈ {atrasada}                                                                                                                                                                                     |            1 |        0     |

Média: com filtro 1.000; sem filtro 0.250.

## Gate de abstenção (status antes do LLM)

Critério tolerante: qualquer abstenção conta como acerto quando a pergunta não é respondível. Critério estrito: o status precisa ser exatamente o do tipo da pergunta.

| tipo          |   perguntas |   acerto_tolerante |   acerto_estrito |
|:--------------|------------:|-------------------:|-----------------:|
| fora_escopo   |           6 |              0.833 |            0.5   |
| insuficiente  |           4 |              0.75  |            0.5   |
| respondivel   |          16 |              0.875 |            0.875 |
| sem_evidencia |           5 |              1     |            0.8   |

Acerto geral: tolerante 87.1%; estrito 74.2%

Divergências do gate (critério estrito):

| id   | tipo          | status_gate              | aceito_no_tolerante   | pergunta                                                                                                      |
|:-----|:--------------|:-------------------------|:----------------------|:--------------------------------------------------------------------------------------------------------------|
| T05  | respondivel   | sem_evidencias           | False                 | Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega? |
| T07  | respondivel   | sem_evidencias           | False                 | O que os clientes mais elogiam em produtos de beleza e saúde?                                                 |
| T17  | sem_evidencia | fora_do_escopo           | True                  | O que os clientes acham de pagar com Pix?                                                                     |
| T23  | insuficiente  | sem_evidencias           | True                  | Como foi o atendimento pelo WhatsApp segundo os clientes?                                                     |
| T25  | insuficiente  | ok                       | False                 | Que problemas os clientes tiveram com geladeiras compradas?                                                   |
| T27  | fora_escopo   | sem_evidencias           | True                  | Qual será a taxa Selic no fim do ano que vem?                                                                 |
| T28  | fora_escopo   | sem_evidencias           | True                  | Escreva um poema sobre o pôr do sol na praia.                                                                 |
| T30  | fora_escopo   | evidencias_insuficientes | False                 | Como faço para trocar o óleo do motor do meu carro?                                                           |

## Ponta a ponta (com LLM, modo automático)

| tipo          |   perguntas |   acerto_tolerante |   acerto_estrito |
|:--------------|------------:|-------------------:|-----------------:|
| fora_escopo   |           6 |              0.833 |              0.5 |
| insuficiente  |           4 |              1     |              0.5 |
| respondivel   |          16 |              1     |              1   |
| sem_evidencia |           5 |              1     |              0.8 |

- Acerto geral de status: tolerante 96.8%; estrito 80.6%
- Respostas com status ok: 16; destas, com ≥ 1 evidência citada: 100%
- Citações inválidas removidas: 0
- Linhas sem citação removidas: 0
- Recusas do próprio LLM: 1
- Tempo mediano por pergunta: 2.5 s

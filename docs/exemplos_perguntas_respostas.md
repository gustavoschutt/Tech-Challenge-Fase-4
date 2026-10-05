# Exemplos de perguntas e respostas

> Gerado automaticamente por `scripts/06_gerar_exemplos.py` em 2026-10-04 22:31.
> Embeddings: `intfloat/multilingual-e5-base` · reranker: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` · LLM: `gemini:gemini-3.8-flash` · limiares calibrados: `True`.
> As respostas abaixo são a saída literal do sistema; nenhuma foi editada.

## 1. Investigação de reclamações

**Pergunta:** Quais são os principais problemas relatados pelos clientes em suas avaliações?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nenhum

Resumo: Os principais problemas relatados são "Pedido não recebido" (3406 avaliações, 23,6%), seguido por "Não recomenda / insatisfação geral" (1710 avaliações, 11,8%) e "Troca, devolução, cancelamento ou reembolso" (1704 avaliações, 11,8%).
- Pedido não recebido (3406 avaliações, 23,6%): Clientes relatam que os produtos não foram entregues e que continuam aguardando a chegada dos itens [E1][E2].
- Não recomenda / insatisfação geral (1710 avaliações, 11,8%): Clientes expressam forte insatisfação e afirmam não recomendar a loja devido ao descumprimento de prazos, falta de profissionalismo e produtos não entregues [E3][E4].
- Troca, devolução, cancelamento ou reembolso (1704 avaliações, 11,8%): Clientes solicitam reembolso ou troca devido a falhas na realização da entrega ou recebimento de produto com defeito [E5][E6].
- Pedido incompleto / faltando itens (1484 avaliações, 10,3%): Clientes apontam o recebimento parcial de seus pedidos, com itens faltando na entrega [E7][E8].
- Atraso na entrega (1215 avaliações, 8,4%): Clientes relatam demora no recebimento dos produtos e descumprimento das datas limites informadas para a entrega [E9][E10].

> Contagens calculadas sobre a base (98410 avaliações no recorte (42138 com comentário); filtros: nenhum).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 68,2% das 14445 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `20787e5ce2d5175a20135e52e3a701e8` | 2 | 2018-05 | sem_itens | nao_entregue | 0.713 | sim | Problemas com entrega. Não chegou ainda estou na esperando ainda |
| E2 | `4c2ab64ffc9119884d87d90bf9d630c3` | 1 | 2017-03 | sem_itens | nao_entregue | 0.561 | sim | Descaso com o cliente. O produto não foi entregue. |
| E3 | `b40a6984937cc251558d24d0b5caa929` | 1 | 2018-07 | beleza_saude | atrasada | 0.462 | sim | não recomendo. Não cumprem os ´prazos e ignoram a expectativa do cliente. Falta profissionalismo. Impossível recomendar..... |
| E4 | `7c8154d29d239b02ad0e8f2dcdf3261a` | 1 | 2017-12 | relogios_presentes | atrasada | 0.121 | sim | Péssimo o procedimento. Vergonha. Nada entregue até agora. Absurdo e descaso com cliente. |
| E5 | `2e6ec612df2ad503c839b73a704566cf` | 1 | 2018-03 | papelaria | atrasada | 0.205 | sim | Ouve um problema e a entrega não foi realizada quero o reembolso |
| E6 | `1750446f4afa39b6cb4905d0ff292c3d` | 3 | 2018-03 | telefonia | no_prazo | 0.154 | sim | Produto chegou no prazo, mas com defeito. Estou pedindo troca ou reembolso. |
| E7 | `84775964cfaf2e4fa80f8657d4cdede8` | 1 | 2018-06 | cine_foto | no_prazo | 0.826 | sim | Problemas. Entregaram apenas um produto dos quatro itens do pedido... Quero o restante do pedido... Cadê? |
| E8 | `7d81592881b86d6ba09e4e5ae954a59d` | 1 | 2017-12 | ferramentas_jardim | no_prazo | 0.639 | sim | Chegou o produto atrasado, faltando outro. Falta chegar os demais. O cliente está zangado. |
| E9 | `1da9941264ebc5c4a093ee4ba82af918` | 1 | 2017-11 | brinquedos | no_prazo | 0.108 | sim | O produto chegou correto. O problema foi a demora na entrega. |
| E10 | `4baf65ba0f4dc753e9cf3d8863433da0` | 2 | 2018-01 | papelaria | atrasada | 0.064 | sim | Falta de consideração com o cliente, pois ocorreu problema na entrega é em momento nenhum o problema foi tratado.duas compras, uma com prazo de entrega 15 e 21/02, as 2 atrasou, liguei dia 23 reclaman |

<sub>tempo total: 34981.7 ms · motivo do gate: -</sub>

## 2. Investigação de um tema específico

**Pergunta:** O que os clientes relatam sobre problemas relacionados à entrega?

**Status:** `ok` · **modo:** `rag`
**Filtros:** nenhum

Resumo: Clientes apontam a falta de recebimento dos produtos, pedidos entregues de forma incompleta e a ausência de posicionamento após serem comunicados sobre ocorrências na entrega [E1][E2][E3][E5].
- Não recebimento do produto: Relato de mercadorias que não foram entregues aos compradores [E2][E3].
- Entrega parcial e divergência de status: Pedido marcado como entregue, mas tendo apenas um dos produtos recebido [E1].
- Falta de comunicação sobre ocorrências: Notificações informando que houve problema na entrega sem contato posterior para esclarecer a situação ou dar um posicionamento [E3][E5].
- Insatisfação com o atendimento no processo de entrega: Relatos de descaso, falta de atenção e falta de respeito com o cliente em relação à entrega [E2][E6].

> Base da resposta: 7 avaliações recuperadas por similaridade semântica e léxica. É uma amostra das avaliações mais relevantes, não uma contagem da base.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `42a3a3e2b79465364e24f6f4c642bc23` | 1 | 2018-06 | cool_stuff | no_prazo | 0.638 | sim | Em partes. Sou cliente há anos, porém sempre há divergências e problemas na entrega, consta como produto entregue, porém foi entregue apenas 1 produto, aguardo solução! |
| E2 | `4c2ab64ffc9119884d87d90bf9d630c3` | 1 | 2017-03 | sem_itens | nao_entregue | 0.146 | sim | Descaso com o cliente. O produto não foi entregue. |
| E3 | `2bf5cb7bc41acf9177350b5eb46a563b` | 1 | 2017-05 | cool_stuff | atrasada | 0.136 | sim | Me informaram que ocorreu um problema na entrega, e até agora não entraram em contato comigo para esclarecer o que houve. Até agora não recebi o produto. |
| E4 | `77ab0789563a5269c2b37be6c00a5148` | 4 | 2017-09 | telefonia | no_prazo | 0.120 | não | Produto entregue se problemas obrigado |
| E5 | `b2253bbad8b5eac2bdc94b8bfa401938` | 1 | 2017-03 | audio | atrasada | 0.102 | sim | Até esta data não recebi um posicionamento sobre a entrega do produto comprado. Apenas enviaram um email dizendo que houve problema na entrega e até agora nada |
| E6 | `9b1e2c9d0d236f106bed5169a8cde33f` | 1 | 2018-06 | esporte_lazer | atrasada | 0.087 | sim | Entrega do produto. Falta de atenção e respeito com cliente |
| E7 | `90fea2d1dc3129c9020543b5bba9394d` | 4 | 2017-10 | esporte_lazer | no_prazo | 0.075 | não | Entrega realizada.....cliente satisfeito |

<sub>tempo total: 2817.0 ms · motivo do gate: 7 evidências relevantes</sub>

## 3. Busca por experiências positivas

**Pergunta:** Quais aspectos da experiência de compra são mais elogiados pelos clientes?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nenhum

Resumo: Os aspectos mais elogiados pelos clientes são a recomendação e satisfação geral, com 17155 avaliações (61,9%), seguidos pela entrega rápida ou no prazo, que soma 9033 avaliações (32,6%).
- Recomenda / satisfação geral (17155 avaliações, 61,9%): Os clientes expressam satisfação com as compras, recomendam a loja e elogiam a honestidade com o consumidor [E1][E2].
- Entrega rápida / no prazo (9033 avaliações, 32,6%): Os relatos destacam entregas realizadas de forma rápida, dentro do prazo e até mesmo bem antes da data prevista [E3][E4].
- Boa qualidade do produto (3917 avaliações, 14,1%): Os consumidores ressaltam a qualidade excelente ou boa dos itens recebidos e o atendimento às expectativas [E5][E6].
- Conforme o anunciado (2333 avaliações, 8,4%): Os comentários apontam que os produtos chegaram de acordo com as informações anunciadas, com medidas corretas e correspondendo ao esperado [E7][E8].
- Embalagem bem feita (759 avaliações, 2,7%): Os clientes elogiam produtos que chegam sempre bem embalados e com embalagens perfeitas [E9][E10].

> Contagens calculadas sobre a base (98410 avaliações no recorte (42138 com comentário); filtros: nenhum).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (elogio): 79,0% das 27693 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `382d855ffdd4f5e66594dbfb1566e96f` | 5 | 2018-05 | moveis_decoracao | no_prazo | 0.684 | sim | 10. Cada vez que compro mais fico satisfeita parabéns pela honestidade com seus clientes 👏👏👏👏? |
| E2 | `57df4621973c6723b2aa4280ab6e19ac` | 5 | 2018-05 | esporte_lazer | no_prazo | 0.391 | sim | recomendo. cliente satisfeito . |
| E3 | `07a4c0fd1d3bc16284492825bdd62bff` | 5 | 2018-07 | moveis_escritorio | no_prazo | 0.421 | sim | Elogio. Entrega bem antes do prazo. Produto de excelente qualidade. Recomendo o produto e a loja. |
| E4 | `58650b37b6c7f5982d4fef12fb56c0d5` | 5 | 2018-04 | beleza_saude | no_prazo | 0.281 | sim | Item de compra. Entrega rápida e dentro do prazo. |
| E5 | `76df0bd8c9d614662a1e893152b4741f` | 5 | 2018-04 | informatica_acessorios | no_prazo | 0.300 | sim | Produto de qualidade! Compromisso com o cliente e qualidade no produto resumem essa compra. |
| E6 | `8ba3773999e3da3fda42bcf36d8343b9` | 4 | 2018-05 | eletronicos | no_prazo | 0.201 | sim | Produto de boa qualidade. Produto atende as expectativas. |
| E7 | `10b46c399d03fa3b754b8d4a6108ddf5` | 5 | 2018-08 | moveis_decoracao | no_prazo | 0.379 | sim | Tudo Certo!Satisfeita. Gostei da experiência o produto conforme o anunciado, bonito as medições corretas. Muito satisfeita. |
| E8 | `7cde8d21243a997106a2ae5149b39627` | 5 | 2017-10 | ferramentas_jardim | atrasada | 0.153 | sim | Produto recebido com conforme anunciado. Cliente satisfeito. |
| E9 | `ba47b17fc0f29051db41fbdf12034e21` | 5 | 2018-07 | papelaria | no_prazo | 0.767 | sim | Recomendo. Confiança e Segurança nas compras, produtos entregues antes do prazo, estão sempre bem embalados. |
| E10 | `cd92b9871d799d527450f33eee615e3e` | 5 | 2017-03 | beleza_saude | no_prazo | 0.303 | sim | Entrega no prazo, embalagem perfeita. Só elogios. |

<sub>tempo total: 4629.6 ms · motivo do gate: -</sub>

## 4. Investigação de insatisfação

**Pergunta:** Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nota 1–2

Resumo: Os temas mais frequentes entre os clientes insatisfeitos são Pedido não recebido (3095 avaliações, 28,5%) e Não recomenda / insatisfação geral (1543 avaliações, 14,2%).
- Pedido não recebido (3095 avaliações, 28,5%): Clientes relatam que ainda não receberam o produto ou que a entrega não ocorreu dentro do prazo estipulado pelo vendedor [E1][E2].
- Não recomenda / insatisfação geral (1543 avaliações, 14,2%): Clientes apontam produtos de péssima qualidade, descumprimento de prazos e falta de profissionalismo para justificar que não recomendam [E3][E4].
- Troca, devolução, cancelamento ou reembolso (1501 avaliações, 13,8%): Relatos envolvem recebimento de produtos com defeito gerando pedidos de devolução do dinheiro, além de compras canceladas sem explicações após autorização do débito [E5][E6].
- Pedido incompleto / faltando itens (1211 avaliações, 11,2%): Clientes descrevem pedidos que chegaram com itens faltantes, por vezes combinados com atraso ou com itens recebidos quebrados [E7][E8].
- Atraso na entrega (928 avaliações, 8,6%): Avaliações relatam aumento do prazo de entrega após a confirmação da compra, recebimento de produto diferente e atrasos sem satisfação ao cliente [E9][E10].

> Contagens calculadas sobre a base (14396 avaliações no recorte (10845 com comentário); filtros: nota 1–2).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 75,7% das 10845 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `c888e9da2845d8f0ec5d682e050205aa` | 1 | 2017-12 | ferramentas_jardim | atrasada | 0.006 | sim | o produto nao poder ser avaliado ainda nao recebi |
| E2 | `431b0cb6f9d2b639e13d0baa4d90633a` | 2 | 2018-03 | ferramentas_jardim | atrasada | 0.005 | sim | Produto não entregue. O produto não foi entregue no prazo estipulado pelo vendedor. |
| E3 | `6698b8cb16ed759ce466fefa70c941b7` | 1 | 2018-07 | automotivo | no_prazo | 0.198 | sim | Avaliação. Loja recomendada. Produto de péssima qualidade. |
| E4 | `b40a6984937cc251558d24d0b5caa929` | 1 | 2018-07 | beleza_saude | atrasada | 0.103 | sim | não recomendo. Não cumprem os ´prazos e ignoram a expectativa do cliente. Falta profissionalismo. Impossível recomendar..... |
| E5 | `7b1f6c0a481e190e038965bbbbe3af9b` | 1 | 2017-10 | moveis_decoracao | no_prazo | 0.009 | sim | Produto com defeito. Furado. Desejo devolução do dinheiro. |
| E6 | `0d592da0b11f266efd97a16956ff6417` | 1 | 2017-02 | sem_itens | nao_entregue | 0.008 | sim | compra efetuada,debito autorizado ,pedido cancelado sem explicações.Extrema falta de respeito e consideração com os clientes. |
| E7 | `7d81592881b86d6ba09e4e5ae954a59d` | 1 | 2017-12 | ferramentas_jardim | no_prazo | 0.026 | sim | Chegou o produto atrasado, faltando outro. Falta chegar os demais. O cliente está zangado. |
| E8 | `c40a6b6e0181e5ec0d12cbc2e12c49d3` | 1 | 2018-05 | utilidades_domesticas | no_prazo | 0.017 | sim | falta de produto e quebra. Faltou 1 produto e os que recebi 1 veio quebrado |
| E9 | `df6c29df5ddd4dee73f86f0251306c87` | 2 | 2017-12 | eletronicos | no_prazo | 0.013 | sim | Demora na entrega.. O prazo da entrega foi aumentado após a confirmação da compra... Insatisfeito, e o produto veio diferente também. .. |
| E10 | `2d0a07007cc84621a2bc694505be4036` | 1 | 2017-11 | moveis_decoracao | atrasada | 0.003 | sim | Atraso na entrega sem dar satisfação ao cliente, que com certeza escolheu o produto com cuidado e com as melhores expectativas. Falta de consideração. |

<sub>tempo total: 5073.9 ms · motivo do gate: -</sub>

## 5. Exploração baseada em evidências

**Pergunta:** Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nenhum

Resumo: Os problemas relacionados ao prazo de entrega concentram-se principalmente em "Pedido não recebido" (3406 avaliações, 23,6%) e "Atraso na entrega" (1215 avaliações, 8,4%), além de impactos citados em pedidos incompletos e problemas de logística.
- Pedido não recebido (3406 avaliações, 23,6%): Os clientes relatam que o produto não foi entregue dentro do prazo estipulado pelo vendedor ou reclamam que o prazo já era extenso e o item não foi entregue [E1][E2].
- Atraso na entrega (1215 avaliações, 8,4%): Os consumidores afirmam estar aguardando a entrega que está em atraso ou reclamam da muita demora e de prazos de entrega longos demais [E5][E6].
- Pedido incompleto / faltando itens (1484 avaliações, 10,3%): Os clientes relatam recebimento de produtos com atraso associado à falta de outros itens da compra [E4].
- Transportadora / Correios (739 avaliações, 5,1%): Os relatos apontam que o prazo de entrega expirou e o produto não foi recebido, associando a espera a problemas com os Correios [E7][E8].

> Contagens calculadas sobre a base (98410 avaliações no recorte (42138 com comentário); filtros: nenhum).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 41,7% das 14445 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `431b0cb6f9d2b639e13d0baa4d90633a` | 2 | 2018-03 | ferramentas_jardim | atrasada | 0.004 | sim | Produto não entregue. O produto não foi entregue no prazo estipulado pelo vendedor. |
| E2 | `3259959e9e52ef23e30b75f259285c5f` | 1 | 2018-04 | eletronicos | no_prazo | 0.004 | sim | Horrivel. Não sei como nos dias atuais ainda existem lojas assim. O prazo de entrega já é muito extenso. E o maior problema de todos: o item não foi entregue! |
| E3 | `facaaf5d1ea43d6ead5d1288fb8eed21` | 1 | 2018-04 | cama_mesa_banho | no_prazo | 0.042 | não | Veio faltando produto Comprei 2 cortinas e só veio 1 Quando resolver o problema mudo minha avaliação. |
| E4 | `7d81592881b86d6ba09e4e5ae954a59d` | 1 | 2017-12 | ferramentas_jardim | no_prazo | 0.027 | sim | Chegou o produto atrasado, faltando outro. Falta chegar os demais. O cliente está zangado. |
| E5 | `364e74a2d89a35fa3fad2beaf96d8f17` | 1 | 2018-07 | moveis_escritorio | atrasada | 0.009 | sim | Aguardando a entrega. Entrega em atraso. |
| E6 | `ba5c756c2d4d0d636ac63de5cedb771d` | 3 | 2018-05 | relogios_presentes | no_prazo | 0.008 | sim | Muita demora. O prazo de entrega foi longo demais |
| E7 | `1fe41a778cdead7d5c5b68b26d1ab504` | 3 | 2017-11 | eletronicos | atrasada | 0.011 | sim | Parece ser problemas do correio. Continuo aguardando o produto. |
| E8 | `05481f963c19afbeeb6c2d48fb31f3fe` | 1 | 2017-06 | market_place | atrasada | 0.009 | sim | nao recebi o produto acredito ser problema dos correios prazo de entrega expirou |

<sub>tempo total: 8531.4 ms · motivo do gate: -</sub>

## 6. Problema de negócio: frequência

**Pergunta:** Quais reclamações relacionadas à entrega aparecem com maior frequência?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nenhum

Resumo: As reclamações sobre entrega com maior frequência são "Pedido não recebido" (3406 avaliações, 23,6%) e "Pedido incompleto / faltando itens" (1484 avaliações, 10,3%), seguidas por atrasos e problemas com a transportadora ou Correios.
- Pedido não recebido (3406 avaliações, 23,6%): Clientes relatam descaso e falta de compromisso, apontando que o produto não foi entregue ou nunca chegou, mesmo após reclamações [E1][E2].
- Pedido incompleto / faltando itens (1484 avaliações, 10,3%): Consumidores apontam que a entrega chegou faltando produtos comprados, sem retorno para envio dos itens restantes ou reembolso [E3][E4].
- Atraso na entrega (1215 avaliações, 8,4%): Relatos indicam insatisfação com a demora e o fato de estarem aguardando pedidos que estão em atraso [E5][E6].
- Transportadora / Correios (739 avaliações, 5,1%): Clientes reclamam da qualidade dos Correios, dificuldades na entrega e divergências no rastreamento, como falsas tentativas de entrega registradas [E7][E8].

> Contagens calculadas sobre a base (98410 avaliações no recorte (42138 com comentário); filtros: nenhum).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 41,7% das 14445 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `4c2ab64ffc9119884d87d90bf9d630c3` | 1 | 2017-03 | sem_itens | nao_entregue | 0.080 | sim | Descaso com o cliente. O produto não foi entregue. |
| E2 | `4966a0af44d9ad8f87d9263c8bacc9dc` | 1 | 2017-10 | beleza_saude | atrasada | 0.039 | sim | Falta de compromisso com o cliente. O meu produto nunca chegou apesar das minhas reclamações. |
| E3 | `5148ad2bdca65375e9504355d79c6ad4` | 1 | 2018-07 | beleza_saude | no_prazo | 0.323 | sim | Entrega incompleta. Realizei um pedido com dois itens e apenas 1 deles foi entregue. Fiz algumas reclamações e até agora ninguém entrou em contato comigo pra entregar o outro produto ou para devolver o dinheiro. |
| E4 | `7d6ba969ba93e86ed87d563c1821616b` | 1 | 2018-08 | eletronicos | no_prazo | 0.081 | sim | Entrega faltando produtos. Fiz pedido de 6 produtos e só recebi 2. |
| E5 | `7528a396f7625967f05cb09946286608` | 1 | 2018-03 | construcao_ferramentas_construcao | atrasada | 0.040 | sim | Ultimamente estão atrasando muito as entregas |
| E6 | `364e74a2d89a35fa3fad2beaf96d8f17` | 1 | 2018-07 | moveis_escritorio | atrasada | 0.034 | sim | Aguardando a entrega. Entrega em atraso. |
| E7 | `3639b950e01dcb13e78d423d698e8b24` | 2 | 2018-06 | cama_mesa_banho | no_prazo | 0.070 | sim | Entrega complicada. Recomendo a loja, porém, a entrega pelo Correios está cada dia mais difícil. Ao rastrear o objeto aparece que o Correios saiu para entrega, sendo que na realidade não houve tentativa de entrega |
| E8 | `6d58ce34b52614647bda4b987b6193c9` | 3 | 2018-07 | moveis_decoracao | no_prazo | 0.028 | sim | entrega. Favor procurar utra transportadora...Correios sem qualidade. |

<sub>tempo total: 5993.3 ms · motivo do gate: -</sub>

## 7. Problema de negócio: qualidade

**Pergunta:** Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nenhum

Resumo: Entre os elogios, o tema mais frequente é "Boa qualidade do produto" (3917 avaliações, 14,1%), enquanto nos problemas destacam-se "Produto errado / diferente do anunciado" (759 avaliações, 5,3%) e "Produto danificado / avariado" (603 avaliações, 4,2%).
- Boa qualidade do produto (3917 avaliações, 14,1%): Os clientes relatam que os itens recebidos possuem boa qualidade ou são excelentes produtos [E1][E2].
- Conforme o anunciado (2333 avaliações, 8,4%): Avaliações destacam itens com ótima qualidade e que vieram exatamente como anunciado [E3][E4].
- Preço / custo-benefício (495 avaliações, 1,8%): Clientes apontam produtos de excelente qualidade e correspondentes ao anúncio que apresentam ótimo custo-benefício [E5][E6].
- Produto errado / diferente do anunciado (759 avaliações, 5,3%): Relatos apontam entrega de itens errados, divergência em cor e tamanho, além de péssima qualidade e falta de conformidade com o pedido [E7][E8].
- Produto danificado / avariado (603 avaliações, 4,2%): Os clientes mencionam o recebimento de itens quebrados ou com avarias em seus componentes [E9][E10].
- Defeito / não funciona (573 avaliações, 4,0%): Avaliações indicam produtos que não funcionam, sem qualidade, ou com falhas de funcionamento como barulhos altos e não leitura de mídia/reconhecimento de drive [E11][E12].
- Qualidade abaixo do esperado (460 avaliações, 3,2%): Os relatos citam qualidade ruim e itens que não são originais [E13][E14].
- Produto falsificado / não original (213 avaliações, 1,5%): Clientes reclamam do envio de itens falsificados, piratas e não originais da marca esperada [E15][E16].

> Contagens calculadas sobre a base (98410 avaliações no recorte (42138 com comentário); filtros: nenhum).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 16,7% das 14445 avaliações da base de contagem. Cobertura da taxonomia (elogio): 23,1% das 27693 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `ebe3982588369e646e6232c16324a85e` | 5 | 2018-01 | cama_mesa_banho | no_prazo | 0.323 | sim | Entrega feita normalmente. Produto de boa qualidade. |
| E2 | `45a301e140af06e6d42ab5be8210795f` | 5 | 2018-08 | utilidades_domesticas | no_prazo | 0.227 | sim | Avaliação de lojista. Excelente produto. |
| E3 | `4bed232702aac2934777e372687df14b` | 5 | 2017-08 | informatica_acessorios | no_prazo | 0.166 | sim | Entrega no prazo, produto de qualidade exatamente como anunciado. Recomendo a loja. |
| E4 | `b1eeb16a661d22a761743bf373e811d4` | 5 | 2018-07 | cama_mesa_banho | no_prazo | 0.098 | sim | Produto muito bom. Produto de ótima qualidade, conforme anunciado. |
| E5 | `41bd1cc0d4979d40814a23908c72b37c` | 5 | 2018-08 | bebes | no_prazo | 0.027 | sim | Produto de excelente qualidade. Vale a pena o custo x benefício. Entrega antes do prazo estipulado. Recomendo |
| E6 | `e986d799c9e32a490dff4adc50f9fb5a` | 5 | 2018-08 | brinquedos | no_prazo | 0.012 | sim | Perfeito. Produto corresponde ao anunciado. Ótimo custo beneficio. |
| E7 | `162be7af72150512d0f5033c45115e06` | 1 | 2018-06 | relogios_presentes | no_prazo | 0.056 | sim | Produto errado. Mais responsabilidade com os clientes quanto origem do produto,qualidade,certeza de que o produto entregue é de acordo com o pedido |
| E8 | `2f2a5e457479ae6caa49462daad14e5d` | 1 | 2017-11 | cama_mesa_banho | no_prazo | 0.047 | sim | Produto errado,cor e tamanho errado, péssima qualidade. |
| E9 | `f5e58f641f22f422a0f0abf47321e722` | 1 | 2018-06 | utilidades_domesticas | no_prazo | 0.037 | sim | Produto Avariado! Produto veio avariado/quebrado, nao me deram bola, pois nao reclamei em 7 dias... |
| E10 | `ade7dccc8ee9d9921b3db67b3ae7d29a` | 2 | 2018-07 | malas_acessorios | no_prazo | 0.034 | sim | PRODUTO AVARIADO. O produto chegou rápido, porém apresenta avarias em seus componentes. |
| E11 | `c76ecf50da6714104597ba87c9d4fd3d` | 1 | 2018-05 | market_place | no_prazo | 0.533 | sim | zero. Produto com defeito, não funciona, sem qualidade, já é a terceira vez que reclamo e nada foi feito. |
| E12 | `16542dee3cb122659e9ef33431f5f4a6` | 1 | 2018-06 | informatica_acessorios | no_prazo | 0.125 | sim | Produto com defeito! O produto veio com defeito, funciona só de vez em quando, pois na maioria das vezes, ele não le o cd, ou o drive nao é reconhecido, e quando isso acontece, faz um barulho diferente e alto |
| E13 | `7f45121d39a6318dff3f2fef124e542a` | 1 | 2017-08 | utilidades_domesticas | no_prazo | 0.078 | sim | Qualidade ruim, produto não é original. |
| E14 | `14c3dacba0817e8565029701ff4ed917` | 3 | 2018-07 | pet_shop | no_prazo | 0.069 | sim | qualidade ruim do produto |
| E15 | `c7ecd3d49473c11b9ba438f8937235b5` | 1 | 2017-12 | cool_stuff | no_prazo | 0.014 | sim | SRS, O PRODUTO ENVIADO É FALSO, NÃO É ORIGINAL LG. TRATA-SE DE PRODUTO PIRATA / CHING LING. SOLICITO TROCA POR ORIGINAL. |
| E16 | `c4f9fc857bd9e8a92ffd9a7f7e2070d2` | 1 | 2017-12 | sem_categoria | no_prazo | 0.014 | sim | O produto é falsificado. |

<sub>tempo total: 10949.0 ms · motivo do gate: -</sub>

## 8. Metadados: nota baixa

**Pergunta:** Quais são os principais problemas relatados nas avaliações com nota baixa?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nota 1–2

Resumo: Os principais problemas relatados nas avaliações com nota baixa são "Pedido não recebido", com 3095 avaliações (28,5%), seguido por "Não recomenda / insatisfação geral", com 1543 avaliações (14,2%).
- Pedido não recebido (3095 avaliações, 28,5%): Clientes relatam que a compra não foi entregue, continuam esperando ou apontam descaso com a falta de entrega [E1][E2].
- Não recomenda / insatisfação geral (1543 avaliações, 14,2%): Os relatos indicam descumprimento de prazos, falta de profissionalismo e indignação por não terem recebido nada até o momento [E3][E4].
- Troca, devolução, cancelamento ou reembolso (1501 avaliações, 13,8%): Consumidores apontam entrega de itens diferentes dos solicitados com demora no atendimento da troca/devolução, ou exigem reembolso devido à entrega não realizada [E5][E6].
- Pedido incompleto / faltando itens (1211 avaliações, 11,2%): Clientes afirmam ter recebido apenas parte dos produtos comprados, cobrando o envio dos itens faltantes [E7][E8].
- Atraso na entrega (928 avaliações, 8,6%): Os relatos mencionam demora no prazo de entrega, inclusive associada a atraso prévio na emissão da nota fiscal [E9][E10].

> Contagens calculadas sobre a base (14396 avaliações no recorte (10845 com comentário); filtros: nota 1–2).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 75,7% das 10845 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `20787e5ce2d5175a20135e52e3a701e8` | 2 | 2018-05 | sem_itens | nao_entregue | 0.298 | sim | Problemas com entrega. Não chegou ainda estou na esperando ainda |
| E2 | `4c2ab64ffc9119884d87d90bf9d630c3` | 1 | 2017-03 | sem_itens | nao_entregue | 0.223 | sim | Descaso com o cliente. O produto não foi entregue. |
| E3 | `b40a6984937cc251558d24d0b5caa929` | 1 | 2018-07 | beleza_saude | atrasada | 0.295 | sim | não recomendo. Não cumprem os ´prazos e ignoram a expectativa do cliente. Falta profissionalismo. Impossível recomendar..... |
| E4 | `7c8154d29d239b02ad0e8f2dcdf3261a` | 1 | 2017-12 | relogios_presentes | atrasada | 0.075 | sim | Péssimo o procedimento. Vergonha. Nada entregue até agora. Absurdo e descaso com cliente. |
| E5 | `fbf6d0b4d03170bf5cf4c0fdc0705364` | 1 | 2018-04 | beleza_saude | no_prazo | 0.066 | sim | Comprei um produto e entregaram outro difetente do que pedi..... E mais solicitei a devolucao ou troca e ate agora nada.... Estam se fazendo de dezentendidos... E fazer pouco caso do cliente ne.. |
| E6 | `2e6ec612df2ad503c839b73a704566cf` | 1 | 2018-03 | papelaria | atrasada | 0.057 | sim | Ouve um problema e a entrega não foi realizada quero o reembolso |
| E7 | `84775964cfaf2e4fa80f8657d4cdede8` | 1 | 2018-06 | cine_foto | no_prazo | 0.580 | sim | Problemas. Entregaram apenas um produto dos quatro itens do pedido... Quero o restante do pedido... Cadê? |
| E8 | `7d81592881b86d6ba09e4e5ae954a59d` | 1 | 2017-12 | ferramentas_jardim | no_prazo | 0.395 | sim | Chegou o produto atrasado, faltando outro. Falta chegar os demais. O cliente está zangado. |
| E9 | `1da9941264ebc5c4a093ee4ba82af918` | 1 | 2017-11 | brinquedos | no_prazo | 0.110 | sim | O produto chegou correto. O problema foi a demora na entrega. |
| E10 | `2dda217cbce6c248f5488fe095547840` | 2 | 2018-03 | automotivo | atrasada | 0.097 | sim | Demora na emissão da nota levou mais de semana pra isso, demora na entrega ainda não recebi. |

<sub>tempo total: 5707.3 ms · motivo do gate: -</sub>

## 9. Metadados: categoria

**Pergunta:** O que os clientes mais elogiam em produtos de beleza e saúde?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** categoria ∈ {beleza_saude}

Resumo: Os aspectos mais elogiados pelos clientes são "Recomenda / satisfação geral", com 1527 avaliações (63,0%), e "Entrega rápida / no prazo", com 875 avaliações (36,1%).
- Recomenda / satisfação geral (1527 avaliações, 63,0%): Os clientes expressam alta recomendação, satisfação com os itens e agradecimento pelo atendimento e agilidade no envio de itens de saúde [E1][E2].
- Entrega rápida / no prazo (875 avaliações, 36,1%): Os relatos destacam o recebimento dos produtos de forma rápida, bem antes do prazo estipulado e em perfeitas condições [E3][E4].
- Boa qualidade do produto (256 avaliações, 10,6%): Os clientes ressaltam que os produtos adquiridos possuem qualidade excelente e muito boa [E5][E6].
- Conforme o anunciado (197 avaliações, 8,1%): As avaliações apontam que os produtos são exatamente como descritos e anunciados [E7][E8].
- Embalagem bem feita (60 avaliações, 2,5%): Os consumidores mencionam que os produtos chegam bem embalados [E9][E10].

> Contagens calculadas sobre a base (8752 avaliações no recorte (3532 com comentário); filtros: categoria ∈ {beleza_saude}).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (elogio): 80,9% das 2424 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `a723c77725076ac873a4cc9c65d209fb` | 5 | 2018-06 | beleza_saude | no_prazo | 0.225 | sim | recomendadissima. Por se tratar de produto referente a saúde e saúde não espera ,agradeço aos profissionais envolvidos. |
| E2 | `d0e8ebec0d960128c72f7d26b459be6f` | 5 | 2018-06 | construcao_ferramentas_construcao | no_prazo | 0.047 | sim | Muito bom os produtos. Estou muito satisfeito com os produtos. Parabéns aos lojistas. |
| E3 | `72888bd255eff79a818d533c426b67f1` | 5 | 2018-05 | beleza_saude | no_prazo | 0.142 | sim | Entrega rapida. Produto chegou em perfeitas condições, bem antes do prazo. Recomendo |
| E4 | `737d8690973d244edd8ce69929debc1e` | 5 | 2018-08 | beleza_saude | no_prazo | 0.116 | sim | Ótimo. O produto chegou rápido antes do prazo |
| E5 | `635b498c97e573f64911343026d14018` | 5 | 2018-08 | beleza_saude | no_prazo | 0.144 | sim | Otimo. Produto de excelente qualidade. |
| E6 | `35bb42b0e58872656bcb1d8a74818149` | 5 | 2018-05 | beleza_saude | no_prazo | 0.021 | sim | Excelente produto. Produto de qualidade muito boa. |
| E7 | `9c0839cd1808c07e2b6b4e552937cbd9` | 5 | 2018-08 | beleza_saude | no_prazo | 0.166 | sim | Tudo certo! Produto conforme descrito, muito bom e entrega rápida. |
| E8 | `8cfd3e245ebfe53461f8591f49395216` | 5 | 2018-07 | beleza_saude | atrasada | 0.066 | sim | Produto exatamente como o descrito. Entrega um pouco antes do prazo informado no site. Recomendo! |
| E9 | `0e6860cd83660872b6f27f9ff75d2dbc` | 5 | 2018-07 | beleza_saude | no_prazo | 0.047 | sim | Entrega rápida. Produto chegou rápido e bem embalado. |
| E10 | `db1893bcebe183686ed630b33a96ef0c` | 5 | 2018-08 | beleza_saude | no_prazo | 0.045 | sim | Entrega rápida. Produto bem embalado. Entrega muito rápida. Muito bom! |

<sub>tempo total: 5265.9 ms · motivo do gate: -</sub>

## 10. Metadados: entrega

**Pergunta:** Quais temas aparecem nas avaliações relacionadas a experiências de entrega insatisfatórias?

**Status:** `ok` · **modo:** `analitico`
**Filtros:** nota 1–2

Resumo: Os temas mais frequentes sobre entrega insatisfatória são "Pedido não recebido" com 3095 avaliações (28,5%) e "Pedido incompleto / faltando itens" com 1211 avaliações (11,2%).
- Pedido não recebido (3095 avaliações, 28,5%): Clientes relatam descaso e queixas sobre produtos que ainda não chegaram ou não foram entregues [E1][E2].
- Pedido incompleto / faltando itens (1211 avaliações, 11,2%): Clientes relatam receber apenas parte dos produtos comprados, ficando no aguardo dos demais [E3][E4].
- Atraso na entrega (928 avaliações, 8,6%): Clientes apontam insatisfação e recomendação negativa devido à espera e entrega em atraso [E5][E6].
- Transportadora / Correios (533 avaliações, 4,9%): Clientes atribuem os problemas de entrega ou o não recebimento dentro do prazo aos Correios [E7][E8].

> Contagens calculadas sobre a base (14396 avaliações no recorte (10845 com comentário); filtros: nota 1–2).
> Método: taxonomia de temas por regras léxicas auditáveis; uma avaliação pode ter mais de um tema. Cobertura da taxonomia (problema): 46,7% das 10845 avaliações da base de contagem.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `4c2ab64ffc9119884d87d90bf9d630c3` | 1 | 2017-03 | sem_itens | nao_entregue | 0.040 | sim | Descaso com o cliente. O produto não foi entregue. |
| E2 | `20787e5ce2d5175a20135e52e3a701e8` | 2 | 2018-05 | sem_itens | nao_entregue | 0.018 | sim | Problemas com entrega. Não chegou ainda estou na esperando ainda |
| E3 | `7d81592881b86d6ba09e4e5ae954a59d` | 1 | 2017-12 | ferramentas_jardim | no_prazo | 0.173 | sim | Chegou o produto atrasado, faltando outro. Falta chegar os demais. O cliente está zangado. |
| E4 | `7d6ba969ba93e86ed87d563c1821616b` | 1 | 2018-08 | eletronicos | no_prazo | 0.057 | sim | Entrega faltando produtos. Fiz pedido de 6 produtos e só recebi 2. |
| E5 | `364e74a2d89a35fa3fad2beaf96d8f17` | 1 | 2018-07 | moveis_escritorio | atrasada | 0.069 | sim | Aguardando a entrega. Entrega em atraso. |
| E6 | `108195606e37780c4335f0725c4c7e3e` | 2 | 2018-08 | automotivo | atrasada | 0.033 | sim | Não recomendado. Atraso na entrega. |
| E7 | `e8044da342e4b422f1ee1274341407d0` | 1 | 2017-11 | esporte_lazer | atrasada | 0.062 | sim | Meu problema foi com os correios. |
| E8 | `05481f963c19afbeeb6c2d48fb31f3fe` | 1 | 2017-06 | market_place | atrasada | 0.039 | sim | nao recebi o produto acredito ser problema dos correios prazo de entrega expirou |

<sub>tempo total: 4642.7 ms · motivo do gate: -</sub>

## 11. Metadados: entrega + UF

**Pergunta:** O que dizem os clientes de SP sobre pedidos entregues com atraso?

**Status:** `ok` · **modo:** `rag`
**Filtros:** UF ∈ {SP}; entrega ∈ {atrasada}

Resumo: Clientes de SP relatam atraso na entrega de seus pedidos, manifestam insatisfação por continuarem aguardando o recebimento e cobram esclarecimentos sobre os motivos da demora [E1][E2][E3][E4][E5][E6][E7][E8].
- Espera e falta de entrega: Clientes afirmam que ainda estão aguardando os produtos e que a entrega está atrasada [E1][E4][E6][E7][E8].
- Falta de informação e cobrança por esclarecimentos: Clientes reclamam que não foram avisados sobre o atraso, não sabem o motivo do ocorrido e pedem esclarecimentos da empresa [E3][E4][E5].
- Não recomendação e intenção de cancelamento: Clientes declaram que não recomendam o serviço pelo descumprimento do prazo e ameaçam cancelar o pedido se a entrega não ocorrer logo [E2][E4][E5].

> Base da resposta: 8 avaliações recuperadas por similaridade semântica e léxica. É uma amostra das avaliações mais relevantes, não uma contagem da base.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `364e74a2d89a35fa3fad2beaf96d8f17` | 1 | 2018-07 | moveis_escritorio | atrasada | 0.237 | sim | Aguardando a entrega. Entrega em atraso. |
| E2 | `108195606e37780c4335f0725c4c7e3e` | 2 | 2018-08 | automotivo | atrasada | 0.133 | sim | Não recomendado. Atraso na entrega. |
| E3 | `f6113199cd352db89c5a7e59b903d549` | 1 | 2018-03 | bebes | atrasada | 0.123 | sim | Entrega super em atraso, preciso de um esclarecimento do ocorrido |
| E4 | `30fe371caf618181a6a5180d076e83fc` | 1 | 2018-07 | telefonia | atrasada | 0.090 | sim | Não recomendo. Atraso na entrega, ainda não recebi e não sou informado o por que do atraso.. Se não receber logo vou cancelar o pedido |
| E5 | `85a391b84d8e864de7190a2afb6f82e4` | 1 | 2018-08 | alimentos | atrasada | 0.085 | sim | não cumpre prazo. Não cumpriram prazo de entrega e nem avisaram o atraso. |
| E6 | `44dba35181b574d4779194f1c8e43e20` | 1 | 2018-02 | automotivo | atrasada | 0.085 | sim | Atraso na entrega. |
| E7 | `9f5600c5ab1457f1ec08b2ea6720cf49` | 1 | 2017-10 | moveis_escritorio | atrasada | 0.073 | sim | entrega muito atrasada. |
| E8 | `5193565d8ef972341e963399fedec040` | 1 | 2018-01 | cama_mesa_banho | atrasada | 0.073 | sim | Entrega atrasada |

<sub>tempo total: 2937.6 ms · motivo do gate: pergunta definida pelos filtros; evidências = recorte filtrado</sub>

## 12. Tema específico

**Pergunta:** O que os clientes contam sobre mercadorias que chegaram quebradas ou danificadas?

**Status:** `ok` · **modo:** `rag`
**Filtros:** nenhum

Resumo: Os clientes relatam o recebimento de produtos quebrados, danificados ou com rachaduras, apontando também embalagens avariadas, voltagem errada e a necessidade de solicitar trocas ou resoluções com os lojistas [E1][E2][E3][E4][E5][E6][E7].
- Estado do produto e da embalagem: Clientes relatam produtos quebrados, rachadura no corpo da fonte apesar de funcional, e embalagem que veio danificada junto com o item quebrado [E1][E2][E3][E5][E6][E7].
- Outros problemas associados e percepção: Há relato de produto que, além de quebrado, chegou com a voltagem errada, e cliente que associa o dano à falta de cuidados do vendedor no envio [E1][E2].
- Pós-venda e solicitações: Clientes mencionam que estão aguardando resolução, aguardando ou tentando a troca ou devolução do item quebrado, e um relato aponta falta de resposta da loja após o contato [E1][E3][E4][E5][E6][E7].

> Base da resposta: 7 avaliações recuperadas por similaridade semântica e léxica. É uma amostra das avaliações mais relevantes, não uma contagem da base.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `ea53de1722fff5127f7a879e308b1b01` | 1 | 2018-07 | artes_e_artesanato | atrasada | 0.604 | sim | Mercadoria quebrada. O produto chegou com a voltagem errada e quebrado. Aguardo a troca da mercadoria. |
| E2 | `aa06519a57b802f420fd54410ad60613` | 1 | 2018-08 | consoles_games | no_prazo | 0.294 | sim | Entrega Danificada. Produto chegou danificado, com rachadura no corpo da fonte, entretanto funcional. Mas o fato do produto chegar dessa forma, demonstra que o vendedor nao tomou os devidos cuidados no envio. |
| E3 | `6afb9abc15ff32b17abc8f3f819ade1f` | 1 | 2017-12 | cool_stuff | no_prazo | 0.145 | sim | A embalagem veio danificada assim como o produto, quebrado. Entrei em contato com a loja e ainda não tive nenhuma resposta. |
| E4 | `6245c9a6fbd78013fb3c7128c1df45bb` | 1 | 2018-01 | eletronicos | no_prazo | 0.091 | sim | recebi a mercadoria danificada e estou aguardando que o problema seja resolvido da melhor maneira possivel |
| E5 | `ac8881d2f6663930894ce0e7dbc711c8` | 1 | 2018-08 | sem_itens | nao_entregue | 0.071 | sim | produto danificado. Meu produto chegou todo quebrado, quero trocar, por outro igual do mesmo |
| E6 | `206fcd0c7e9d3248ff891f31205ae6ee` | 3 | 2018-08 | moveis_decoracao | no_prazo | 0.057 | sim | O produto veio quebrado. Um os produtos veio quebrado, estou tentando trocar ou devolver. |
| E7 | `ea13f408f9f91e78274f1caa2c9663d2` | 1 | 2017-07 | moveis_decoracao | no_prazo | 0.056 | sim | Produto chegou quebrado, ficaram de trocar |

<sub>tempo total: 3403.2 ms · motivo do gate: 7 evidências relevantes</sub>

## 13. Tema específico

**Pergunta:** Como a paralisação dos caminhoneiros afetou as entregas segundo os clientes?

**Status:** `ok` · **modo:** `rag`
**Filtros:** nenhum

Resumo: Clientes relatam que a paralisação gerou lentidão, atrasos e não realização de entregas, embora também haja relatos de pedidos entregues no prazo e de impactos no atendimento. [E1][E2][E3][E4][E5][E6][E7][E8]
- Atrasos e entregas não realizadas: clientes associam a greve à demora na entrega, a atrasos e ao fato de o pedido não ter sido entregue. [E2][E3][E5][E7][E8]
- Cumprimento de prazos: clientes relatam que as encomendas foram entregues dentro do prazo e de forma rápida, mesmo durante a paralisação. [E4][E6]
- Atendimento: cliente aponta que a greve dos caminhoneiros e dos petroleiros afetou o atendimento aos clientes. [E1]

> Base da resposta: 8 avaliações recuperadas por similaridade semântica e léxica. É uma amostra das avaliações mais relevantes, não uma contagem da base.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| E1 | `5dddfbd7c02c4b525c553c66687a7e85` | 4 | 2018-05 | cool_stuff | no_prazo | 0.994 | sim | recomendo. Sempre fui bem atendida, entregaram no prazo. Acredito que por causa das greves dos caminhoneiros e dos petroleiros, isso afetou o atendimento aos clientes. |
| E2 | `bc462fa81c4a6cdf26c36ec388874f48` | 3 | 2018-04 | utilidades_domesticas | atrasada | 0.971 | sim | Entrega não efetuada. A greve dos caminhoneiros afetou diretamente o serviço de entregas de encomendas, mas estou no aguardo. |
| E3 | `acde992281f8a62cb1bbe210ddccfd8e` | 3 | 2018-05 | brinquedos | no_prazo | 0.925 | sim | Recomendo. Demorou mas chegou. A produção foi lenta. Envio após confirmação de pagamento também. No mais, a greve dos caminhoneiros afetou a entrega. |
| E4 | `7badf50b8b9567162b169939d11311bb` | 5 | 2018-06 | beleza_saude | no_prazo | 0.884 | sim | ***** super recomendo. Muito rápida mesmo com paralisação dos caminhoneiros não atrasou a entrega. Gostei muito. |
| E5 | `787fc6b4477d970b62ed684c6e92c35a` | 4 | 2018-05 | utilidades_domesticas | no_prazo | 0.714 | sim | Recomendo. Muito bom produto. Entrega demorou em razão da greve dos caminhoneiros. |
| E6 | `08b10333cabfbee56b6814150affd467` | 5 | 2018-05 | utilidades_domesticas | no_prazo | 0.455 | sim | Ótimo. Mesmo com a greve dos caminhoneiros entregaram dentro do prazo. Comprei no segundo dia de greve. |
| E7 | `8f7693b9a1a00201b178450c375b80ea` | 5 | 2018-05 | brinquedos | atrasada | 0.440 | sim | Sei que o atraso da entrega ocorreu em virtude da paralisação. |
| E8 | `8eddc1c3ac2a98a69844e8857d734614` | 1 | 2018-05 | automotivo | atrasada | 0.339 | sim | Atraso na Entrega. Acredito que a entrega não foi realizada devido a greve dos caminhoneiros. Vou aguardar mais alguns dias. |

<sub>tempo total: 3242.7 ms · motivo do gate: 21 evidências relevantes</sub>

## 14. Sem evidência na base

**Pergunta:** Quais reclamações existem sobre o programa de cashback?

**Status:** `sem_evidencias` · **modo:** `rag`
**Filtros:** nenhum

> Não encontrei avaliações relevantes para responder a esta pergunta na base de conhecimento (termos sem nenhuma ocorrência na base: 'cashback'; se houve erro de digitação ou você usou um sinônimo, reformule a pergunta). Não vou responder para não gerar informação sem suporte nos dados.

<sub>tempo total: 1681.3 ms · motivo do gate: termos sem nenhuma ocorrência na base: 'cashback'</sub>

## 15. Evidência insuficiente

**Pergunta:** O que os clientes relatam sobre a compra de patinetes?

**Status:** `evidencias_insuficientes` · **modo:** `rag`
**Filtros:** nenhum

> Encontrei apenas 1 avaliação(ões) relevante(s) (termos raros na base: 'patinetes' (1 avaliação(ões))). É pouco para identificar um padrão, então não vou generalizar. As evidências encontradas estão listadas abaixo para consulta direta.

| rótulo | review_id | nota | compra | categoria | entrega | relevância | citada | texto |
|---|---|---|---|---|---|---|---|---|
| - | `f49ea8aa6eb53bdd29e18cb2f9bbecc0` | 1 | 2017-09 | cool_stuff | no_prazo | 0.001 | não | Comprei um patinete pra minha filha recebi uma escova rotatória to aguardando até agora a troca e nada |

<sub>tempo total: 1339.7 ms · motivo do gate: termos raros na base: 'patinetes' (1 avaliação(ões))</sub>

## 16. Fora do escopo

**Pergunta:** Qual é a capital da Austrália?

**Status:** `fora_do_escopo` · **modo:** `rag`
**Filtros:** nenhum

> A pergunta parece estar fora do escopo da base de conhecimento, que contém apenas avaliações de clientes da Olist (pedidos de 2016 a 2018). Não vou responder para não gerar informação sem suporte nos dados. Além disso, o termo 'australia' não aparece em nenhuma avaliação da base.

<sub>tempo total: 1368.1 ms · motivo do gate: score de escopo 0.780 < limiar 0.832; termos sem nenhuma ocorrência na base: 'australia'</sub>

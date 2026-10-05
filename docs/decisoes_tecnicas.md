# Decisões técnicas

Formato: **decisão** · alternativas consideradas · motivo · como verificar.

## D1. Unidade do documento: 1 avaliação, sem chunking
- **Alternativas:** chunking de tamanho fixo com overlap (padrão em RAG de documentos longos); agrupar avaliações por pedido ou produto.
- **Motivo:** mediana de 9 palavras e máximo de 208 caracteres no comentário. Cortar não reduz nada e quebra o sentido. Agrupar misturaria clientes diferentes numa só evidência e destruiria a rastreabilidade por `review_id`.
- **Verificar:** notebook 01, seção 3.

## D2. Limpeza diferente por ramo (denso × léxico)
- **Alternativas:** um único pré-processamento pesado (minúsculas, sem acento, sem stopwords, lematização) para tudo.
- **Motivo:** modelos Transformer foram treinados com texto natural; remover acentos, stopwords e negações tira sinal. O BM25, ao contrário, ganha com normalização e stemming. Nos dois ramos as **negações são preservadas** ("não chegou" ≠ "chegou").
- **Verificar:** `text_utils.py`; teste `test_bm25_mantem_negacao`.

## D3. Modelo de embeddings: `intfloat/multilingual-e5-base`
- **Alternativas:** `paraphrase-multilingual-MiniLM-L12-v2` (usado em aula, 384 dim.), `multilingual-e5-small`, `BAAI/bge-m3` (568M parâmetros, mais pesado), LSA treinado no corpus (baseline clássico).
- **Motivo:** modelo multilíngue treinado para recuperação **assimétrica** (pergunta curta × passagem), com prefixos `query:`/`passage:`. Tamanho viável em CPU para indexar ~42 mil textos curtos.
- **Verificar:** `scripts/07_comparar_embeddings.py` mede precisão@8 e MRR de LSA, MiniLM, e5-small e e5-base no conjunto de teste. A escolha pode ser trocada no `.env` sem mudar código.
- **Medido** (`eval/resultados/comparacao_embeddings.csv`, sem reranker, 8 perguntas): na busca densa, precisão@8 de 0,625 no LSA, 0,594 no MiniLM, 0,844 no e5-small e 0,797 no e5-base. Na híbrida, 0,750, 0,828, 0,859 e 0,875. A medição confirma a família e5 sobre MiniLM e LSA, mas **não separa** e5-small de e5-base (diferença de 1 documento em 64 na híbrida). O e5-small indexa em um terço do tempo (636 s contra 1.846 s). O e5-base foi mantido porque índice, calibração e avaliação foram feitos com ele; o e5-small é alternativa equivalente nesta medição.

## D4. Índice: FAISS `IndexFlatIP` com pré-filtragem
- **Alternativas:** HNSW/IVF (aproximados); ChromaDB/Qdrant (bancos vetoriais com filtro de metadados embutido).
- **Motivo:** com ~42 mil vetores a busca exata custa milissegundos, então não há ganho que justifique o erro de aproximação. O `IDSelector` aplica o filtro **antes** da busca, e o top-k já sai do recorte certo (pós-filtragem poderia devolver menos de k). Também dispensa um serviço extra para instalar.
- **Quando mudar:** acima de milhões de vetores, ou com atualização contínua da base, um banco vetorial com HNSW passa a compensar.

## D5. Recuperação híbrida (densa + BM25) com RRF
- **Alternativas:** só densa; só BM25; soma ponderada de scores.
- **Motivo:** a densa acha paráfrases; o BM25 acha termos exatos e raros (marcas, "nota fiscal", "Correios"). O RRF combina **posições**, não scores, e por isso dispensa calibrar escalas diferentes (cosseno × BM25).
- **Verificar:** `recuperacao.csv` compara os quatro modos. No conjunto de teste (sem reranker), a precisão@8 foi de 0,672 no BM25, 0,797 na densa e 0,875 na híbrida, que também levou o MRR de 0,938 para 1,0. A fusão foi o maior ganho entre as etapas de recuperação.

## D6. Deduplicação de textos idênticos na recuperação
- **Motivo:** 17% dos documentos repetem literalmente outro texto ("muito bom", "recomendo"). Sem deduplicação, várias das 8 vagas de contexto poderiam ser ocupadas pela mesma frase. A cópia mantida é anotada com "texto idêntico em N avaliações". Na base, todas as avaliações continuam existindo e contando nas estatísticas.

## D7. Re-ranking com cross-encoder multilíngue
- **Alternativas:** sem re-ranking; reranking por LLM.
- **Motivo:** o cross-encoder lê pergunta e avaliação juntas e mede relevância melhor que o cosseno entre vetores independentes. Seu score (0–1) é a medida de relevância usada no gate. Rerankear com LLM seria mais caro e menos determinístico.
- **Medido:** no conjunto de teste, o reranker levou a precisão@8 da híbrida de 0,875 para 0,922 (melhorou 3 perguntas e piorou 1).
- **Configurável:** `RERANKER_MODEL` vazio desliga o reranker; nesse caso a relevância passa a ser o cosseno denso (só para o gate; a ordem continua a da fusão RRF) e os limiares devem ser recalibrados.

## D8. Filtros de metadados por regras explícitas
- **Alternativas:** *self-query* com LLM (o LLM gera o filtro em JSON).
- **Motivo:** regras são auditáveis, determinísticas e não inventam filtros. Também funcionam com LLM pequeno ou sem LLM. Os filtros inferidos e suas justificativas são mostrados ao usuário, que pode sobrescrevê-los na interface. Regra deliberada: "atraso" como **tema** não vira filtro de pedidos atrasados. O filtro só é aplicado quando a pergunta delimita pedidos ("pedidos entregues com atraso"), porque filtrar por atraso registrado excluiria quem reclama de **não ter recebido** (sem data de entrega).
- **Por que melhora a recuperação:** a busca semântica só enxerga o texto, e nota, data, categoria, UF e situação da entrega não estão nele. O pré-filtro faz a busca acontecer dentro do recorte pedido. Medido em `efeito_filtros.csv`: com o filtro, 100% das 8 evidências caem no recorte; sem ele, 25% em média, com 50% quando o critério aparece no texto (insatisfação) e de 0% a 12,5% quando ele só existe nos metadados (categoria, período, UF, entrega).
- **Verificar:** `filtros.csv`, `efeito_filtros.csv`; testes `test_filtros_*`, `test_sem_falso_filtro_em_tema`. Limitação: a extração de filtros é testada em só 5 perguntas.

## D9. Abstenção em três camadas, com limiares calibrados
- **Alternativas:** confiar só na instrução do prompt ("se não souber, diga que não sabe").
- **Motivo:** LLMs, sobretudo os pequenos, nem sempre obedecem. Por isso:
  - o **gate** decide antes de gerar (escopo, cobertura léxica, relevância, quantidade). Na camada analítica, as mesmas barreiras (termo ausente, escopo, termo raro) valem antes de contar;
  - o **prompt** permite recusar;
  - a **verificação** remove o que não tem citação.

  Os limiares de escopo e de relevância são escolhidos em um conjunto de calibração **separado** do teste, maximizando a acurácia balanceada (0,75 para escopo e 0,85 para relevância, nas 20 perguntas de calibração).
- **Resultado:** no teste, o gate sozinho acertou 87,1% dos status no critério tolerante (qualquer abstenção vale) e 74,2% no estrito (motivo exato); o sistema completo, 96,8% (30/31) e 80,6% (25/31). A diferença entre gate e sistema vem da camada analítica e de uma recusa do próprio LLM (T25); a diferença entre critérios, de abstenções pelo motivo vizinho. O score de escopo é o elo fraco: separa mal perguntas fora do escopo que usam vocabulário de compra (T30, "trocar o óleo do carro").
- **Checagem léxica:** um termo de conteúdo da pergunta que não aparece em **nenhuma** das 42 mil avaliações (ex.: "Pix", que surgiu depois do período dos dados) é evidência direta de ausência. Esse caso é declarado ao usuário, com o termo e uma sugestão de reformular.

## D10. Código conta, LLM redige (camada analítica)
- **Alternativas:** RAG puro para tudo; map-reduce com LLM sobre milhares de avaliações; GraphRAG.
- **Motivo:** frequência é uma pergunta sobre a **base inteira**, e top-k não responde isso. Contar com código é exato, barato e reprodutível. Map-reduce com LLM sobre ~42 mil textos seria caro e não determinístico. O LLM recebe a tabela pronta, e a verificação descarta qualquer número que não esteja nela.
- **Taxonomia por regras:** auditável (cada contagem pode ser explicada e conferida), roda em segundos e não exige rótulos. O custo é o recall incompleto, e por isso a cobertura é sempre informada. Problemas são contados nas notas 1–3 e elogios nas 4–5, para que menções neutras (ex.: "com nota fiscal, tudo certo", nota 5) não contem como problema.
- **Base mínima:** abaixo de 30 avaliações com comentário na base de contagem, o sistema não calcula porcentagens e mostra exemplos. É uma regra prática fixada, não calibrada.
- **Precisão auditada:** 94,9% (298/314) na auditoria humana de 15 avaliações sorteadas por tema (uma saiu do cálculo por não ser mais rotulada pelas regras atuais); o tema mais fraco é "Pedido não recebido" (11/14), que absorve entregas parciais. As regras não foram ajustadas depois da auditoria.

## D11. Pipeline explícito, sem framework de orquestração
- **Alternativas:** LangChain/LangGraph (vistos em aula).
- **Motivo:** o enunciado pede para **demonstrar** cada etapa. Funções Python explícitas deixam cada passo inspecionável (notebook 02), sem abstrações escondendo o prompt, o contexto ou a verificação. A camada de LLM é plugável e o restante não depende de framework, o que facilita migrar para LangGraph se o sistema virar um agente.

## D12. LLM plugável: local por padrão, Gemini nos resultados publicados
- **Resultados publicados:** gerados com `gemini-3.8-flash` via API (projeto com faturamento ativo, porque a cota diária do plano gratuito não comporta a avaliação completa). O modelo local não foi avaliado ponta a ponta nesta entrega.
- **Motivo do padrão local:** qualquer pessoa reproduz a solução sem chave de API nem custo (`Qwen2.5-1.5B-Instruct`, modelo usado em aula). Provedores de API (OpenAI, Gemini, Anthropic) e Ollama entram trocando três linhas do `.env` (provedor, modelo e chave). A temperatura é 0 quando o provedor aceita. Modelos com raciocínio embutido recusam (`claude-opus-5-5`) ou desaconselham (Gemini 3, cujo padrão recomendado é 1.0) temperatura customizada. Nesses casos ela não é enviada, o raciocínio é limitado por nível de esforço (`ANTHROPIC_EFFORT` / `LLM_REASONING_EFFORT`) e a reprodutibilidade exata da redação não é garantida; a verificação de citações e números, sim.
- **Trade-off:** o modelo local é lento em CPU e redige pior que modelos grandes; as camadas de verificação limitam o efeito disso na fundamentação.

## D13. Privacidade
- Telefones, e-mails, CPFs e URLs são mascarados na preparação, **antes** de embeddings, índice e LLM. Os nomes de vendedores já vêm anonimizados no dataset (casas de *Game of Thrones*).

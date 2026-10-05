"""Etapa 6 — Executa as perguntas de exemplo e grava as respostas REAIS em
docs/exemplos_perguntas_respostas.md (e .json). Nada neste arquivo é escrito
à mão: ele é a saída do sistema com a configuração do .env.

Uso:  python scripts/06_gerar_exemplos.py
"""
import json
from datetime import datetime

import _bootstrap  # noqa: F401

from voc_rag.config import ROOT, get_settings
from voc_rag.pipeline import VoCRAG

PERGUNTAS = [
    # Exemplos de exploração do enunciado
    ("Investigação de reclamações", "Quais são os principais problemas relatados pelos clientes em suas avaliações?"),
    ("Investigação de um tema específico", "O que os clientes relatam sobre problemas relacionados à entrega?"),
    ("Busca por experiências positivas", "Quais aspectos da experiência de compra são mais elogiados pelos clientes?"),
    ("Investigação de insatisfação", "Quais padrões podem ser identificados nas avaliações de clientes insatisfeitos?"),
    ("Exploração baseada em evidências", "Quais avaliações sustentam a conclusão de que existem problemas recorrentes relacionados ao prazo de entrega?"),
    # Perguntas do "Problema de Negócio" do enunciado que não repetem as anteriores
    ("Problema de negócio: frequência", "Quais reclamações relacionadas à entrega aparecem com maior frequência?"),
    ("Problema de negócio: qualidade", "Existem padrões recorrentes nas avaliações relacionadas à qualidade dos produtos?"),
    # Desafio adicional (metadados)
    ("Metadados: nota baixa", "Quais são os principais problemas relatados nas avaliações com nota baixa?"),
    ("Metadados: categoria", "O que os clientes mais elogiam em produtos de beleza e saúde?"),
    ("Metadados: entrega", "Quais temas aparecem nas avaliações relacionadas a experiências de entrega insatisfatórias?"),
    ("Metadados: entrega + UF", "O que dizem os clientes de SP sobre pedidos entregues com atraso?"),
    # Tema específico (RAG)
    ("Tema específico", "O que os clientes contam sobre mercadorias que chegaram quebradas ou danificadas?"),
    ("Tema específico", "Como a paralisação dos caminhoneiros afetou as entregas segundo os clientes?"),
    # Limitações reconhecidas (requisito 4)
    ("Sem evidência na base", "Quais reclamações existem sobre o programa de cashback?"),
    ("Evidência insuficiente", "O que os clientes relatam sobre a compra de patinetes?"),
    ("Fora do escopo", "Qual é a capital da Austrália?"),
]

if __name__ == "__main__":
    s = get_settings()
    rag = VoCRAG(s)
    gerado = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = ["# Exemplos de perguntas e respostas", "",
          f"> Gerado automaticamente por `scripts/06_gerar_exemplos.py` em {gerado}.",
          f"> Embeddings: `{rag.index.meta.get('embedding_model')}` · reranker: `{s.reranker_model or 'desligado'}` · "
          f"LLM: `{s.llm_provider}:{s.llm_model}` · limiares calibrados: `{rag.thresholds_calibrados}`.",
          "> As respostas abaixo são a saída literal do sistema; nenhuma foi editada.", ""]
    dump = []
    for i, (cat, q) in enumerate(PERGUNTAS, 1):
        print(f"[{i}/{len(PERGUNTAS)}] {q}")
        r = rag.ask(q)
        dump.append(r.to_dict())
        md += [f"## {i}. {cat}", "", r.to_markdown(), "",
               f"<sub>tempo total: {r.diagnostico.get('tempos_ms', {}).get('total') or r.diagnostico.get('tempo_total_ms')} ms"
               f" · motivo do gate: {r.diagnostico.get('motivo_gate', '-')}</sub>", ""]
    (ROOT / "docs" / "exemplos_perguntas_respostas.md").write_text("\n".join(md), encoding="utf-8")
    (ROOT / "docs" / "exemplos_perguntas_respostas.json").write_text(
        json.dumps(dump, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("Gravado em docs/exemplos_perguntas_respostas.md")

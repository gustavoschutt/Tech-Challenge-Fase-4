"""Executa o pipeline completo, na ordem, com a configuração do .env:

  1. preparação da base             4. calibração dos limiares do gate
  2. índice vetorial                5. avaliação no conjunto de teste
  3. temas (camada analítica)       6. exemplos de perguntas e respostas
                                    7. execução dos notebooks (outputs salvos)

Uso:  python scripts/run_all.py            (tudo)
      python scripts/run_all.py --sem-llm  (pula as etapas que precisam do LLM, inclusive os notebooks)
      python scripts/run_all.py --sem-notebooks
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("\n" + "=" * 80 + "\n$ " + " ".join(cmd) + "\n" + "=" * 80, flush=True)
    t = time.time()
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"-> ok em {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-llm", action="store_true")
    ap.add_argument("--sem-notebooks", action="store_true")
    a = ap.parse_args()
    py = sys.executable
    run([py, "scripts/01_preparar_base.py"])
    run([py, "scripts/02_construir_indice.py"])
    # Os temas vêm antes da avaliação: sem temas.parquet a camada analítica
    # fica indisponível e as perguntas agregadas cairiam no RAG puro.
    run([py, "scripts/03_construir_temas.py"])
    run([py, "scripts/04_calibrar_limiares.py"])
    run([py, "scripts/05_avaliar.py"] + (["--sem-llm"] if a.sem_llm else []))
    if not a.sem_llm:
        run([py, "scripts/06_gerar_exemplos.py"])
    if a.sem_llm and not a.sem_notebooks:
        print("\n--sem-llm: notebooks não executados (o notebook 02 chama o LLM). "
              "Rode sem --sem-llm para reexecutá-los.")
    elif not a.sem_notebooks:
        for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
            run([py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.timeout=3600", str(nb.relative_to(ROOT))])
    print("\nPronto. Resultados em eval/resultados/ e docs/exemplos_perguntas_respostas.md")

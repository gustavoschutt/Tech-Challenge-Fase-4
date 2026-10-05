"""Permite rodar os scripts sem `pip install -e .` (adiciona src/ ao path)
e garante saída UTF-8 no console (Windows usa cp1252 quando a saída é
redirecionada, o que quebraria caracteres como '≤' e '∈')."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

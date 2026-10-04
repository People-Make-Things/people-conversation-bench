from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# The PersonaPlex server modules import each other by bare name because they are
# bundled flat into the Modal image (see deploy/common.py SERVER_PYTHONPATH).
SERVER_DIR = ROOT / "models" / "personaplex" / "server"

for entry in (ROOT, SERVER_DIR):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

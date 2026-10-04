"""Repo-relative paths for analysis scripts. Do not hardcode a machine checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parents[1]
PHONATION = ANALYSIS / "phonation"
DOSE = ANALYSIS / "dose"
VOICE_PROMPT = ANALYSIS / "voice_prompt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PHONATION) not in sys.path:
    sys.path.insert(0, str(PHONATION))

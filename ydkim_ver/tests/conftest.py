"""Make the local ``src`` tree importable for source-layout tests.

AI disclosure: this test configuration was generated with OpenAI Codex
assistance on 2026-08-17. Independent review is pending.
"""

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

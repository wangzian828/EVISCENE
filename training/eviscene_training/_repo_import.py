"""Make the monorepo runtime importable from the training package."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "eviscene" / "__init__.py").is_file() and str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

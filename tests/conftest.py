import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "machine"))

from orchestrator.config import Config  # noqa: E402


@pytest.fixture
def cfg() -> Config:
    """The shipped config.yaml, as-is (VERIFY values null — fine for the sim)."""
    return Config.load(ROOT / "orchestrator" / "config.yaml")

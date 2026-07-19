from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_livemap_frontend_regressions() -> None:
    """Run the dependency-free behavioral tests against the shipped app.js."""
    repo = Path(__file__).parents[1]
    subprocess.run(
        [NODE, "--test", "tests/test_livemap_frontend.mjs"],
        cwd=repo,
        check=True,
    )

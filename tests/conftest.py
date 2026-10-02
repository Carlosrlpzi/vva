"""Shared test helpers: run the real CLI entry point in-process."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))  # make `builders` importable

from vva_contracts.cli import main


@pytest.fixture
def run_cli(monkeypatch, capsys):
    """Call `vva run --workspace <ws>` with a request dict; return (exit, json)."""

    def _run(workspace: Path, request: dict) -> tuple[int, dict]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
        code = main(["run", "--workspace", str(workspace)])
        out = capsys.readouterr().out.strip().splitlines()
        assert len(out) == 1, "the CLI must print exactly one JSON document"
        return code, json.loads(out[0])

    return _run

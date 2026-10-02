"""Shared fixtures: a temporary workspace, a synthetic YOLO dataset and a CLI runner.

Images are random noise so every image has a distinct dHash, unless a test
copies one on purpose to plant a near duplicate.
"""

from __future__ import annotations

import io
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from vva_contracts.cli import main

CLASS_NAMES = ["person", "car"]


def write_image(path: Path, seed: int, size: tuple[int, int] = (320, 240)) -> None:
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rng.integers(0, 256, size=(size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)


def write_labels(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws.resolve()


@pytest.fixture
def dataset(workspace: Path) -> Path:
    """Two splits, 4 images each, file names ``clipA_000`` etc. (group = clip)."""
    root = workspace / "data" / "ds"
    (root).mkdir(parents=True)
    (root / "data.yaml").write_text("names:\n  0: person\n  1: car\n", encoding="utf-8")
    seed = 0
    for split, clip in (("train", "clipA"), ("val", "clipB")):
        for i in range(4):
            seed += 1
            write_image(root / "images" / split / f"{clip}_{i:03d}.jpg", seed)
            # One person per image; a car in every other image.
            lines = ["0 0.5 0.5 0.2 0.6"]
            if i % 2 == 0:
                lines.append("1 0.25 0.75 0.3 0.2")
            write_labels(root / "labels" / split / f"{clip}_{i:03d}.txt", lines)
    return root


CliRunner = Callable[[dict[str, Any], Path], tuple[int, dict[str, Any]]]


@pytest.fixture
def run_cli(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """Run ``vva-contract run`` in-process; return (exit code, parsed stdout)."""

    def _run(request: dict[str, Any], workspace: Path) -> tuple[int, dict[str, Any]]:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request)))
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        code = main(["run", "--workspace", str(workspace)])
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        assert len(lines) == 1, "stdout must contain exactly one JSON document"
        return code, json.loads(lines[0])

    return _run

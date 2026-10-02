"""Deterministic synthetic inputs for tests and for the ``examples/`` workspace.

Run ``python tests/builders.py examples`` to regenerate the example workspace.
Every builder uses a fixed seed, so the files (and their SHA-256) are stable.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["person", "car"]


def _noise_image(path: Path, seed: int, size: tuple[int, int] = (320, 240)) -> None:
    """Random-noise image: two different seeds give unrelated perceptual hashes."""
    rng = np.random.default_rng(seed)
    # Low-resolution noise upscaled: coarse structure that survives the 9x8 dHash.
    coarse = rng.integers(0, 256, size=(12, 16, 3), dtype=np.uint8)
    Image.fromarray(coarse).resize(size, Image.Resampling.NEAREST).save(path)


def make_dataset(root: Path, *, leak: bool = False, bad_label: bool = False) -> Path:
    """YOLO dataset with 6 train and 4 val images; returns the data.yaml path.

    ``leak`` copies a train frame (slightly brightened) into val, the classic
    consecutive-frame leakage. ``bad_label`` writes an out-of-range class id.
    """
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    boxes = {  # key -> list of "class cx cy w h"
        "train/cam1_20261001_0001": ["0 0.50 0.50 0.20 0.40", "1 0.20 0.70 0.30 0.20"],
        "train/cam1_20261001_0002": ["0 0.40 0.50 0.10 0.30"],
        "train/cam2_20261001_0001": ["1 0.60 0.60 0.40 0.30"],
        "train/cam2_20261001_0002": ["0 0.30 0.40 0.05 0.10"],
        "train/cam2_20261001_0003": [],
        "train/cam1_20261001_0003": ["0 0.70 0.50 0.15 0.35"],
        "val/cam3_20261002_0001": ["0 0.50 0.50 0.25 0.50"],
        "val/cam3_20261002_0002": ["1 0.40 0.60 0.30 0.25", "0 0.80 0.50 0.10 0.30"],
        "val/cam3_20261002_0003": ["0 0.20 0.30 0.10 0.20"],
        "val/cam4_20261002_0001": [],
    }
    for seed, (key, lines) in enumerate(boxes.items()):
        split, name = key.split("/")
        _noise_image(root / "images" / split / f"{name}.jpg", seed)
        if lines or name.endswith("0003"):  # some background images have no label file
            (root / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n")
    if leak:
        with Image.open(root / "images/train/cam1_20261001_0001.jpg") as img:
            arr = np.clip(np.asarray(img, dtype=np.int16) + 6, 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(root / "images/val/cam1_20261001_0004.jpg")
        (root / "labels/val/cam1_20261001_0004.txt").write_text("0 0.50 0.50 0.20 0.40\n")
    if bad_label:
        (root / "labels/val/cam3_20261002_0003.txt").write_text("7 0.20 0.30 0.10 0.20\n")
    yaml_path = root / "data.yaml"
    yaml_path.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [person, car]\n")
    return yaml_path


def gt_predictions(root: Path, split: str = "val", jitter: float = 0.0, seed: int = 0) -> list[dict[str, object]]:
    """Predictions equal to the ground truth (plus optional jitter), in xyxyn."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for label in sorted((root / "labels" / split).glob("*.txt")):
        for line in label.read_text().split("\n"):
            if not line.strip():
                continue
            c, cx, cy, w, h = line.split()
            x1, y1 = float(cx) - float(w) / 2, float(cy) - float(h) / 2
            box = np.array([x1, y1, x1 + float(w), y1 + float(h)]) + rng.normal(0, jitter, 4)
            rows.append({"image": label.stem, "class_id": int(c), "confidence": 0.9, "box": [float(v) for v in box]})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    """Write one JSON object per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def frames_log(fps: float = 10.0, seconds: float = 600.0) -> list[dict[str, object]]:
    """Two cameras; 'front' sees a person from t=100 s to t=130 s, plus noise.

    The 'yard' camera has a bush detected as a person at 0.55 confidence for
    long stretches: the kind of correlated false positive that p^k ignores.
    """
    rows: list[dict[str, object]] = []
    n = int(fps * seconds)
    for i in range(n):
        ts = 1_000_000.0 + i / fps
        t = i / fps
        front = []
        if 100.0 <= t <= 130.0:
            front.append({"class_name": "person", "confidence": 0.82, "box": [0.40, 0.30, 0.55, 0.90]})
        if i % 97 == 0:  # isolated single-frame false positive
            front.append({"class_name": "person", "confidence": 0.65, "box": [0.05, 0.05, 0.10, 0.20]})
        rows.append({"camera_id": "front", "frame_idx": i, "ts": ts, "detections": front})
        yard = []
        if (t // 60) % 2 == 0:  # bush "person" for 60 s on, 60 s off
            yard.append({"class_name": "person", "confidence": 0.55, "box": [0.70, 0.50, 0.80, 0.80]})
        rows.append({"camera_id": "yard", "frame_idx": i, "ts": ts, "detections": yard})
    return rows


def events_log() -> list[dict[str, object]]:
    """The one real event in ``frames_log``."""
    return [{"camera_id": "front", "start_ts": 1_000_100.0, "end_ts": 1_000_130.0, "label": "person"}]


def timing_log(n: int = 2000, fps: float = 10.0, drop_every: int = 0) -> list[dict[str, object]]:
    """Per-frame timing for one camera; inference ~40 ms, end-to-end ~80 ms."""
    rng = np.random.default_rng(1)
    rows: list[dict[str, object]] = []
    for i in range(n):
        t = 1_000_000.0 + i / fps
        if drop_every and i % drop_every == 0:
            rows.append({"camera_id": "front", "frame_idx": i, "t_capture": t, "dropped": True})
            continue
        start = t + abs(rng.normal(0.02, 0.005))
        end = start + abs(rng.normal(0.04, 0.005))
        rows.append(
            {
                "camera_id": "front",
                "frame_idx": i,
                "t_capture": t,
                "t_infer_start": start,
                "t_infer_end": end,
                "t_done": end + 0.02,
                "dropped": False,
            }
        )
    return rows


def make_video(path: Path, fps: int = 10, seconds: int = 6, size: str = "320x240") -> Path:
    """Encode a short H.264 test clip with ffmpeg (keyframe every 20 frames)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={size}:rate={fps}",
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-g",
            "20",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


def build_examples(root: Path) -> None:
    """Populate an example workspace exercising every task."""
    make_dataset(root / "dataset")
    write_jsonl(root / "predictions/val_fp32.jsonl", gt_predictions(root / "dataset", jitter=0.004, seed=1))
    write_jsonl(root / "predictions/val_int8.jsonl", gt_predictions(root / "dataset", jitter=0.012, seed=2))
    write_jsonl(root / "logs/frames.jsonl", frames_log())
    write_jsonl(root / "logs/events.jsonl", events_log())
    write_jsonl(root / "logs/timings.jsonl", timing_log(drop_every=250))
    write_jsonl(
        root / "logs/system.jsonl", [{"ts": 1_000_000.0 + 5 * i, "cpu_temp_c": 62.0 + i % 7} for i in range(40)]
    )
    make_video(root / "clips/front_sub.mp4")


if __name__ == "__main__":
    build_examples(Path(sys.argv[1]))

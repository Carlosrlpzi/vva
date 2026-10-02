"""DETECTION_EVAL, QUANT_PARITY and DATASET_AUDIT through the real CLI path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from conftest import CliRunner, write_labels

from vva_contracts.contracts.dataset_audit import DatasetAuditReport
from vva_contracts.contracts.detection_eval import DetectionEvalReport


def _yolo_to_pixels(line: str, w: int, h: int) -> tuple[str, list[float]]:
    cls, cx, cy, bw, bh = line.split()
    name = ["person", "car"][int(cls)]
    x1, y1 = (float(cx) - float(bw) / 2) * w, (float(cy) - float(bh) / 2) * h
    return name, [x1, y1, x1 + float(bw) * w, y1 + float(bh) * h]


def _write_predictions(
    dataset: Path, out: Path, *, drop_cars: bool = False, size: tuple[int, int] = (320, 240)
) -> None:
    """Perfect predictions copied from the val labels (optionally missing all cars)."""
    lines = []
    for label in sorted((dataset / "labels" / "val").glob("*.txt")):
        dets = []
        for raw in label.read_text().splitlines():
            name, box = _yolo_to_pixels(raw, 320, 240)
            if drop_cars and name == "car":
                continue
            dets.append({"class_name": name, "confidence": 0.9, "bbox_xyxy": box})
        lines.append(json.dumps({"image_id": label.stem, "width": size[0], "height": size[1], "detections": dets}))
    out.write_text("\n".join(lines) + "\n")


def test_perfect_predictions_score_one(workspace: Path, dataset: Path, run_cli: CliRunner) -> None:
    _write_predictions(dataset, workspace / "preds.jsonl")
    code, report = run_cli(
        {"task": "DETECTION_EVAL", "dataset_root": "data/ds", "predictions": "preds.jsonl"}, workspace
    )
    assert code == 0, report
    assert report["map50"] == pytest.approx(1.0)
    assert report["map50_95"] == pytest.approx(1.0)
    assert {i["role"] for i in report["inputs"]} == {"data_yaml", "split_manifest", "predictions"}


def test_letterbox_coordinates_are_rejected(workspace: Path, dataset: Path, run_cli: CliRunner) -> None:
    _write_predictions(dataset, workspace / "preds.jsonl", size=(640, 640))
    code, report = run_cli(
        {"task": "DETECTION_EVAL", "dataset_root": "data/ds", "predictions": "preds.jsonl"}, workspace
    )
    assert code == 3
    assert "original-image pixels" in report["error"]["message"]


def test_tampered_report_is_rejected(workspace: Path, dataset: Path, run_cli: CliRunner) -> None:
    """Editing one AP by hand breaks the recomputation invariants."""
    _write_predictions(dataset, workspace / "preds.jsonl", drop_cars=True)
    code, report = run_cli(
        {"task": "DETECTION_EVAL", "dataset_root": "data/ds", "predictions": "preds.jsonl"}, workspace
    )
    assert code == 0
    car = next(c for c in report["per_class"] if c["class_name"] == "car")
    assert car["ap50"] == 0.0
    car["ap50"] = 0.9  # pretend the car detector works
    with pytest.raises(ValueError, match="ap50"):
        DetectionEvalReport.model_validate_json(json.dumps(report))


def test_quant_parity_fails_when_candidate_loses_a_class(workspace: Path, dataset: Path, run_cli: CliRunner) -> None:
    _write_predictions(dataset, workspace / "fp32.jsonl")
    _write_predictions(dataset, workspace / "int8.jsonl", drop_cars=True)
    code, report = run_cli(
        {
            "task": "QUANT_PARITY",
            "dataset_root": "data/ds",
            "reference_predictions": "fp32.jsonl",
            "candidate_predictions": "int8.jsonl",
        },
        workspace,
    )
    assert code == 0, report
    assert report["delta_map50"] == pytest.approx(-0.5)  # car AP 1 -> 0, mean over 2 classes
    assert report["verdict"] == "FAIL"
    # 4 people match; 2 reference cars have no partner: dice = 2*4 / (6 + 4).
    assert report["box_agreement"]["dice"] == pytest.approx(0.8)


def test_clean_dataset_warns_only_about_unchecked_groups(workspace: Path, dataset: Path, run_cli: CliRunner) -> None:
    code, report = run_cli(
        {"task": "DATASET_AUDIT", "dataset_root": "data/ds", "inference_size": [640, 360]}, workspace
    )
    assert code == 0, report
    assert report["risk_flags"] == ["GROUP_LEAKAGE_UNCHECKED"]
    assert report["verdict"] == "WARN"
    person = next(s for s in report["splits"][0]["box_sizes"] if s["class_name"] == "person")
    assert person["height_px_p50"] == pytest.approx(0.6 * 360)


def test_audit_detects_leakage_and_label_errors(workspace: Path, dataset: Path, run_cli: CliRunner) -> None:
    # Plant a near duplicate: copy a train image into val under a new name and group.
    shutil.copy(dataset / "images/train/clipA_000.jpg", dataset / "images/val/clipA_900.jpg")
    write_labels(dataset / "labels/val/clipA_900.txt", ["0 0.5 0.5 0.2 0.6", "7 0.5 0.5 0.1 0.1"])  # class 7 invalid
    code, report = run_cli(
        {
            "task": "DATASET_AUDIT",
            "dataset_root": "data/ds",
            "inference_size": [640, 360],
            "group_regex": r"^(?P<group>clip[A-Z])_",
        },
        workspace,
    )
    assert code == 0, report
    assert report["verdict"] == "FAIL"
    assert {"LABEL_ERRORS", "CROSS_SPLIT_NEAR_DUPLICATES", "GROUP_LEAKAGE"} <= set(report["risk_flags"])
    assert report["group_leakage"]["groups_in_multiple_splits"] == ["clipA"]
    assert report["near_duplicates"]["images_with_match_in_reference"] == {"val": 1}

    # A hand-edited PASS cannot survive validation.
    report["verdict"] = "PASS"
    with pytest.raises(ValueError, match="verdict"):
        DatasetAuditReport.model_validate_json(json.dumps(report))

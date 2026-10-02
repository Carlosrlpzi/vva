---
name: ml-train-baseline
description: Train a reproducible supervised baseline (sklearn) on a CSV/Parquet file. Emits ONLY a ModelArtifactSpec JSON (algorithm, serialized hyperparameters, CV + holdout ROC-AUC/F1/RMSE, confusion matrix, ONNX/joblib artifact with sha256).
user-invocable: true
metadata: {"openclaw": {"skillKey": "ml-train-baseline", "primaryEnv": "DEEPSEEK_API_KEY", "requires": {"bins": ["python3"], "env": ["DEEPSEEK_API_KEY"]}}}
---

# ml-train-baseline

Use when the router context says `task='TRAINING'`. Requires a data path AND a target column; if either is missing, ask for it.

## Run

```bash
python3 -m mlclaw.skills.train_baseline_model --path '<DATA_PATH>' --target '<TARGET>' [--drop '<COL>' ...] --output-dir "$HOME/.openclaw/workspace/artifacts"
```

- If the latest EDA report flagged `leakage_suspect` columns, pass them in `--drop`.
- The skill writes a standalone `train_<experiment_id>.py` next to the artifact. Report its path; do not rewrite it.

## Output contract

- Exit 0: return the `ModelArtifactSpec` JSON verbatim in ONE ```json block.
- Exit 2: training failed or the plan violated sklearn constraints. Return the `contract_violation` JSON verbatim.
- Do not claim model quality beyond the `validation` and `cross_validation` fields.

Schema: `{baseDir}/../../schemas/ModelArtifactSpec.schema.json`

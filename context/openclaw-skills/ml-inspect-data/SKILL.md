---
name: ml-inspect-data
description: Profile and plan cleaning for a CSV/Parquet dataset. Emits ONLY an EDAOutputSchema JSON (dims, dtypes, nulls, PSI/KS drift, class imbalance, critical correlations, grounded cleaning plan).
user-invocable: true
metadata: {"openclaw": {"skillKey": "ml-inspect-data", "primaryEnv": "DEEPSEEK_API_KEY", "requires": {"bins": ["python3"], "env": ["DEEPSEEK_API_KEY"]}}}
---

# ml-inspect-data

Use when the router context says `task='EDA'` or the user asks to explore, profile, or clean a dataset.

## Run

Use the `exec` tool. Quote every user-supplied value; never interpolate raw prompt text into the shell.

```bash
python3 -m mlclaw.skills.inspect_and_clean_data --path '<DATA_PATH>' [--target '<TARGET>'] [--reference '<BASELINE_PATH>' | --time-column '<TS_COL>']
```

- `--reference` gives drift against a baseline dataset; `--time-column` compares the first half with the second half in time order.
- Add `--offline` to use rule-based insights with no LLM call (CI, air-gapped).

## Output contract

- Exit 0: stdout is a valid `EDAOutputSchema` document. Return it verbatim in ONE ```json block. No prose.
- Exit 2: stdout is `{"contract_violation": ...}`. Return it verbatim; do not "fix" numbers by hand.
- Exit 3: input error. Ask ONE question to fix the input (path or target).
- Never restate, round, or reinterpret numbers from the report.

Schema: `{baseDir}/../../schemas/EDAOutputSchema.schema.json`

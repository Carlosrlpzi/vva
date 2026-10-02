---
name: ml-validate-code
description: Strict review of Python/ML code with Ruff, Mypy --strict, Radon cyclomatic complexity, an AST vectorization scan (iterrows, range(len), apply(axis=1), concat-in-loop), and generated Pytest tests. Emits ONLY a CodeReviewAndRefactor JSON.
user-invocable: true
metadata: {"openclaw": {"skillKey": "ml-validate-code", "primaryEnv": "DEEPSEEK_API_KEY", "requires": {"bins": ["python3", "ruff"], "env": ["DEEPSEEK_API_KEY"]}}}
---

# ml-validate-code

Use when the router context says `task='CODE_REVIEW'`. If the code is inline in the chat, first write it to
`$HOME/.openclaw/workspace/review/input.py` with the `write` tool, then run the skill on that file.

## Run

```bash
python3 -m mlclaw.skills.validate_ml_code --file '<PY_FILE>' --max-iterations 3
```

The skill runs the model-generated code and tests. Only run it inside the OpenClaw sandbox.

## Output contract

- Exit 0: return the `CodeReviewAndRefactor` JSON verbatim in ONE ```json block.
- `verdict` is computed from the evidence (lint, mypy, complexity <= 10, no vectorization issues, green tests). Never override it.
- If the user asks for "just the fixed code", return `proposal.refactored_code` in ONE ```python block and nothing else.

Schema: `{baseDir}/../../schemas/CodeReviewAndRefactor.schema.json`

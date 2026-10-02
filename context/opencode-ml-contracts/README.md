# OpenCode ML Contracts

A Jev-free adaptation of the original ML skill suite for OpenCode and DeepSeek Flash.
The Python contracts and computational core are retained; the OpenClaw plugin, server,
configuration, classifier, and TypeSafe SDK dependency are not included.

## Architecture

```text
OpenCode primary agent: DeepSeek Flash
    -> explicit ml_contract tool arguments
    -> TypeScript wrapper, spawning Python without a shell
    -> strict MLRequest validation and project-path policy
    -> EDA / supervised training / Python code review / bounded technical answer
    -> measured evidence + optional additional DeepSeek Flash call
    -> complete Pydantic contract validation
    -> one JSON document
```

The model can select a tool task, but the task dispatcher is deterministic.
There is no intent-confidence estimator, regex classification fallback, Jev SDK, or Jev key.
Ambiguous inputs should trigger clarification instead of an invented confidence score.

OpenCode supports project `SKILL.md` files and JavaScript/TypeScript tools that invoke Python;
this package uses those documented extension points ([OpenCode skills](https://opencode.ai/docs/skills/),
[OpenCode custom tools](https://opencode.ai/docs/custom-tools/)).
The provider configuration uses a custom OpenAI-compatible provider and an environment-variable
API key rather than a secret in the project files ([OpenCode providers](https://opencode.ai/docs/providers/),
[OpenCode configuration](https://opencode.ai/docs/config/)).
DeepSeek's documentation identifies `deepseek-flash` as DeepSeek-V4.1-Flash as of this adaptation
([DeepSeek changelog](https://api-docs.deepseek.com/updates/)).

## Requirements

- Python 3.11 or later. Use an isolated virtual environment.
- OpenCode 1.18.34 is the runtime tested here.
- Node.js 22.22.2 or later is recommended for the bundled npm dependencies.
- Python dependencies from `pyproject.toml`, including `.[dev]` for Pandas typing stubs.
- A DeepSeek key for the OpenCode agent and any model-assisted skill operation.
- No Jev account, key, classifier, FastAPI service, or OpenClaw installation.

The Python distribution is named `mlcode-contracts`, but its import namespace remains `mlclaw`
to minimize changes to the reusable core. Do not install the original `mlclaw` distribution
and this distribution into the same virtual environment; they share an import namespace.

## Install on WSL2 or Linux

Unzip this folder, enter it, and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
npm --prefix .opencode ci
```

Optional ONNX export:

```bash
python -m pip install -e ".[dev,onnx]"
```

Supply `DEEPSEEK_API_KEY` in the environment before launching OpenCode. Do not commit it or put it
in `opencode.json`. You can use a shell prompt that does not record the value in history:

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
export DEEPSEEK_API_KEY
printf '\n'
npx --yes opencode-ai@1.18.34
```

This custom provider uses the environment key directly. A key stored only through OpenCode's
`/connect` is not automatically made available to the Python client's environment.
The same Flash model is configured for primary and lightweight OpenCode tasks.

The tool automatically uses `.venv/bin/python` when present. The operator may override the interpreter
with `MLCODE_PYTHON=/absolute/path/to/python`. Arguments from the model cannot set that variable.
For native Windows, activate `.venv\Scripts\Activate.ps1`; the wrapper also detects the Windows
virtual-environment interpreter. Native Windows has not been runtime-tested in this environment.
WSL2 is the recommended starting point.

## Start without an API key

The Python CLI and offline checks work without any model key:

```bash
printf '%s\n' '{"task":"EDA","path":"examples/churn.csv","target":"churn","time_column":"ts","offline":true}' \
  | mlcode run --workspace .
```

The exact same request can be supplied to the OpenCode `ml_contract` tool. However, OpenCode's
agent itself still needs a model provider; `offline=true` disables the skill's additional
DeepSeek call, not the OpenCode agent's reasoning calls.

## Tasks and contracts

| Task | Required input | Output | Offline behavior |
|---|---|---|---|
| `EDA` | Relative CSV/Parquet path | `EDAOutputSchema` | Deterministic statistics and cleaning recommendations |
| `TRAINING` | Relative CSV/Parquet path and target | `ModelArtifactSpec` | Real training with a deterministic baseline plan |
| `CODE_REVIEW` | Relative `.py` path | `CodeReviewAndRefactor` | Ruff, Mypy, Radon, AST diagnostics only |
| `RAW_QUERY` | Query text, `offline=false` | `RawQueryAnswer` | Not supported; returns an input-validation error |

All tasks reject extra request fields. Booleans and integers are not silently coerced from strings.
Task-specific fields are checked: for example, `target` is not allowed on a code-review request.
Use workspace-relative paths; absolute paths, parent-directory escapes, and symlinks resolving
outside the active project directory are rejected. The workspace is OpenCode's session directory,
not its potentially different Git worktree root.

### EDA

```json
{
  "task": "EDA",
  "path": "examples/churn.csv",
  "target": "churn",
  "time_column": "ts",
  "offline": true
}
```

Use `reference` instead of `time_column` for drift against a separate dataset.
`max_rows` defaults to 200,000 and is bounded at 1,000,000. Optional `seed` controls sampling.
Live EDA sends computed summaries to DeepSeek for grounded insights; it does not ask the model
to invent the measured statistics. The original dataset is never cleaned in place.

### Training

Training requires an operator-set environment gate, including in offline mode:

```bash
export MLCODE_ALLOW_TRAINING=1
printf '%s\n' '{"task":"TRAINING","path":"examples/churn.csv","target":"churn","drop_columns":["leak"],"offline":true}' \
  | mlcode run --workspace .
```

The synthetic example includes a deliberately target-like `leak` column. Inspect the EDA first;
the training skill does not automatically detect or remove every form of leakage.
Outputs are created under `artifacts/mlcode/training/<experiment-id>/`.
The template performs a holdout split, train-only preprocessing, cross-validation, fitting,
measured validation metrics, serialization, and artifact/script hashes.
ONNX is selected when its optional dependencies are installed; otherwise the baseline uses joblib.
Never load untrusted joblib/pickle artifacts.

Live training sends a compact dataset summary to DeepSeek to choose a legal training plan.
Numerical metrics still come from executing the training template. The model does not write
an arbitrary training program. For temporal, grouped, survival, or causal tasks, extend the
split/task contracts first; the existing baseline is not a general experiment-design engine.

### Code review and refactoring

Safe starting point, diagnostics only:

```bash
printf '%s\n' '{"task":"CODE_REVIEW","path":"examples/bad_features.py","offline":true}' \
  | mlcode run --workspace .
```

For live refactoring, use an isolated execution environment and deliberately enable:

```bash
export MLCODE_ALLOW_GENERATED_CODE=1
printf '%s\n' '{"task":"CODE_REVIEW","path":"examples/bad_features.py","offline":false,"max_iterations":3}' \
  | mlcode run --workspace .
```

The source is sent to DeepSeek. Its proposed module and tests are written to new directories;
the original file is not modified. Ruff/Mypy/AST checks and Pytest produce the review evidence.
Ruff and Mypy infrastructure failures reject the review rather than being interpreted as clean code.
Generated Pytest processes receive a small environment allowlist, not API keys or arbitrary
parent environment variables. Plugin autoload is disabled and ancestor pytest configurations
are not used.

**These measures are not a sandbox.** Generated code still has the process user's file/network
privileges. Run untrusted live refactors only in a properly isolated, disposable environment
with restricted mounts, credentials, network, CPU/memory, and filesystem permissions.
The included gate does not verify that such an environment exists. Do not enable it on a sensitive
host merely because the tool asks. Full per-test container isolation is not implemented here.
Passing model-generated tests also does not prove equivalence or adversarial safety.

### Conceptual questions

```json
{"task":"RAW_QUERY","query":"Explain the difference between RMSE and MAE.","offline":false}
```

This returns a bounded answer contract, not a fabricated EDA or training report.
There is no offline stock-answer fallback.

## Output enforcement

The default Python model client uses DeepSeek JSON mode plus dynamically grounded schemas,
Pydantic validation, and up to two repair attempts. Failure after all attempts raises a typed
contract violation; it never silently accepts malformed output.
DeepSeek documents JSON mode separately from its beta strict tool-calling mode, which supports
only a schema subset ([DeepSeek JSON output](https://api-docs.deepseek.com/guides/json_mode),
[DeepSeek tool calls](https://api-docs.deepseek.com/guides/tool_calls)).
This package deliberately does not enable beta strict decoding by default.

Pydantic also checks custom semantic relationships that an ordinary JSON schema cannot enforce,
including accuracy versus confusion-matrix counts, MAE versus RMSE, fold means, drift severity,
grounded flags, and evidence-derived review verdicts.
These checks reject specific inconsistencies; they do not make every model judgment correct.

`mlcode run` is the strict machine boundary: successful stdout is exactly one JSON object with the
selected contract, validated again after the skill returns. In the interactive UI, the agent is
instructed to repeat that JSON unchanged, but its final chat message is not mechanically guaranteed.
For automation, consume the CLI/tool result, not a model-paraphrased final message.

OpenCode also has SDK-level JSON-schema output, but that does not replace Python's custom validators;
this adaptation uses the Python tool boundary instead of an SDK final-answer wrapper
([OpenCode SDK](https://opencode.ai/docs/sdk/)).

| Exit code | Meaning |
|---|---|
| `0` | Accepted output contract |
| `2` | Request/output contract violation |
| `3` | Missing or invalid input |
| `4` | Execution policy blocked |
| `5` | DeepSeek provider failure; no artifact accepted |
| `6` | Subprocess timeout; no artifact accepted |

The TypeScript wrapper rejects nonzero exits and malformed/non-object stdout. It does not
present typed errors as successful artifacts. It uses no shell interpolation, limits output
to 8 MiB, and applies a 40-minute outer timeout.

The operator gates and path checks apply to the OpenCode bridge. The retained low-level Python
skill functions and standalone skill module CLIs are library APIs, not a security boundary;
callers invoking them directly are responsible for their own policy and isolation.

## Configuration and permissions

`opencode.json` sets a primary `ml-contracts` agent with read/search/skill access and an
approval-required `ml_contract` tool. General shell execution, editing, and other tools are denied
by the wildcard rule. Review permission prompts carefully before allowing live data/code transfer.
Agent-level permissions are a documented configuration mechanism
([OpenCode agents](https://opencode.ai/docs/agents/)).

When integrating into an existing repository, merge `opencode.json` rather than overwriting
your existing providers or agents. Copy `.opencode/tools`, `.opencode/lib`, and `.opencode/skills`,
merge `.opencode/package.json`, install this Python package into that project's virtual environment,
and place/merge `AGENTS.md` where the prompt config references it.
Re-export or copy `schemas/` for local inspection.

## Verify locally

```bash
python -m ruff check src tests
python -m mypy --strict src
python -m pytest -q
npm --prefix .opencode run typecheck
npm --prefix .opencode test
npm --prefix .opencode run smoke
```

The smoke test starts a loopback-only mock model server and invokes the actual OpenCode 1.18.34 CLI.
It temporarily permits the tool only in that child process, confirms discovery and execution,
and uses no real DeepSeek key or external model inference.
It may download the pinned OpenCode CLI through npm if not cached.

Schemas can be regenerated:

```bash
mlcode schemas --output schemas
```

An existing output can be checked with the full Python contract:

```bash
mlcode validate --task EDA --file report.json
```

Validation checks structure and invariants, not provenance. Do not treat an arbitrary external
JSON report as measured evidence merely because it passes this command.
For reproducibility across machines, also capture a tested Python dependency lock/environment;
the package's minimum dependency constraints are not a complete environment lock.

## Package layout

```text
opencode.json
AGENTS.md
pyproject.toml
.opencode/
  package.json, package-lock.json, tsconfig.json
  tools/ml_contract.ts
  lib/bridge.ts
  skills/
    ml-inspect-data/SKILL.md
    ml-train-baseline/SKILL.md
    ml-validate-code/SKILL.md
  tests/
src/mlclaw/
  opencode.py
  contracts/
  analysis/
  llm/
  skills/
  router/prompts.py
schemas/
examples/
tests/
VERIFICATION.md
```

The original OpenClaw package is left unchanged. This distribution is a separate adaptation.

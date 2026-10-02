# Verification Record

Verified on October 1, 2026 in the shared Linux workspace. The original OpenClaw source package
was preserved; all adaptation work is in a separate `opencode-ml-contracts` directory.

## Results

| Check | Result |
|---|---|
| Ruff over `src` and `tests` | Pass |
| Mypy strict over all 22 Python source files | Pass |
| Python unit/integration tests | 33 passed |
| TypeScript strict typecheck | Pass |
| Node/TypeScript adapter tests | 4 passed |
| Actual OpenCode 1.18.34 runtime smoke test | Pass |
| Actual OpenCode resolved configuration | Accepted |
| Agent Skills format validation | All 3 skills valid |
| Editable Python wheel build/install in a separate virtual environment | Pass |
| Canonical JSON schema export | All 5 schemas exported |

## What executed

- Real EDA on the bundled synthetic churn dataset, including temporal drift.
- Real scikit-learn classification and regression training with five-fold cross-validation,
  holdout metrics, artifact generation, and full contract validation.
- Real Ruff, Mypy, AST/Radon diagnostics and Pytest against a fixed mock refactor.
- Real Node-to-Python subprocess calls, including invalid-path and invalid-argument rejection.
- Rejection of unknown tasks, missing inputs, incorrect types, extraneous fields, task-inappropriate
  fields, path escapes, outside-workspace symlinks, and unauthorized execution.
- JSON-mode repair and exhausted-repair behavior against a fake completion client.
- Verification that generated Pytest processes do not inherit the test API-key environment variables.
- Actual OpenCode CLI execution with a loopback mock OpenAI-compatible model server:
  the runtime registered `ml_contract`, called it, and returned measured EDA data to the mock model.

The runtime smoke test made three requests to the local mock server. It used only a dummy
credential in the child process and did not call DeepSeek inference.

## Not verified

- Real DeepSeek Flash inference, latency, schema adherence, or provider account permissions.
- DeepSeek beta strict tool decoding; the adapted default is JSON mode plus Python validation.
- Native Windows execution; the interpreter lookup and `os.devnull` paths are portable, but runtime
  verification was on Linux.
- Security-grade containment of adversarial generated code. The execution gate, working-directory
  separation, environment scrubbing, and static checks are not a sandbox.
- Bit-for-bit reproducibility across arbitrary Python dependency versions or platforms.

The tested npm plugin/runtime version is pinned to 1.18.34. The Python dependency minimums are
not a complete environment lock. The test host used Python 3.14 and Node 20.20.1; npm reported a
Node engine warning for a transitive dependency, so Node 22.22.2 or later is recommended for use.

## Acceptance boundary

Only a successful Python CLI/tool result is an accepted structured artifact. JSON/schema validity
alone does not establish measured provenance, data quality, test independence, or model correctness.
The interactive assistant's final wording remains model-generated and is not the machine contract.
For automation, consume the CLI/tool JSON rather than the final chat text.

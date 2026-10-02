# TASK: Skills refactor + Flash/Pro model routing for the Video Surveillance Assistant (VVA)

## 0. Roles and model routing

You are the orchestrator, running on DeepSeek Flash. DeepSeek Pro is available as a
subagent (`vva-pro`) for hard work. Routing depends on the TYPE of work:

| Work                                                                   | Model        |
|------------------------------------------------------------------------|--------------|
| Skill files, docs, translations, consistency tests, formatting         | Flash (you)  |
| B1: design of complex analytics layers (YOLO excluded)                 | Pro          |
| B2: debugging                                                          | Pro          |
| B3: optimization                                                       | Pro          |

Escalate to Pro when ANY of these holds:
1. The task is in scope B1, B2 or B3 (section 5).
2. The same failure persists after two of your own attempts.
3. The change touches numerical or algorithmic code under `src/vva_contracts/core/`.
4. The task requires profiling or reasoning about performance.
Never escalate: skill files, docs, test scaffolding, formatting, mechanical renames.
Reason: Flash follows patterns well and is cheap; Pro's extra cost is only justified where
a wrong idea is expensive (algorithms, root causes, performance).

Handoff protocol. Each call to `vva-pro` carries a bounded brief:
goal, exact file paths, constraints (section 7), acceptance commands, and what NOT to touch.
Pro returns a patch plus its reasoning and the evidence it claims.

Verification rule. Pro PROPOSES; you VERIFY. Run every acceptance command yourself.
Never relay "tests pass" from Pro's text. Pro's work is accepted only through: pytest,
`ruff check`, `ruff format --check`, `mypy --strict src`, and, when the `ml_contract` tool is
available, an offline CODE_REVIEW of changed Python files as an independent check.

Separation of duties. The certifier agent `vva-contracts` stays read-only: no `edit`, no
`bash`, no `task` permission. Whoever generates code is not whoever accepts it.

Model identifiers. Verify exact DeepSeek model ids in https://api-docs.deepseek.com
before writing any config. Do not guess a Pro id: if you cannot confirm it, write the
placeholder `DEEPSEEK_PRO_MODEL_ID` and list it in the report. API keys come from the
environment (`{env:DEEPSEEK_API_KEY}`), never from files.

Privacy toward the provider. Everything you send to Flash or Pro leaves the machine.
Never include frames, clips, RTSP URLs, credentials, `.env` content or REAL detection logs:
timestamps of detections reveal when the house is occupied. Use synthetic fixtures
(`tests/builders.py`) or aggregated numbers.

## 1. Context

Repo `vva-agent`. Python package `vva_contracts`: deterministic, auditable measurement
tasks for a private surveillance system (Raspberry Pi 5 + Hailo-10H, RTSP cameras). No LLM
runs inside the package. Dispatch is a registry lookup (`src/vva_contracts/registry.py`).

You authored the first package (`opencode-ml-contracts`: `mlclaw`, `ml_contract`, tabular
ML skills). You have NOT seen `vva_contracts`, which was written separately: do not assume
it mirrors `mlclaw` internals. Reuse what you know about `ml_contract` conventions (bridge,
exit codes, operator gates) only where the code confirms it. A copy of the first package's
top-level files is in `context/opencode-ml-contracts/` (its `src/mlclaw` and `.opencode/` are
NOT in this repo: they live in your own workspace).

The agent that will USE the skills is `vva-contracts` on Flash, with permissions
`"*": "deny"`, `read`, `glob`, `grep`, `skill` = allow, `vva_contract`/`ml_contract` = ask.
It cannot run shell or edit files. A skill is a precise procedure for a small, constrained
model: short, unambiguous, never contradicting those permissions or AGENTS.md.

Project rules that everything must protect:
- No LLM decides whether an alert fires. LLMs are for development and post-event enrichment.
- Evidence is measured, never claimed. No estimating, rounding or recomputing numbers.
- Edge-first privacy: frames, clips and RTSP URLs never enter the chat. Only the NAME of an
  env var (pattern `^VVA_[A-Z0-9_]+_URL$`) is ever used.

## 2. Problem to fix (Workstream A)

The current skills (`ml-inspect-data`, `ml-train-baseline`, `ml-validate-code`; copies in
`context/openclaw-skills/`) are OpenClaw originals and contradict the rest of the package:
1. They order `exec` and `write`, which the agent is denied.
2. They demand JSON "in ONE ```json block"; AGENTS.md says "no markdown fences or prose".
3. They reference `$HOME/.openclaw/...`, "router context", "OpenClaw sandbox", and a
   `metadata.openclaw` block requiring `DEEPSEEK_API_KEY` (not needed offline).
4. They cover tabular ML, not video. VVA tasks: DATASET_AUDIT, DETECTION_EVAL, QUANT_PARITY,
   RULE_REPLAY, STREAM_PROBE, PIPELINE_BENCH.

## 3. Sources of truth (read BEFORE writing anything)

1. `src/vva_contracts/contracts/requests.py`: exact request fields, defaults, bounds.
2. `src/vva_contracts/contracts/{dataset_audit,detection,rule_replay,stream,logs}.py`:
   reports, flags, invariants, log-row formats (PredictionRow, FrameRow, EventRow,
   TimingRow, SystemRow).
3. `src/vva_contracts/errors.py`: exit codes (0, 2, 3, 4, 6; 1 = internal).
4. `src/vva_contracts/registry.py` and `src/vva_contracts/tasks/*.py`: real behaviour.
5. `src/vva_contracts/core/*.py`: shared algorithms (AP, IoU, state machine, phash).
6. `context/Video_vigilance_assistant.md` (project state and roadmap),
   `context/opencode-ml-contracts/{AGENTS.md,opencode.json}`, `context/openclaw-skills/*`.
7. OpenCode docs, verify instead of recalling: https://opencode.ai/docs/skills/ (frontmatter:
   `name` 1-64 chars, lowercase alphanumerics and single hyphens, equal to its directory;
   `description` 1-1024 chars) and https://opencode.ai/docs/agents/ (per-agent `model`,
   `mode: subagent`, `permission`, including delegation permissions).

Never write a field, default, flag or exit code from memory: copy it from the code.
If a fact is not in the code, do NOT invent it: list it in D4.

## 4. Workstream A (Flash): skills refactor

D1. Skills under `.opencode/skills/<name>/SKILL.md`:
    `vva-audit-dataset` (DATASET_AUDIT), `vva-eval-detection` (DETECTION_EVAL),
    `vva-check-quantization` (QUANT_PARITY), `vva-replay-rules` (RULE_REPLAY),
    `vva-probe-stream` (STREAM_PROBE), `vva-bench-pipeline` (PIPELINE_BENCH),
    `ml-validate-code` (CODE_REVIEW via the existing `ml_contract` tool: adapt, do not
    rewrite its logic; default `offline=true`; live refactor needs explicit user consent).
    Move the copies of `ml-inspect-data` and `ml-train-baseline` to `archive/openclaw-skills/`
    untouched (they are not ported: they cover tabular ML).
D2. `docs/skills.md` in SPANISH: per skill, the problem it solves, WHY its rules exist (the
    relevant math in one or two lines) and a worked example.
D3. `tests/test_skills_consistency.py` (section 6).
D4. `docs/skills-refactor-report.md` (SPANISH): discrepancies found, decisions taken, anything
    you could not verify.
D5. `docs/model-routing.md` (SPANISH): the routing table of section 0 with its rationale, plus
    a PROPOSED `opencode.json` snippet defining a `vva-builder` primary agent (Flash) and the
    `vva-pro` subagent (Pro). Least privilege: `edit` and `bash` set to "ask" for both. Do NOT
    overwrite any existing `opencode.json`; propose, do not apply. Generated code runs with
    your user's privileges and is not sandboxed: say so.

Language: skill files and test code in ENGLISH; D2, D4, D5 in SPANISH.

Template for every `vva-*` SKILL.md (max ~120 lines):
Frontmatter: `name`, `description` (what it does + when to use it; mentions the task name),
`compatibility: opencode`, `metadata` (string values; `task: <TASK>`). Nothing runtime-specific.
Body, in this order:
1. When to use / When NOT to use (name the neighbouring skill to use instead).
2. Preconditions: what to verify with read/glob/grep BEFORE calling. If one fails, ask ONE
   concise question.
3. Request: exact fields (required vs optional, defaults copied from code) and ONE complete
   example of the tool arguments in a ```json fence labelled illustrative.
4. Call: the `vva_contract` tool with that single object. Never a shell command.
5. Result handling, table exit code -> action: 0 return the tool's JSON exactly as received;
   2/3/4/6 return the typed error JSON unchanged, then ask for the ONE missing input.
6. Interpretation boundary: which report fields may be quoted (flags, verdict, counts);
   explicit ban on recomputing, rounding, averaging or "improving" any number.
7. Domain pitfalls: 3-5 bullets giving the reason behind each rule.
8. Output rule, verbatim: "The final answer is the raw JSON document only: no markdown
   fences, no prose." (The fenced example in step 3 is illustrative only.)

Per-skill specifics (must appear):
- vva-audit-dataset: split by recording (clip/camera/day), never by frame: near-duplicate
  consecutive frames inflate validation mAP. `group_pattern` needs a named group `(?P<group>...)`.
  A FAIL verdict is a valid report, not a tool error: relay it. Run it BEFORE vva-eval-detection
  on any dataset not yet audited.
- vva-eval-detection: predictions are PredictionRow JSONL (state the row format) for the same
  split; any label error aborts (exit 3: run the audit). Never say "the detector works" without
  citing `map50`/`map50_95` and operating-point precision/recall.
- vva-check-quantization: reference (e.g. FP32 ONNX) and candidate (INT8 HEF) scored on the same
  split and operating confidence; the verdict follows `failed_checks`.
- vva-replay-rules: frame logs MUST include zero-detection frames (they break streaks). State
  the known risk: a false-positive alert opens the per-camera cooldown and can suppress a real
  event right after it. Tell the user to compare policies on the same log, never to choose
  thresholds "on paper" (per-frame false positives are correlated, so p^k understates real
  alert rates). Do NOT claim the cooldown design is settled: it is an open project decision.
- vva-probe-stream: ask for the env var NAME only, never a URL or credentials; substream
  (inference) and main stream (evidence) are probed separately; relay `declared_fps_unknown`
  and `timestamp_gaps` as-is.
- vva-bench-pipeline: explain warm-up exclusion and that the budget applies to end-to-end p95,
  not the mean; Pi-side logs are produced outside OpenCode, so tell the user which log
  (TimingRow / SystemRow) is missing instead of creating it.
- ml-validate-code: keep "verdict is computed from evidence"; remove every OpenClaw reference;
  keep the warning that generated code is not sandboxed.

## 5. Workstream B (Pro): complex analytics, debugging, optimization

Shared rules for all of B: brief Pro per section 0; code follows the standards of the repo
(docstring per public function stating the contract, comments explaining WHY, ruff + pydocstyle,
mypy --strict, McCabe <= 10, English); every behaviour change lands as its own commit with a
message explaining why, so the user can review line by line.

B1. Complex analytics layers (YOLO excluded: it is well documented and out of scope).
    Interpretation: layers that sit above raw detection, as listed in
    `context/Video_vigilance_assistant.md`. Candidates to evaluate, NOT a commitment: object
    tracking and association, zone/time analytics over event logs, alert-threshold optimisation
    from replays, analysis of correlated false positives (e.g. spatial clustering of recurring
    false boxes), and the boundary of post-event LLM/VLM enrichment. Derive your own list from
    the project docs; drop what the docs do not support.
    THIS RUN PRODUCES DESIGN ONLY: `docs/analytics-layers.md` (SPANISH). Per layer: the problem
    it solves, input/output contract (a Pydantic report with recomputed invariants, same pattern
    as existing tasks), the math, how it is measured against a synthetic case with a known
    analytical answer, failure modes, and what it must NOT do (no LLM in the alert path).
    Implement code ONLY for layers listed here; if the list is `none`, implement nothing:
    APPROVED_LAYERS: none

B2. Debugging. Protocol: (1) reproduce with a failing test, (2) find the root cause with
    evidence (hypothesis, then check), (3) fix at the cause, never at the symptom, (4) add a
    regression test. Never edit a test to make it pass. If the failure is a disagreement about
    intended behaviour instead of a bug, stop and escalate to the user. Known case:
    `test_rule_replay_tradeoff` fails because a false-positive alert opens the cooldown and
    suppresses a real event. That is an open design decision (options: keep per-camera cooldown,
    reset it only on confirmed alerts, or add a "suppressed events" metric to RULE_REPLAY).
    Do NOT change cooldown semantics: document the options with their trade-offs in D4.

B3. Optimization. Protocol: (1) baseline benchmark first: deterministic input, several runs,
    report median and spread plus hardware; (2) profile (cProfile) and optimise only the proven
    hotspot, one change at a time; (3) prove equivalence: outputs identical or inside a stated
    tolerance. Existing oracles: `tests/test_ap_crosscheck.py` (AP vs pycocotools) and the banded
    near-duplicate search vs brute force in `tests/test_core.py`; (4) report before/after numbers.
    Readability is a requirement: do not trade reviewability for speed. Pi/Hailo performance
    cannot be verified here: label it UNVERIFIED and provide the command for the user to run
    PIPELINE_BENCH on the Pi.

## 6. Consistency test (tests/test_skills_consistency.py), all must pass

1. Every `.opencode/skills/*/SKILL.md` has valid frontmatter (name == directory name, regex,
   description length) and body <= 150 lines.
2. Every task in `registry.TASKS` has exactly one `vva-*` skill whose `metadata.task` matches;
   no skill references a task absent from the registry.
3. Backticked tool names in skill bodies are a subset of ALLOWED_TOOLS =
   {read, glob, grep, skill, vva_contract, ml_contract}. If `opencode.json` defines the
   `vva-contracts` agent, read the set from its permissions instead.
4. Forbidden strings in any skill: `exec`, `bash`, `shell command`, `openclaw`, `$HOME`,
   `router`, `classifier`, `DEEPSEEK`; in `vva-*` skills also `offline`.
5. Every ```json block containing a `"task"` key validates against `REQUEST_ADAPTER`
   (proves examples use real fields and defaults).
6. Every `Schema:` / file path referenced exists after `vva schemas --output schemas`
   (the `schemas/` folder is already generated in this repo).
7. Every skill contains the verbatim output-rule sentence (the non-vva skill keeps its own
   equivalent).
8. If `opencode.json` defines `vva-contracts`: it has no `edit`, `bash` or `task` permission
   set to allow or ask (certifier stays read-only).

Also run and report: `ruff check`, `ruff format --check`, `mypy --strict src`, `pytest -q`.
Do not skip or weaken existing tests.

## 7. Hard constraints

- Workstream A does not modify `src/vva_contracts/`. Only Workstream B may, and only within
  the scope it states (B2 with a regression test, B3 with equivalence proof, B1 only for
  APPROVED_LAYERS).
- Do not invent fields, defaults, thresholds, exit codes or model ids; copy or verify them.
- Skills never instruct the agent to use `exec`, `bash`, `write`, `edit`, or to create files.
- No secrets, real URLs or credentials anywhere, even as examples (use `VVA_FRONT_SUB_URL`).
- No LLM in the alert path; no confidence scores or intent classification requested from models.
- Do not add dependencies. Do not merge `vva_contracts` into the `mlclaw` namespace.
- Never present as verified anything you did not run.

## 8. Method

1. Read all sources. Write a short plan listing every contradiction between old skills,
   AGENTS.md and the code. Do not wait for input; continue.
2. Workstream A first: create skills ONE AT A TIME; run the consistency test after each.
3. Workstream B next, escalating to Pro only per section 0, verifying each result yourself.
4. Write D2, D4, D5 last, from what you actually verified.
5. Final message: files created/moved, real lint/type/test results, which tasks went to Pro
   and why, open questions.

## 9. Acceptance criteria

- All 8 consistency checks pass; ruff, mypy and pytest results reported honestly.
- A reviewer can trace each skill rule to code (field, flag, exit code) or to a stated reason.
- No skill contradicts the agent's permissions or the "raw JSON, no fences" output rule.
- Every Pro-originated change has a failing-then-passing test or an equivalence proof.
- B1 delivers a design document and no unapproved code.

"""Consistency checks between the OpenCode skills, the agent config and the code.

Why this test exists: a skill is a procedure for a small model. If it names a field,
default, tool or schema that the code does not have, the model will follow it and
fail at runtime. These checks make every such contradiction a test failure.

Scope:
* "certifier skills" = skills loaded by the read-only `vva-contracts` agent
  (`vva-*`). Checks 3-7 apply to them.
* "dev skills" (`dev-*`) are for the builder and Pro agents; they only need valid
  frontmatter (check 1) and must not carry a `task` in metadata.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

from vva_contracts.contracts.requests import REQUEST_ADAPTER
from vva_contracts.registry import TASKS

ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = ROOT / ".opencode" / "skills"
SCHEMAS_DIR = ROOT / "schemas"
OPENCODE_JSON = ROOT / "opencode.json"

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_BODY_LINES = 150
OUTPUT_RULE = "The final answer is the raw JSON document only: no markdown fences, no prose."
DEFAULT_ALLOWED_TOOLS = {"read", "glob", "grep", "skill", "vva_contract"}
# Every tool name OpenCode or this project defines; only these are checked in check 3,
# so ordinary backticked words (field names, flags) are not mistaken for tools.
KNOWN_TOOLS = DEFAULT_ALLOWED_TOOLS | {
    "ml_contract",
    "bash",
    "edit",
    "write",
    "patch",
    "apply_patch",
    "list",
    "task",
    "webfetch",
    "websearch",
    "todowrite",
    "todoread",
    "lsp",
    "question",
}
FORBIDDEN_ALL = [
    "exec",
    "bash",
    "shell command",
    "openclaw",
    "$home",
    "router",
    "classifier",
    "deepseek",
]
FORBIDDEN_VVA = ["offline"]
JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL)
SCHEMA_REF_RE = re.compile(r"`((?:schemas|context)/[^`]+\.json)`")
BACKTICK_RE = re.compile(r"`([^`\n]+)`")


def _split(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path}: missing frontmatter"
    _, front, body = text.split("---\n", 2)
    data = yaml.safe_load(front)
    assert isinstance(data, dict), f"{path}: frontmatter is not a mapping"
    return data, body


SKILL_FILES = sorted(SKILLS_DIR.glob("*/SKILL.md"))
SKILLS = {p.parent.name: _split(p) for p in SKILL_FILES}
CERTIFIER = {n: s for n, s in SKILLS.items() if n.startswith("vva-")}


def _config() -> dict[str, object] | None:
    return json.loads(OPENCODE_JSON.read_text(encoding="utf-8")) if OPENCODE_JSON.exists() else None


def _certifier_permissions() -> dict[str, object] | None:
    cfg = _config()
    agents = cfg.get("agent", {}) if cfg else {}
    agent = agents.get("vva-contracts") if isinstance(agents, dict) else None
    return agent.get("permission") if isinstance(agent, dict) else None


def _allowed_tools() -> set[str]:
    perms = _certifier_permissions()
    if perms is None:
        return DEFAULT_ALLOWED_TOOLS
    return {k for k, v in perms.items() if k != "*" and (v in ("allow", "ask") or isinstance(v, dict))}


def test_skills_exist() -> None:
    assert SKILL_FILES, "no skills found under .opencode/skills"


@pytest.mark.parametrize("name", sorted(SKILLS))
def test_1_frontmatter_and_length(name: str) -> None:
    front, body = SKILLS[name]
    assert front.get("name") == name, "name must equal the directory name"
    assert NAME_RE.fullmatch(name) and len(name) <= 64
    desc = front.get("description")
    assert isinstance(desc, str) and 1 <= len(desc) <= 1024
    meta = front.get("metadata", {})
    assert isinstance(meta, dict) and all(isinstance(v, str) for v in meta.values()), "metadata: string values"
    assert len(body.splitlines()) <= MAX_BODY_LINES
    assert name.startswith(("vva-", "dev-")), "skills are vva-* (certifier) or dev-* (builder/pro)"
    if name.startswith("dev-"):
        assert "task" not in meta, "dev skills must not claim a vva task"


def test_2_registry_coverage() -> None:
    claimed: dict[str, list[str]] = {}
    for name, (front, _) in SKILLS.items():
        if name.startswith("vva-"):
            task = front.get("metadata", {}).get("task")  # type: ignore[union-attr]
            assert task in TASKS, f"{name}: metadata.task {task!r} not in registry"
            claimed.setdefault(task, []).append(name)
    for task in TASKS:
        assert len(claimed.get(task, [])) == 1, f"{task}: needs exactly one vva-* skill, got {claimed.get(task)}"


@pytest.mark.parametrize("name", sorted(CERTIFIER))
def test_3_tools_subset(name: str) -> None:
    _, body = CERTIFIER[name]
    named = {t for t in BACKTICK_RE.findall(body) if t in KNOWN_TOOLS}
    assert named <= _allowed_tools(), f"{name}: tools outside the certifier permissions: {named - _allowed_tools()}"


@pytest.mark.parametrize("name", sorted(CERTIFIER))
def test_4_forbidden_strings(name: str) -> None:
    front, body = CERTIFIER[name]
    text = (json.dumps(front) + body).lower()
    banned = FORBIDDEN_ALL + FORBIDDEN_VVA
    found = [w for w in banned if w in text]
    assert not found, f"{name}: forbidden strings {found}"


@pytest.mark.parametrize("name", sorted(CERTIFIER))
def test_5_examples_validate(name: str) -> None:
    _, body = CERTIFIER[name]
    blocks = [json.loads(b) for b in JSON_BLOCK_RE.findall(body)]
    with_task = [b for b in blocks if "task" in b]
    assert with_task, f"{name}: needs one illustrative example"
    for args in with_task:
        assert set(args) == {"task", "params"}, "tool arguments are {task, params}"
        assert "task" not in args["params"], "task must not be inside params"
        # validate_json, not validate_python: the bridge sends JSON on stdin and strict
        # mode accepts JSON arrays for tuples (zone points) only in JSON mode.
        request = REQUEST_ADAPTER.validate_json(json.dumps({"task": args["task"], **args["params"]}))
        assert request.task == args["task"]


@pytest.mark.parametrize("name", sorted(CERTIFIER))
def test_6_schema_paths_exist(name: str) -> None:
    _, body = CERTIFIER[name]
    for ref in SCHEMA_REF_RE.findall(body):
        assert (ROOT / ref).is_file(), f"{name}: referenced schema {ref} does not exist"


@pytest.mark.parametrize("name", sorted(CERTIFIER))
def test_7_output_rule(name: str) -> None:
    _, body = CERTIFIER[name]
    assert OUTPUT_RULE in body


def test_8_certifier_is_read_only() -> None:
    perms = _certifier_permissions()
    if perms is None:
        pytest.skip("opencode.json does not define vva-contracts")
    assert perms.get("*") == "deny", "certifier must default-deny"
    for key in ("edit", "bash", "task", "webfetch", "websearch"):
        assert perms.get(key, "deny") == "deny", f"certifier must not have {key}"

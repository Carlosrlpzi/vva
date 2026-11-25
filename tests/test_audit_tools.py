"""Regression tests for the auditor's client-side tools: the pre-push hook, the wrapper and the review script.

Why this test exists: the hook, ``install/vva-audit-run`` and ``install/vva-review-bundle`` are the
deterministic half of the A6 audit. Until now their behaviour was only checked by hand and by the
auditor (an LLM). These tests pin it down without root, without the ``vva-audit`` user and without
network access:

* the hook runs against synthetic repositories in ``tmp_path``; ``sudo`` is replaced on ``PATH`` by a
  fake that records the audited range and prints a canned report;
* the wrapper is *sourced* (its ``main`` only runs when executed) so its checkout and ``/etc`` checks
  can be called on temporary directories;
* the review script's location guard and protected-path detection run on a synthetic bundle.

Owner-only file: like ``install/`` and ``.githooks/``, it is drafted and committed by the owner, never
by the builder (``install/vva-review-bundle`` flags any change to it).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / ".githooks" / "pre-push"
WRAPPER = ROOT / "install" / "vva-audit-run"
REVIEW = ROOT / "install" / "vva-review-bundle"
ZERO = "0" * 40

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None,
    reason="needs git and bash",
)

FAKE_SUDO = """#!/bin/bash
cat > /dev/null
printf '%s\\n' "${@: -1}" >> "$FAKE_LOG"
cat "$FAKE_REPORT"
"""


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    out = subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True, text=True, env=env)
    return out.stdout.strip()


def _commit(repo: Path, name: str, text: str = "x\n") -> str:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


class HookEnv:
    """A work clone with an ``origin`` remote, a fake ``sudo`` and a canned audit report."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        seed = tmp_path / "seed"
        seed.mkdir()
        _git(seed, "init", "-q", "-b", "main")
        self.c1 = _commit(seed, "a.txt")
        self.c2 = _commit(seed, "b.txt")
        self.c3 = _commit(seed, "c.txt")
        remote = tmp_path / "remote.git"
        subprocess.run(["git", "clone", "-q", "--bare", str(seed), str(remote)], check=True)
        self.work = tmp_path / "work"
        subprocess.run(["git", "clone", "-q", str(remote), str(self.work)], check=True)
        _git(self.work, "remote", "set-head", "origin", "main")
        bindir = tmp_path / "bin"
        bindir.mkdir()
        sudo = bindir / "sudo"
        sudo.write_text(FAKE_SUDO, encoding="utf-8")
        sudo.chmod(0o755)
        self.log = tmp_path / "audited.log"
        self.report = tmp_path / "report.md"
        self.set_report("VERDICT: APPROVE")
        self.env = {
            **os.environ,
            "PATH": f"{bindir}:{os.environ['PATH']}",
            "FAKE_LOG": str(self.log),
            "FAKE_REPORT": str(self.report),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
        }
        self.env.pop("VVA_AUDIT", None)

    def set_report(self, text: str) -> None:
        self.report.write_text(text + "\n", encoding="utf-8")

    def push(self, *lines: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(HOOK), "origin", "x"],
            cwd=self.work,
            input="".join(line + "\n" for line in lines),
            capture_output=True,
            text=True,
            env=self.env,
            check=False,
        )

    def audited(self) -> list[str]:
        return self.log.read_text(encoding="utf-8").split() if self.log.exists() else []

    def branch(self, name: str, start: str, file: str) -> str:
        _git(self.work, "switch", "-q", "-c", name, start)
        sha = _commit(self.work, file)
        _git(self.work, "switch", "-q", "main")
        return sha


@pytest.fixture
def hook(tmp_path: Path) -> HookEnv:
    return HookEnv(tmp_path)


# ------------------------------------------------------------------------- pre-push hook
def test_fast_forward_audits_remote_to_local(hook: HookEnv) -> None:
    result = hook.push(f"refs/heads/main {hook.c3} refs/heads/main {hook.c1}")
    assert result.returncode == 0, result.stderr
    assert hook.audited() == [f"{hook.c1}..{hook.c3}"]


@pytest.mark.parametrize(
    ("report", "code"),
    [
        ("VERDICT: APPROVE", 0),
        ("[MEDIUM] x\nVERDICT: APPROVE_WITH_CHANGES", 0),
        ("[LOW] Grouped\nVERDICT: APPROVE_WITH_CHANGES", 0),
        ("[HIGH] x\nVERDICT: APPROVE_WITH_CHANGES", 1),
        ("[CRITICAL] x\nVERDICT: APPROVE", 1),
        ("VERDICT: REQUEST_CHANGES", 1),
        ("no verdict here", 1),
        ("VERDICT: APPROVE\ntrailing text", 1),
    ],
)
def test_verdict_and_blocking_severities(hook: HookEnv, report: str, code: int) -> None:
    hook.set_report(report)
    result = hook.push(f"refs/heads/main {hook.c3} refs/heads/main {hook.c1}")
    assert result.returncode == code, result.stderr


@pytest.mark.parametrize(
    "line",
    [
        "**[HIGH]** bold",
        "### [HIGH] heading",
        "- [CRITICAL] list item",
        "  [HIGH] indented",
        "> [HIGH] quote",
        "`[HIGH]` x",
    ],
)
def test_markdown_wrapped_blocking_severities_still_block(hook: HookEnv, line: str) -> None:
    hook.set_report(f"{line}\nVERDICT: APPROVE_WITH_CHANGES")
    result = hook.push(f"refs/heads/main {hook.c3} refs/heads/main {hook.c1}")
    assert result.returncode == 1, result.stderr
    assert "CRITICAL/HIGH" in result.stderr


def test_a_severity_mentioned_mid_sentence_does_not_block(hook: HookEnv) -> None:
    hook.set_report("Summary mentions [HIGH] only in passing.\n**[MEDIUM]** x\nVERDICT: APPROVE_WITH_CHANGES")
    result = hook.push(f"refs/heads/main {hook.c3} refs/heads/main {hook.c1}")
    assert result.returncode == 0, result.stderr


def test_non_fast_forward_to_default_branch_is_blocked_without_auditing(hook: HookEnv) -> None:
    other = hook.branch("other", hook.c1, "z.txt")
    result = hook.push(f"refs/heads/other {other} refs/heads/main {hook.c3}")
    assert result.returncode == 1
    assert "default branch" in result.stderr
    assert hook.audited() == []


def test_rewind_of_default_branch_is_blocked(hook: HookEnv) -> None:
    _git(hook.work, "branch", "old", hook.c1)
    result = hook.push(f"refs/heads/old {hook.c1} refs/heads/main {hook.c3}")
    assert result.returncode == 1
    assert hook.audited() == []


def test_force_push_to_other_branch_audits_what_it_adds(hook: HookEnv) -> None:
    other = hook.branch("other", hook.c1, "z.txt")
    result = hook.push(f"refs/heads/other {other} refs/heads/topic {hook.c3}")
    assert result.returncode == 0, result.stderr
    assert "not audited" in result.stderr
    assert hook.audited() == [f"{hook.c1}..{other}"]


def test_rewind_of_other_branch_has_nothing_to_audit(hook: HookEnv) -> None:
    _git(hook.work, "branch", "old", hook.c1)
    result = hook.push(f"refs/heads/old {hook.c1} refs/heads/topic {hook.c3}")
    assert result.returncode == 0, result.stderr
    assert "no new commits" in result.stderr
    assert hook.audited() == []


def test_annotated_tag_on_new_commit_is_peeled(hook: HookEnv) -> None:
    new = hook.branch("feat", hook.c3, "n.txt")
    _git(hook.work, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", "v1", "-m", "v1", new)
    tag_object = _git(hook.work, "rev-parse", "v1")
    assert tag_object != new
    result = hook.push(f"refs/tags/v1 {tag_object} refs/tags/v1 {ZERO}")
    assert result.returncode == 0, result.stderr
    assert hook.audited() == [f"{hook.c3}..{new}"]


def test_ref_that_does_not_match_its_sha_is_blocked(hook: HookEnv) -> None:
    result = hook.push(f"refs/heads/main {hook.c2} refs/heads/main {hook.c1}")
    assert result.returncode == 1
    assert hook.audited() == []


def test_missing_origin_head_is_blocked(hook: HookEnv) -> None:
    _git(hook.work, "remote", "set-head", "origin", "-d")
    result = hook.push(f"refs/heads/main {hook.c3} refs/heads/main {hook.c1}")
    assert result.returncode == 1
    assert "origin's default branch" in result.stderr


# ------------------------------------------------------------------------------- wrapper
def _source_wrapper(script: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'source "{WRAPPER}"; {script}'],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _nested_repo(tmp_path: Path) -> tuple[Path, str]:
    src = tmp_path / "src"
    src.mkdir()
    _git(src, "init", "-q", "-b", "main")
    for name in (
        "AGENTS.md",
        "docs/x/AGENTS.md",
        "docs/CLAUDE.md",
        "docs/CONTEXT.md",
        "b/opencode.jsonc",
        "install/vva-audit/opencode/opencode.json",
        "a/.opencode/agents/evil.md",
        "a/.claude/s.md",
        "c/.agents/k.md",
        "prompts/root.md",
        "src/app/prompts/vlm.md",
        "install/vva-audit/opencode/prompts/vva-audit.md",
        "keep.py",
        "docs/AGENTS.md.txt",
    ):
        path = src / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    _git(src, "add", "-A")
    _git(src, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "seed")
    tip = _git(src, "rev-parse", "HEAD")
    bundle = tmp_path / "in.bundle"
    _git(src, "bundle", "create", "-q", str(bundle), "main")
    return bundle, tip


def test_wrapper_checkout_drops_agent_config_at_any_depth_and_prompts_only_at_root(tmp_path: Path) -> None:
    bundle, tip = _nested_repo(tmp_path)
    repo = tmp_path / "repo"
    result = _source_wrapper(f'clone_checkout "{bundle}" "{repo}" "{tip}" "{tip}"', tmp_path)
    assert result.returncode == 0, result.stderr
    files = sorted(str(p.relative_to(repo)) for p in repo.rglob("*") if p.is_file() and ".git" not in p.parts)
    assert files == [
        "docs/AGENTS.md.txt",
        "install/vva-audit/opencode/prompts/vva-audit.md",
        "keep.py",
        "src/app/prompts/vlm.md",
    ]


def test_wrapper_rejects_an_excluded_name_left_in_the_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "z" / ".claude").mkdir(parents=True)
    result = _source_wrapper(f'check_no_excluded "{checkout}"', tmp_path)
    assert result.returncode == 11
    assert "z/.claude" in result.stderr


def test_wrapper_rejects_a_root_prompts_dir_left_in_the_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "prompts").mkdir(parents=True)
    result = _source_wrapper(f'check_no_excluded "{checkout}"', tmp_path)
    assert result.returncode == 11


def test_wrapper_rejects_a_commit_missing_from_the_bundle(tmp_path: Path) -> None:
    bundle, tip = _nested_repo(tmp_path)
    result = _source_wrapper(f'clone_checkout "{bundle}" "{tmp_path / "repo"}" "{"1" * 40}" "{tip}"', tmp_path)
    assert result.returncode == 10


@pytest.mark.skipif(os.geteuid() == 0, reason="as root every file is root-owned")
def test_wrapper_etc_tree_check_rejects_files_not_owned_by_root(tmp_path: Path) -> None:
    (tmp_path / "opencode").mkdir()
    (tmp_path / "opencode" / "AGENTS.md").write_text("x\n", encoding="utf-8")
    result = _source_wrapper(f'check_root_tree 8 "{tmp_path / "opencode"}"', tmp_path)
    assert result.returncode == 8


@pytest.mark.parametrize("bad", ["bad", f"{'a' * 40}..{'b' * 39}", f"{'A' * 40}..{'b' * 40}", ""])
def test_wrapper_rejects_malformed_ranges(tmp_path: Path, bad: str) -> None:
    result = _source_wrapper(f'check_range "{bad}"', tmp_path)
    assert result.returncode == 2


# ------------------------------------------------------------------------- review script
def _review(cwd: Path, home: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
    return subprocess.run(["bash", str(REVIEW), *args], cwd=cwd, capture_output=True, text=True, env=env, check=False)


def test_review_refuses_to_run_outside_home_before_running_git(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    outside = tmp_path / "taller"
    outside.mkdir()
    (outside / "x.bundle").write_text("not a bundle\n", encoding="utf-8")
    result = _review(outside, home, "x.bundle")
    assert result.returncode == 3
    assert "trusted repository" in result.stderr


class ReviewEnv:
    """A trusted repository under a fake HOME and a taller clone that produces bundles."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.trusted = self.home / "vva"
        self.trusted.mkdir(parents=True)
        _git(self.trusted, "init", "-q", "-b", "main")
        _commit(self.trusted, "keep.py")
        self.taller = tmp_path / "taller"
        subprocess.run(["git", "clone", "-q", str(self.trusted), str(self.taller)], check=True)

    def bundle(self, branch: str, *files: str) -> Path:
        _git(self.taller, "switch", "-q", "-c", branch, "main")
        for name in files:
            _commit(self.taller, name)
        path = self.tmp / f"{branch}.bundle"
        _git(self.taller, "bundle", "create", "-q", str(path), f"main..{branch}")
        return path

    def review(self, bundle: Path, wrapper: Path = WRAPPER) -> subprocess.CompletedProcess[str]:
        # Sourced, so the wrapper under test is this commit's install/vva-audit-run (or a
        # synthetic one), not whatever happens to be installed in /usr/local/bin.
        env = {**os.environ, "HOME": str(self.home), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"}
        return subprocess.run(
            ["bash", "-c", f'source "{REVIEW}"; review "{wrapper}" "{bundle}"'],
            cwd=self.trusted,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


@pytest.fixture
def rev(tmp_path: Path) -> ReviewEnv:
    return ReviewEnv(tmp_path)


def test_review_passes_a_clean_branch(rev: ReviewEnv) -> None:
    result = rev.review(rev.bundle("clean", "src/ok.py", "src/app/prompts/vlm.md"))
    assert result.returncode == 0, result.stderr
    assert "ok: no protected path changed" in result.stdout


@pytest.mark.parametrize(
    "path",
    [
        "src/app/AGENTS.md",
        "AGENTS.md",
        "opencode.json",
        "x/opencode.json",
        "b/opencode.jsonc",
        "x/.opencode/agents/evil.md",
        "docs/CLAUDE.md",
        "prompts/p.md",
        "install/vva-audit-run",
        ".githooks/pre-push",
        "docs/guia-opencode-ubuntu.md",
        "docs/security-decisions.md",
        "tests/test_audit_tools.py",
    ],
)
def test_review_flags_protected_paths(rev: ReviewEnv, path: str) -> None:
    result = rev.review(rev.bundle("dirty", "src/ok.py", path))
    assert result.returncode == 1, result.stderr
    assert path in result.stdout


def test_review_reports_refs_that_are_not_branches(rev: ReviewEnv) -> None:
    _git(rev.taller, "switch", "-q", "-c", "work", "main")
    _commit(rev.taller, "src/ok.py")
    _git(rev.taller, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", "v1", "-m", "v1")
    bundle = rev.tmp / "all.bundle"
    _git(rev.taller, "bundle", "create", "-q", str(bundle), "main..work", "refs/tags/v1")
    result = rev.review(bundle)
    assert result.returncode == 1, result.stderr
    assert "refs/tags/v1" in result.stdout


def test_review_rejects_a_bundle_without_branches(rev: ReviewEnv) -> None:
    _git(rev.taller, "-c", "user.name=t", "-c", "user.email=t@t", "tag", "-a", "v1", "-m", "v1")
    bundle = rev.tmp / "tags.bundle"
    _git(rev.taller, "bundle", "create", "-q", str(bundle), "refs/tags/v1")
    assert rev.review(bundle).returncode == 4


@pytest.mark.parametrize(
    "wrapper_text",
    [
        None,  # no wrapper at all
        "readonly EXCLUDED_ROOT_NAMES=(prompts)\n",  # no EXCLUDED_NAMES
        "readonly EXCLUDED_NAMES=(CLAUDE.md)\nreadonly EXCLUDED_ROOT_NAMES=(prompts)\n",  # AGENTS.md missing
        "readonly EXCLUDED_NAMES=(AGENTS.md opencode.json .opencode)\n",  # no EXCLUDED_ROOT_NAMES
        "readonly EXCLUDED_NAMES=(AGENTS.md opencode.json .opencode *)\nreadonly EXCLUDED_ROOT_NAMES=(prompts)\n",
    ],
)
def test_review_fails_closed_when_the_wrapper_lists_cannot_be_read(rev: ReviewEnv, wrapper_text: str | None) -> None:
    wrapper = rev.tmp / "fake-vva-audit-run"
    if wrapper_text is not None:
        wrapper.write_text("#!/bin/bash\n" + wrapper_text, encoding="utf-8")
    result = rev.review(rev.bundle("dirty", "src/app/AGENTS.md"), wrapper=wrapper)
    assert result.returncode == 5, result.stdout + result.stderr
    assert "ok: no protected path changed" not in result.stdout

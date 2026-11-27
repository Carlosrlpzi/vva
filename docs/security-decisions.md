# Security decisions

Accepted risks and security-relevant decisions that are not visible in the code. Each entry
records what happened, what was decided, who decided it, and what would reopen it. Never
copy a secret, key, token or credential into this file; refer to it by commit and path.

## SD-1 WireGuard private key in public git history (accepted risk)

- **Status:** accepted, documented. History is not rewritten.
- **Decided by:** the repository owner, 2026-11-25 (audit finding F5 on `ded76ac..84b830f`).
- **What happened:** commit `4ce758c` (2026-10-02) added a pasted WireGuard terminal session,
  including a private key, to `docs/01-setup-guide.md`. Commit `3924eed` (2026-11-24) removed
  it from the working tree, but the repository is public and the key remains readable in the
  history of that file.
- **Why it is accepted:** the key belonged to a Raspberry Pi 5 that no longer exists. With
  the device gone, the key no longer authenticates anything. Rewriting public history would
  force a re-clone of every copy and would not remove copies already fetched by others.
- **Conditions for this to hold:** no live WireGuard peer may still list the matching public
  key in an `[Peer]` section. If any server, router or VPS that was this Pi's peer still
  exists, remove that peer entry there.
- **Reopen if:** the same key pair is found configured on any live device, or the key is
  reused anywhere.
- **Prevention:** the pre-push audit (A6) reviews every pushed range. Pasted terminal
  sessions and configuration snippets in `docs/` must be scrubbed of keys and credentials
  before commit (AGENTS.md invariant 3).

## SD-2 The pre-push audit is a client-side control (known limitation)

- **Status:** known limitation; server-side protection pending.
- **Decided by:** the repository owner, 2026-11-25 (audit findings on `0a57365..dd10361`).
- **What it means:** the A6 hook runs only in clones where it is installed and can be skipped
  with `git push --no-verify` or `VVA_AUDIT=0`. It blocks non-fast-forward pushes to the
  default branch, but it does not block deleting a remote branch.
- **Mitigation (planned, not yet active as of 2026-11-25):** GitHub branch protection on
  `main` ("Do not allow force pushes", "Do not allow deletions"), to be configured by the
  owner in the repository settings. GitHub already refuses to delete the default branch.
- **Merge rule enforced by the hook (2026-11-25):** the line right before the `VERDICT` must be
  `FINDINGS: critical=N high=N medium=N low=N`; the push is blocked if it is missing or
  malformed, or if `critical` or `high` is above 0, even under `APPROVE_WITH_CHANGES`. A
  regex over severity-tagged lines (markdown, case and Spanish variants) is a second net.
- **Deterministic review:** `vva-review-bundle` (root-owned in `/usr/local/bin`) flags any
  owner-only, agent-config or git-metadata path a builder bundle changes, using the installed
  wrapper's `EXCLUDED_NAMES` and `EXCLUDED_ROOT_NAMES`, so the LLM auditor is not the only
  control. It reads a private copy of the bundle (bounded in time and size, so a FIFO swapped into the
  mailbox cannot hang the review or fill the disk), fetches only the listed branches (plain
  ASCII names) with `--no-tags` and `fetch.fsckObjects`, and pins the base to a SHA first.
- **Tests:** `tests/test_audit_tools.py` covers the hook, the wrapper's checkout and `/etc`
  checks, and the review script, with temporary repositories and a fake `sudo` (no root).
  It is owner-only, like `install/` and `.githooks/`. It runs in the taller as
  `opencode-dev`, so the pytest configuration (`conftest.py` at any depth, `pyproject.toml`,
  `pytest.ini`, `tox.ini`, `setup.cfg`) is owner-only too and `vva-review-bundle` flags it.
  Limit: the builder still controls the taller's whole environment (`.venv`, site-packages),
  so a green run there is evidence, not proof; the proof for a change to these tools is the
  owner's own review of the diff.
- **Pending:** GitHub branch protection on `main` (owner, repository settings).

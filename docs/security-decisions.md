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
- **Mitigation:** GitHub branch protection on `main` ("Do not allow force pushes", "Do not
  allow deletions"), configured by the owner in the repository settings. GitHub also refuses
  to delete the default branch.
- **Pending:** automated tests for `.githooks/pre-push` and `install/vva-audit-run`
  (temporary repositories, a fake `sudo`, no root), written outside the builder because the
  builder never touches the auditor.

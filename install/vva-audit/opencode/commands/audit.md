---
description: Read-only audit of current changes by vva-audit (Claude Opus)
agent: vva-audit
---
Audit the changes in this repository.

Scope: $ARGUMENTS
If the scope above is empty, audit uncommitted changes (`git diff HEAD`) plus any commits
not yet pushed to the upstream branch. If it is a git range (e.g. `abc123..def456`),
run `git diff <range>` and `git log <range>` yourself and audit exactly that range.

Repository state at the moment this audit was requested:

## git status
!`git status --short`

## Recent commits
!`git log --oneline -n 15`

Follow your workflow and output format. End with the VERDICT line.

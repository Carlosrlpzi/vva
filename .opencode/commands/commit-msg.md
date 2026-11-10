---
description: Propose a commit message for the staged changes (does not commit)
agent: vva-builder
---
Propose a commit message for the staged changes. Do NOT run git commit or any
command that modifies the repository.

Staged files:
!`git diff --staged --stat`

Recent commits (match their language and style):
!`git log --oneline -n 10`

Steps:
1. Run `git diff --staged` and read it. If nothing is staged, say so and stop.
2. Describe what the diff actually does, not what it was meant to do. If the
   staged changes mix unrelated concerns, say so and suggest splitting them.
3. Output exactly one message in a code block:
   - Subject: Conventional Commits `type(scope): summary`, imperative mood,
     max 72 characters, no trailing period. Types: feat, fix, refactor, perf,
     test, docs, chore.
   - Body only if the why is not obvious, wrapped at 72 columns: explain why,
     not a list of files.
4. After the block, one line with any doubt (e.g. why chore and not feat).

Extra context from the user: $ARGUMENTS


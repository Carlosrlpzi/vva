"""Content fingerprints used to prove which inputs produced a report.

Why this module exists
----------------------
A report that only passes schema validation proves *shape*, not *provenance*.
Every report therefore lists the SHA-256 of each input. ``vva validate
--workspace`` recomputes those hashes, so a report whose inputs changed (or
were never there) is rejected instead of being trusted.

For a directory the fingerprint is the SHA-256 of a sorted manifest of
``relative_path<TAB>file_sha256`` lines. Sorting makes it independent of
filesystem listing order, which differs between ext4 (Pi) and other systems.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# 1 MiB chunks: large enough to be fast, small enough to keep memory flat when
# hashing multi-GB evidence clips on a Raspberry Pi.
_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes, streamed in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Fingerprint a file, or a directory through its sorted manifest."""
    if path.is_file():
        return sha256_file(path)
    manifest = hashlib.sha256()
    # Hidden files (e.g. .DS_Store, editor swap files) are skipped: they are
    # not data, and including them would make the hash machine-dependent.
    # The test uses parts *relative to the root*: checking absolute parts would
    # wrongly skip everything when the project itself lives under a dot-folder.
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )
    for file in files:
        line = f"{file.relative_to(path).as_posix()}\t{sha256_file(file)}\n"
        manifest.update(line.encode("utf-8"))
    return manifest.hexdigest()

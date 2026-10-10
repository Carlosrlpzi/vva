"""Keep the versioned JSON Schemas in ``schemas/`` in sync with the Pydantic models.

Why this test exists
--------------------
The JSON Schemas are generated artifacts of the Pydantic models in
``src/vva_contracts/`` (the single source of truth), but six ``vva-*`` skills
reference ``schemas/*.json`` by path. If someone changes a contract model and
forgets to regenerate, the repository keeps serving stale schemas to the
certifier agent and nothing fails.

This test regenerates into a temporary directory with the *same* function the
CLI uses (:func:`vva_contracts.cli.export_schemas`) and fails on any byte
difference, so the drift is caught in CI rather than in a clean clone.

On failure, the fix is always::

    vva schemas --output schemas

and commit the result. Never hand-edit a JSON schema: the models are the only
source of truth.
"""

from __future__ import annotations

from pathlib import Path

from vva_contracts.cli import export_schemas

ROOT = Path(__file__).resolve().parents[1]
VERSIONED_DIR = ROOT / "schemas"
REGENERATE_HINT = "regenerate with 'vva schemas --output schemas', never edit the JSON by hand"


def test_versioned_schemas_match_models(tmp_path: Path) -> None:
    """Every versioned schema must be byte-identical to a fresh regeneration."""
    generated_names = set(export_schemas(tmp_path))
    versioned_names = {path.name for path in VERSIONED_DIR.glob("*.json")}

    missing = sorted(generated_names - versioned_names)
    extra = sorted(versioned_names - generated_names)
    assert not missing, f"schemas/ is missing {missing}; {REGENERATE_HINT}"
    assert not extra, f"schemas/ contains files the models no longer produce: {extra}; {REGENERATE_HINT}"

    stale = [
        name
        for name in sorted(generated_names)
        if (tmp_path / name).read_text(encoding="utf-8") != (VERSIONED_DIR / name).read_text(encoding="utf-8")
    ]
    assert not stale, f"schemas out of sync with the Pydantic models: {stale}; {REGENERATE_HINT}"

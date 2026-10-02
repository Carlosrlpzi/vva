"""Allows ``python -m vva_contracts ...`` (used by the OpenCode bridge)."""

from vva_contracts.cli import main

raise SystemExit(main())

"""Verifiable audit contracts for the Video Surveillance Assistant (VVA).

Every task reads files produced by the surveillance pipeline, computes its
metrics locally (no LLM call anywhere in this package) and returns a report
whose summary numbers and verdict are re-derived by Pydantic validators.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"

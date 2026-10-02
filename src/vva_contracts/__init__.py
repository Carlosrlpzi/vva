"""vva-contracts: deterministic measurement contracts for the Video Surveillance Assistant.

No language model is called anywhere in this package. Every number in a
report is computed from files in the workspace (or from a stream probed with
ffprobe) and re-checked by the Pydantic validators of its contract.
"""

__version__ = "0.1.0"

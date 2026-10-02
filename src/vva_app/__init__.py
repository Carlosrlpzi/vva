"""Smart Camera System application (runs on the Raspberry Pi 5 + Hailo-10H).

Package map (MVP Development Guide v11.1, guide path ``src/<x>`` -> ``src/vva_app/<x>``):
common (shared types), config (startup validation), ingest (RTSP/ONVIF, motion gate,
pre-roll), inference (detector, tracker), events (state machine, store), api,
notify, enrichment, eval (replay harness), observability. Module contracts:
``docs/module-contracts.md``.
"""

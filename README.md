# vva-agent: contratos de medición para el Video Surveillance Assistant

Paquete Python `vva_contracts`: tareas **deterministas y auditables** que miden la
calidad de un sistema de videovigilancia edge-first (Raspberry Pi 5 + Hailo-10H,
cámaras RTSP/ONVIF). **Ningún LLM se ejecuta dentro del paquete**: cada número sale de
archivos del workspace (o de `ffprobe`) y lo re-verifican los validadores Pydantic.

## Qué hay aquí

| Ruta | Contenido |
|---|---|
| `src/vva_contracts/` | Paquete: `core/` (IoU, AP COCO, phash, máquina de estados), `contracts/` (requests y reportes), `readers/`, `tasks/`, `cli.py`, `registry.py` |
| `tests/` | Pruebas unitarias, de extremo a extremo y el cross-check de AP contra `pycocotools` |
| `schemas/` | JSON Schemas generados con `vva schemas --output schemas` (la fuente de verdad son los modelos Pydantic) |
| `examples/` | Workspace sintético para probar las 6 tareas (se regenera con `python tests/builders.py examples`) |
| `context/` | Archivos del paquete anterior (`opencode-ml-contracts`), las 3 skills de OpenClaw y el documento del proyecto |
| `prompts/` | Prompt para el agente que refactoriza las skills (`refactor-skills-and-routing.prompt.md`) |

## Tareas

`DATASET_AUDIT`, `DETECTION_EVAL`, `QUANT_PARITY`, `RULE_REPLAY`, `STREAM_PROBE`, `PIPELINE_BENCH`.
Exit codes: 0 aceptado, 2 violación de contrato, 3 entrada inválida, 4 política bloqueada,
6 timeout de subproceso (1 = error interno).

## Uso

```bash
python3 -m venv .venv --system-site-packages && . .venv/bin/activate
pip install -e ".[dev]"            # añade ".[dev,crosscheck]" para la prueba contra pycocotools
echo '{"task":"DATASET_AUDIT","data_yaml":"dataset/data.yaml"}' | vva run --workspace examples
vva validate --task DATASET_AUDIT --file report.json --workspace examples   # verifica también la procedencia (SHA-256)
```

Verificación: `ruff check src`, `ruff format --check src`, `mypy src`, `pytest -q`.

## Estado honesto

**Existe y está verificado en Linux x86 (no en la Pi):** las 6 tareas, la CLI, los contratos,
las pruebas, `ruff` y `mypy --strict` sobre `src`.

**NO existe todavía** (lo debe producir el agente o hacerse en una segunda pasada):
- `.opencode/` completo: el tool `vva_contract` (bridge TypeScript), las skills `vva-*` y su prueba de consistencia.
- `opencode.json` y `AGENTS.md` del proyecto (los de `context/` son del paquete anterior, de ML tabular).
- Documentación en español (`docs/`), `VERIFICATION.md` de este paquete.

**Prueba que falla a propósito:** `tests/test_tasks_e2e.py::test_rule_replay_tradeoff`.
Una alerta de falso positivo abre el cooldown por cámara y suprime un evento real que llega justo después.
Es una **decisión de diseño abierta** (mantener cooldown por cámara / reiniciarlo solo con alertas
confirmadas / añadir a RULE_REPLAY la métrica de "eventos suprimidos"). No se debe "arreglar" editando la prueba.

**No verificado aquí:** rendimiento en la Pi/Hailo, un stream RTSP real, la compilación ONNX -> HEF.

## Privacidad

Nunca enviar frames, clips, URLs RTSP (llevan `usuario:contraseña@`) ni logs reales de detecciones
a un proveedor de LLM: las marcas de tiempo revelan cuándo hay gente en casa. Usa `tests/builders.py`.

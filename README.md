# Smart Camera System (VVA)

Sistema de videovigilancia inteligente, privado y **edge-first**: Raspberry Pi 5 + AI HAT+ 2
(Hailo-10H, 40 TOPS), cámaras PoE RTSP/ONVIF (Uniarch), switch Ruijie RG-ES209GC-P. La
detección, el seguimiento, la lógica de eventos y el almacenamiento corren en la Pi; internet
sólo se usa para alertas salientes. **Ningún LLM decide si una alerta se dispara**: el camino
de alerta es determinista (detector → tracker → regla de zona → máquina de estados →
persistencia → notificación). Los LLM/VLM se reservan para el desarrollo y para
enriquecimiento asíncrono posterior al evento.

## Un solo repositorio, dos paquetes

| Ruta | Qué es |
|---|---|
| `src/vva_app/` | La aplicación que corre en la Pi. Se construye hito a hito (hoy: esqueleto de paquetes). Cada ruta de la guía `src/<x>/...` vive en `src/vva_app/<x>/...` |
| `src/vva_contracts/` | Tareas de medición **deterministas y auditables** (sin LLM): DATASET_AUDIT, DETECTION_EVAL, QUANT_PARITY, RULE_REPLAY, STREAM_PROBE, PIPELINE_BENCH |
| `docs/mvp-guide.md` | MVP Development Guide **v11.2** (fuente de verdad del diseño); `docs/mvp-guide.es.md` en español |
| `docs/01-setup-guide.md` | Hardware, SO, red y Hailo |
| `docs/guia-opencode-ubuntu.md` | OpenCode con usuario dedicado y permisos mínimos |
| `docs/module-contracts.md` | Contrato por módulo (insumos, salidas, invariantes, tests, medición que lo verifica) |
| `AGENTS.md`, `opencode.json`, `prompts/`, `.opencode/` | Agentes de OpenCode (DeepSeek Flash + Pro), skills y el tool `vva_contract` |
| `tests/` | Pruebas del paquete de medición y `test_skills_consistency.py` |
| `schemas/` | JSON Schemas generados con `vva schemas --output schemas` (la fuente de verdad son los modelos Pydantic) |
| `examples/` | Workspace sintético para las 6 tareas (`python tests/builders.py examples`) |
| `context/`, `archive/` | Material del paquete anterior (ML tabular) y skills de OpenClaw archivadas, sin modificar |

## Mapa de paquetes de `vva_app` (hitos de la guía)

| Paquete | Módulos | Hito |
|---|---|---|
| `common` | Tipos compartidos (`FramePacket`, `Detection`, `Track`, `Event`) | antes de M1 |
| `config` | Validación de arranque de todos los YAML (C1) | desde M1 |
| `inference` | Detector YOLOv8n en Hailo-10H (M1), tracker Kalman/IoU (M5) | 1, 5 |
| `ingest` | RTSP/ONVIF (M2), puerta de movimiento (M4), pre-grabación (M9) | 2, 4, 9 |
| `eval` | Banco de reproducción de producción (M3) | 3 |
| `events` | Máquina de estados de entrada (M6), almacén SQLite (M7) | 6, 7 |
| `api`, `observability` | FastAPI, `/health`, `/metrics`, logs de evidencia (M7) | 7 |
| `notify` | Notificaciones y throttling (M8) | 8 |
| `enrichment` | Enriquecimiento local, asíncrono, sólo texto (M10) | 10 |

## Decisiones de diseño vigentes (guía v11.2)

- **Cooldowns de eventos en M6:** 30 s por `(track_id, zone_id)` y re-armado de 5 s. Personas
  distintas (tracks confirmados distintos) producen eventos de entrada distintos.
- **Throttling de notificaciones en M8:** 10 s por `(camera_id, event_type)`; nunca borra
  eventos y registra cada notificación suprimida (`notifications_throttled_total`).
- **Validación separada:** RULE_REPLAY compara políticas de detección **a nivel fotograma**;
  los eventos de producción se validan con el banco M3 ejecutando el código real de M6.
  Una tarea de reproducción basada en tracks queda para después.
- **Limitación conocida de RULE_REPLAY:** una alerta falsa abre el cooldown por cámara (p. ej.
  45 s) y puede suprimir un evento real justo después. `test_rule_replay_tradeoff` la fija
  como prueba de regresión (política laxa: recall 0.0; política por defecto: recall 1.0).
- **PIPELINE_BENCH:** presupuesto por defecto 400 ms sobre el p95 end-to-end (alarma de §1.10).

## Uso

```bash
python3 -m venv .venv --system-site-packages && . .venv/bin/activate
pip install -e ".[dev]"            # ".[dev,crosscheck]" añade la prueba contra pycocotools
npm --prefix .opencode ci          # dependencias del tool vva_contract de OpenCode
echo '{"task":"DATASET_AUDIT","data_yaml":"dataset/data.yaml"}' | vva run --workspace examples
vva validate --task DATASET_AUDIT --file report.json --workspace examples   # verifica también la procedencia (SHA-256)
```

Verificación: `ruff check src tests`, `ruff format --check src tests`, `mypy --strict src`, `pytest -q`.

Exit codes de `vva`: 0 aceptado, 2 violación de contrato, 3 entrada inválida, 4 política
bloqueada, 6 timeout de subproceso (1 = error interno).

## Variables de entorno (sólo nombres; los valores nunca van al repo)

| Variable | Uso |
|---|---|
| `VVA_CAM_<ID>_SUB_URL`, `VVA_CAM_<ID>_MAIN_URL` | URL RTSP de inferencia y de evidencia; el bridge de OpenCode sólo deja pasar `^VVA_CAM_[A-Z0-9_]+_URL$` |
| `VVA_CAM_<ID>_ONVIF_HOST`, `_ONVIF_USER`, `_ONVIF_PASSWORD` | Descubrimiento ONVIF (M2) |
| `DEEPSEEK_API_KEY` | Sólo en el entorno del usuario `opencode-dev`, para OpenCode |

## Privacidad

Nunca enviar frames, clips, URLs RTSP (llevan `usuario:contraseña@`) ni logs reales de
detecciones a un proveedor de LLM: las marcas de tiempo revelan cuándo hay gente en casa.
Usa `tests/builders.py` para datos sintéticos.

## Estado honesto

- **Verificado en Linux x86 (no en la Pi):** las 6 tareas, la CLI, los contratos, las pruebas,
  `ruff` y `mypy --strict` sobre `src`, y la consistencia de las skills.
- **Sin verificar:** el bridge TypeScript de `.opencode/` (falta `npm ci` + smoke test), OpenCode
  con DeepSeek real, rendimiento en la Pi/Hailo, un stream RTSP real, la compilación ONNX → HEF.
- **`src/vva_app/` es sólo un esqueleto:** los módulos se implementan hito a hito con OpenCode.
- El historial de git aún contiene `packages/vva-contracts/` (implementación duplicada borrada
  del árbol de trabajo); confirma su borrado con `git rm -r packages/vva-contracts`.

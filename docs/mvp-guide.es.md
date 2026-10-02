# Smart Camera System — MVP Development Guide

Esta guía retoma exactamente donde `01-setup-guide.md` la deja. El hardware está ensamblado, el sistema operativo está flasheado, el Hailo‑10H está verificado en tres niveles (`lspci`, `hailortcli scan`, `hailortcli fw-control identify`), las cámaras responden a `ffprobe`, y el venv de Python importa `cv2`, `onvif` y `hailo_platform` sin errores. Este documento trata sobre el **código de la aplicación** para el MVP de una sola cámara: qué hace cada elemento, por qué existe, cómo funciona internamente y dónde leer más si quieres profundizar más allá del resumen.

Sigue la arquitectura parchada ya acordada para este proyecto — **YOLOv8n** (no YOLO26, que el Hailo‑10H no soporta actualmente), una **puerta de movimiento** antes del NPU, un **tracker Kalman/IoU parametrizado en el tiempo** en lugar de la confirmación ingenua por fotogramas consecutivos, y **enriquecimiento GenAI en el chip** mediante Hailo‑Ollama/HailoRT en lugar de cualquier API en la nube.

```text
RTSP substream → threaded single-slot capture → MOG2 motion gate (zone-masked) →
Hailo-10H YOLOv8n inference (letterboxed) → time-parameterized Kalman/IoU tracking →
track-based zone/event state machine → async SQLite write + pre-roll clip flush → API →
optional local Hailo GenAI enrichment
```

El objetivo del MVP es deliberadamente acotado: **una cámara, una clase (`person`), una regla de zona**, funcionando de extremo a extremo. Una vez que ese segmento sea estable, los mismos módulos se generalizan a múltiples cámaras.

---

## Notas de revisión

### v2 — Primera revisión arquitectónica (8 de sept. de 2026)

Una revisión de diseño de la guía v1 sacó a la luz varios puntos donde el pipeline, tal como estaba especificado, se habría descompuesto silenciosamente a sí mismo o habría producido resultados engañosos. Esta revisión incorporó cuatro categorías de correcciones:

1. **Correcciones matemáticas/arquitectónicas** — el umbral de confianza del detector estaba descartando silenciosamente las cajas que el nivel de baja confianza del tracker fue diseñado para usar; `min_confirmed_hits` estaba duplicado en dos archivos de configuración; el filtro de Kalman asumía un intervalo de fotogramas fijo que la programación irregular de la puerta de movimiento rompe; la expiración de tracks estaba especificada en fotogramas en lugar de segundos; la puerta de movimiento por diferenciación de fotogramas es poco adecuada para una escena exterior de 24/7; y la entrada cuadrada del detector estaba estirando una fuente 16:9 en lugar de aplicarle letterbox.
2. **Cuellos de botella del sistema** — el buffering interno de `cv2.VideoCapture` (el "smear de OpenCV") puede servir silenciosamente fotogramas cada vez más obsoletos bajo carga sin ningún error visible; no se definió ningún presupuesto de latencia explícito ni un endpoint `/metrics`; y las marcas de tiempo de los eventos necesitaban fijarse al momento de captura del fotograma, no al momento de procesamiento.
3. **Componentes faltantes** — no había un banco de reproducción/evaluación fuera de línea para calibrar los umbrales contra la verdad de referencia, no había un búfer de evidencia de pre-grabación, no había una verificación de precisión INT8 contra FP32 en el HEF compilado, no había un comportamiento definido para disco lleno/rotación de logs/desgaste de la SD, y no había ninguna nota sobre las implicaciones de la licencia AGPL-3.0 del toolchain de Ultralytics YOLOv8.
4. **Reordenamiento del cronograma** — el elemento de mayor riesgo (compatibilidad del HEF) se movió a la semana 1 como una prueba exploratoria fuera de línea, la ingesta de RTSP se movió a la semana 2, el banco de reproducción fuera de línea se construyó antes de cualquier ajuste de umbrales, y la máquina de estados de eventos se entrega solo con entrada (entry-only).

### v3 — Segunda revisión (8 de sept. de 2026)

Una revisión de seguimiento de v2 encontró que varias de las correcciones de v2 introdujeron defectos nuevos, y que una corrección se aplicó en un lugar pero no de forma consistente en otros. Esta revisión agrega:

1. **Errores en los ejemplos de código de v2** — el fragmento de MOG2 cuenta los píxeles de sombra (valor 127) como movimiento, anulando por completo `detectShadows: true`; `mog2_history` sigue siendo un *conteo de fotogramas* aunque el mismo problema de Δt idéntico se corrigió para el filtro de Kalman; `FreshFrameReader` descarta silenciosamente la lógica de reconexión que exige §1.1 y entra en un bucle activo (busy-loop) en la lectura fallida sin ninguna señal de obsolescencia; y `time.monotonic()` no puede servir como la marca de tiempo registrada de un evento.
2. **Correcciones conceptuales** — §1.10 confundía *throughput* (debe caber en un periodo de fotograma) con *latencia* (captura → evento), y su presupuesto sumaba exactamente el 100% del tiempo disponible sin ningún margen; la escritura en SQLite no pertenece a la ruta crítica por fotograma; el banco de reproducción nunca definió su regla de coincidencia de eventos, sin la cual la precisión y el recall no son calculables; y el ajuste de umbrales contra un conjunto pequeño etiquetado a mano necesita una partición reservada.
3. **Componentes faltantes** — un **presupuesto de píxeles previo al arranque** (§0) que determina si YOLOv8n puede ver a una persona a la distancia de tu zona *en absoluto*, lo cual podría invalidar la elección de modelo o de lente antes de escribir cualquier código; `storage_mode: encoded_packets` (§1.12) no es alcanzable con OpenCV y requiere **PyAV** además de un corte alineado a keyframes y un cambio de GOP del lado de la cámara; el manejo de fallas del NPU; y una nota sobre la batería del RTC.
4. **Refinamientos matemáticos de la fase 2 (§4)** — la forma continua físicamente correcta de ruido blanco de \(Q(\Delta t)\) (cúbica en posición, no meramente \(\propto \Delta t^2\)), y el mapeo de perspectiva inversa (Inverse Perspective Mapping) mediante una matriz de homografía para reemplazar las distancias en píxeles dependientes de la profundidad por distancias métricas en el plano del suelo.
5. **Correcciones de cronograma** — el hito 4 dependía de un artefacto (polígonos de zona) que no se producía hasta el hito 6; la grabación continua de video es el punto crítico del *calendario* y debe iniciar de forma pasiva al final del hito 2; y el hito 8 agrupaba los dos elementos restantes de mayor riesgo en una sola semana.

### v4 — Revisión independiente de ingeniero senior (8 de sept. de 2026)

Una revisión externa de v3 encontró que las revisiones previas eran tácticamente sólidas pero dejaban sin resolver varias brechas de nivel de dominio y operativas — del tipo que no se manifiestan como errores de código, sino como "el pipeline funciona pero el producto no". Esta revisión agrega:

1. **Riesgo de brecha de dominio del detector** — la validación cruzada INT8 contra FP32 en §1.3 solo demuestra que la cuantización no perjudicó la precisión respecto a los *mismos* pesos preentrenados; no dice nada sobre si los pesos calibrados con COCO2017 generalizan a la cámara fija en ángulo oblicuo de la entrada de este proyecto, a la distancia, y — de forma crítica — al video nocturno con IR, que arquitectónicamente es el caso menos probado y de mayor valor para un sistema de seguridad doméstica.
2. **Un cuello de botella de decodificación estructural, no incidental** — la Raspberry Pi 5 **no tiene ningún decodificador de hardware H.264**; §1.10 subestimó esto como "una etapa a vigilar en un dashboard" cuando en realidad es un techo duro de CPU con una mitigación concreta (substreams H.265/HEVC, si las cámaras lo soportan).
3. **Un intervalo de confianza estadísticamente incorrecto en el propio ejemplo resuelto del banco de reproducción** — el intervalo de aproximación normal/Wald usado en §1.11 está documentado como subcobertor con n pequeña y proporciones cercanas a 1.0, que es exactamente el régimen en el que estarán los conjuntos de etiquetas de este proyecto durante meses; los intervalos de Wilson o de Jeffreys son la herramienta correcta.
4. **Ninguna supervisión de fallas en tiempo de ejecución** — las verificaciones de arranque de §1.13 cubren el fallo al *iniciar*; nada en v1–v3 cubre una falla, un colgado o un OOM en mitad de la ejecución una vez que el pipeline ya está corriendo.
5. **Brechas a nivel de producto que ninguna corrección de precisión del pipeline resuelve** — no hay canal de notificación/alerta, no hay forma de ver un clip sin extraer archivos manualmente, no hay evaluación por etapa (solo de extremo a extremo), no hay una frontera declarada de autenticación/red de la API, no hay un presupuesto de memoria para todo el sistema, no hay detección de manipulación/obstrucción de cámara, no hay una historia de pruebas unitarias/CI para las piezas de función pura (IoU, \(Q(\Delta t)\), punto-en-polígono, homografía), y no hay una política de retención independiente de la presión de disco.

Las nuevas secciones **1.14–1.17** cubren estos puntos (canal de notificación, UI mínima de operador, detección de manipulación/obstrucción de cámara, y pruebas unitarias/CI para funciones puras); §1.3, §1.7, §1.10, §1.11 y §1.13 reciben correcciones puntuales; y el cronograma de hitos y la lista de verificación de preparación se actualizan en consecuencia.

### v5 — Revisión independiente del parche v4 (8 de sept. de 2026)

Una revisión adicional cotejó las correcciones de v4 contra evidencia externa y el propio historial del proyecto, y encontró la sustancia sólida pero varios detalles necesitaban precisión. Esta revisión:

1. **Replantea el cuello de botella de decodificación de §1.10 de un techo duro a un elemento de medición del hito 1** — se reporta ampliamente que la decodificación H.264 por software de la Pi 5 supera al antiguo decodificador de hardware de la Pi 4 en lugar de ser estrictamente peor, por lo que el presupuesto `decode: 25 ms` de §1.10 ahora es explícitamente provisional en espera de una medición directa, y la recomendación de H.265 se corrige para señalar específicamente la bandera `-hwaccel drm` (no solo `v4l2m2m`), ya que reduce de forma medible el costo de CPU en la Pi 5.
2. **Reemplaza la cita CASRAI de §1.11** por el artículo canónico de Brown, Cai y DasGupta (2001) sobre intervalos de confianza para proporciones, que es la fuente real detrás de la recomendación de Wilson frente a Wald.
3. **Refuerza las citas de baja luz/IR nocturno de §1.3** con dos benchmarks publicados concretos (una evaluación de vigilancia nocturna con YOLOv8 y una revisión sistemática de Springer sobre YOLOv8–v11 en ExDark) en lugar de dos fuentes más débiles.
4. **Rastrea la brecha de IR nocturno hasta tres consecuencias concretas río abajo** que v4 nombró pero no conectó con correcciones específicas: la elección BGR contra escala de grises de MOG2 (§1.2) es un ajuste válido solo en modo diurno una vez que las cámaras cambian a video IR monocromático; los insectos atraídos por el IR son una fuente de falsos positivos distinta de las sombras/ramas diurnas que ni MOG2 ni el margen de ROI filtran, lo que requiere una corrección río abajo en el tracker/clasificador y un caso de prueba negativo explícito `night_ir`; y el alcance del iluminador IR (típicamente 20–30 m) limita el límite de la zona nocturna independientemente de la geometría de altura en píxeles de §0.
5. **Corrige la heurística de manipulación de §1.16** para que un cambio rutinario de IR día/noche — que cada cámara de este proyecto realiza dos veces al día — ya no active `tamper_suspected`, mediante una validación cruzada de modo IR y/o un requisito de persistencia posterior al cambio.
6. **Le da a la notificación su propio espacio de hito.** El hito 7 (v4) agrupaba persistencia/API/`/metrics` con el cableado del consumidor de notificaciones en una sola semana; la notificación ahora tiene su propio hito 8, empujando una semana cada uno los hitos de pre-grabación, enriquecimiento local, y endurecimiento/soak/demo (el hito final ahora es el 11, no el 10). §1.14 también queda vinculado explícitamente al patrón de bandeja de salida de alertas ya decidido previamente para el proyecto, webhook primero y Telegram en segundo lugar.

### v6 — Cinco correcciones puntuales (8 de sept. de 2026)

Un pase enfocado que corrige cinco problemas específicos identificados en v5, sin una revisión más amplia. Esta revisión:

1. **Replantea el benchmark de IR nocturno de §1.3 como un techo, no como evidencia de degradación.** Las cifras nocturnas citadas (mAP@50 0.908/0.819/0.886) están por encima del propio mAP de COCO de YOLOv8n (~0.52), por lo que describen lo que logra un modelo ajustado al dominio de noche, no lo que harán los pesos preentrenados con COCO del proyecto — la revisión sistemática de ExDark sigue siendo la fuente correcta para la afirmación de degradación, y el ajuste fino (fine-tuning) del hito 3 sobre las etiquetas `night_ir` es la acción declarada de la fase 2.
2. **Renombra `kalman_sigma_accel` a `kalman_sigma_accel_sq`** (§1.4/§4.1), ya que el campo contenía \(\sigma_a^2\) (tal como lo requiere el parámetro `spectral_density` de `filterpy`) a pesar de que su nombre implicaba \(\sigma\). Declara explícitamente sus unidades como px\(^2\)/s\(^4\) (espacio de píxeles, previo a la homografía), reemplaza el marcador de posición implausible `1.0` por un valor inicial de `30.0` para que el banco de §1.11 lo ajuste, y señala que un valor demasiado pequeño reproduce la falla de covarianza sobreconfiada que describe §4.1.
3. **Agrega procedencia de configuración al banco de reproducción de §1.11** — `replay_runner.py` ahora emite un hash de la configuración fusionada junto con `n_labeled_events`, con un campo correspondiente agregado a `eval.yaml`, de modo que un número de precisión/recall reportado pueda rastrearse hasta la configuración exacta (entre los once archivos YAML del pipeline) que lo produjo.
4. **Saca las pruebas unitarias de función pura del backlog de la fase 2** (§1.17/§3). IoU y \(Q(\Delta t)\) ahora se escriben en el hito 5 y punto-en-polígono en el hito 6 — junto con el código que primero las necesita — en lugar de diferirse a una única semana de limpieza posterior.
5. **Reubica el cambio de decodificación H.265/HEVC** (§3 hito 2 / §1.10) fuera de los criterios de salida del hito 2 y de la lista de verificación de preparación de una sola cámara, ya que el ahorro citado (13% → 9% de un núcleo, ~1% del CPU total) no vale la complejidad adicional de ruta de remux de §1.12 con una sola cámara. Ahora vive en el párrafo de generalización multi-cámara al final de §3, donde cuatro streams hacen que el ahorro valga la pena; el detalle de `-hwaccel drm` se traslada con él.

### v7 — Dos correcciones puntuales (9 de sept. de 2026)

Un pase adicional enfocado que corrige dos problemas específicos, sin una revisión más amplia. Esta revisión:

1. **Implementa la división `day` / `night_ir` y el recall por fotograma del detector en `config/eval.yaml`** (§1.11). La división se había solicitado en cinco lugares (la nota v4 de esta sección, la corrección v4 de §1.3, la adición v5 de §1.2, los criterios de salida del hito 3, y la lista de verificación de preparación) pero nunca se agregó al bloque de configuración que la implementa, por lo que nunca se calculó ningún recall del detector por fotograma para distinguir una brecha de dominio genuina de un umbral mal ajustado. Agrega `conditions: [day, night_ir]` y `condition_source` al bloque `split:`, una lista separada `detector_metrics: [per_frame_recall, per_frame_precision]`, y una entrada `negative_test_cases:` para el caso de insectos IR de §1.2; agrega una columna `condition` al esquema del CSV de verdad de referencia, ya que de otro modo no había dónde registrarla.
2. **Corrige la aritmética y el valor de `kalman_sigma_accel_sq`** (§1.4). La propia afirmación de la corrección v6 era internamente inconsistente — "decenas de px/s²" al cuadrado da cientos a miles, no "decenas" de px²/s⁴ — lo que hacía que el valor v6 de `30.0` fuera aproximadamente dos órdenes de magnitud demasiado pequeño. Al rederivar a partir de la tabla de presupuesto de píxeles de §0 (54 px a 15 m, 163 px a 5 m, ambos para el lente de 4.0 mm) se obtiene un rango defendible de 10³–10⁴ px²/s⁴ dependiendo de la distancia a la cámara; establece `kalman_sigma_accel_sq: 2000.0` como el nuevo punto de partida para que el banco de §1.11 lo ajuste.

### v8 — Tres correcciones puntuales (9 de sept. de 2026)

Un pase adicional enfocado que corrige tres problemas específicos, sin una revisión más amplia. Esta revisión:

1. **Agrega `kalman_sigma_accel_sq` a lo que corrige el mapeo de perspectiva inversa** (§4.2). La aritmética rederivada de §1.4 muestra que \(\sigma_a^2\) oscila aproximadamente 9× entre 5 m y 15 m en el mismo fotograma — la misma dependencia de profundidad que §4.2 ya señala para `max_centroid_distance_px`. En coordenadas métricas del plano del suelo, \(\sigma_a^2\) se convierte en una varianza de aceleración física (aproximadamente 2–3 m²/s⁴ para una persona caminando) que es constante en toda la zona y se transfiere entre cámaras, igual que `max_association_distance_m`.
2. **Corrige la acción de ajuste fino de la fase 2** (§1.3). La corrección v6 decía que el ajuste fino procede "sobre esas etiquetas", refiriéndose a la división `night_ir` a nivel de evento del hito 3 — pero el ajuste fino del detector necesita cajas delimitadoras por fotograma, que esas etiquetas no contienen y no pueden derivarse de las marcas de tiempo de los eventos. Reformulado para indicar que las etiquetas del hito 3 solo revelan si existe una brecha nocturna (mediante el recall del detector por fotograma, §1.11); cerrarla es un proyecto de fase 2 separado y acotado que requiere su propio pase de anotación de cajas delimitadoras, un reentrenamiento, y una nueva compilación de HEF con un conjunto de calibración específico del dominio, y el reentrenamiento hereda la preocupación existente de la sección sobre AGPL-3.0 ya que se ejecuta mediante las herramientas de entrenamiento de Ultralytics.
3. **Corrige la contradicción de fechas entre `eval.yaml` y el hito 3** (§1.11/§3). `split.tuning_days` y `heldout_days` estaban codificados de forma fija (hardcoded) al 28 de sept.–1 de oct. de 2026, lo cual cae dentro de la propia semana del hito 3 — después de que el CSV de verdad de referencia se etiqueta a partir del material de la semana 2 (21–27 de sept.), no antes. Se reemplazaron las fechas literales por marcadores de posición y un comentario que señala que se mueven con el cronograma, en lugar de afirmar una fecha fija distinta que podría volver a desincronizarse.

### v9 — Tres correcciones puntuales (9 de sept. de 2026)

Un pase enfocado que aplica tres de los elementos de más alta prioridad de una revisión de ingeniero senior de v8, sin una revisión más amplia. Esta revisión:

1. **Extiende el canal de notificación de §1.14 a señales de salud del sistema, no solo a eventos emitidos.** `min_severity: entry_event` significaba que las transiciones de `/health` a degradado/caído, `npu_failures_total`, y la señal `tamper_suspected` de §1.16 nunca llegaban a un humano — el mismo punto ciego de "a nadie se le avisa" que §1.14 fue construido para cerrar, solo que para fallas de infraestructura en lugar de un evento perdido. Agrega los niveles de severidad `health_degraded` / `health_down` / `tamper_suspected` y una segunda ruta de consumidor que vigila las transiciones de `/health` junto con la ruta existente de confirmación de eventos.
2. **Agrega una capa de validación de configuración en el arranque** (§1.18, nueva). Cuatro de las ocho revisiones previas fueron causadas por el mismo problema de fondo — desincronización (drift) entre los archivos YAML independientes del proyecto (la duplicación v1 de `min_confirmed_hits`/`required_track_hits`, los errores de nombre y valor de `kalman_sigma_accel` de v6/v7, la contradicción de fecha de `eval.yaml`/hito de v8). §1.18 agrega una verificación de fallo temprano, ejecutada antes de que inicie el hilo de ingesta, que confirma que `detector.confidence_threshold == tracker.low_confidence_threshold` y que existe el archivo de polígono de zona de cada cámara configurada — la misma clase de invariante que previamente solo se hacía cumplir mediante una lectura cuidadosa.
3. **Agrega una verificación de fallo temprano por marcador de posición al banco de reproducción de §1.11.** v8 reemplazó las fechas literales y desincronizadas de `eval.yaml` por marcadores de posición estilo `<tuning-day-1>`, pero nada impedía que `replay_runner.py` se ejecutara silenciosamente contra un marcador de posición sin rellenar. Agrega `reject_unresolved_placeholders: true` a `eval.yaml` y una verificación de arranque correspondiente en `replay_runner.py` que se niega a ejecutarse — en lugar de producir una falla confusa o un no-op silencioso — mientras cualquier entrada de `split.tuning_days`/`heldout_days` siga coincidiendo con el patrón de marcador de posición `<...>`.

### v10 — Tres correcciones puntuales (9 de sept. de 2026)

Un pase adicional enfocado que aplica tres elementos más de la misma revisión de ingeniero senior de v8, sin una revisión más amplia. Esta revisión:

1. **Agrega una ruta de reversión (rollback) y validación en sombra para el reentrenamiento del detector en fase 2** (§1.3). La corrección v6 ya documenta cómo producir un HEF ajustado finamente para la brecha de dominio de IR nocturno; no decía qué pasa si ese HEF resulta peor que el ya desplegado. Agrega el requisito de validar el modelo reentrenado contra la misma partición reservada (§1.11) antes de promoverlo, y de mantener direccionable el HEF anterior y sus valores de `models.yaml` para que una regresión pueda revertirse.
2. **Agrega una cadencia de etiquetado continua y progresiva más allá del hito 3** (§1.11/§3). Es poco probable que `minimum_labeled_events_for_hard_gate: 100` se alcance con una sola semana de etiquetado del hito 3, lo que dejaría los criterios de salida de CI en modo consultivo indefinidamente. Agrega una nota en ambas secciones indicando que el etiquetado continúa con una cadencia regular a partir de la grabación 24/7 ya en marcha, en lugar de tratar el hito 3 como un pase de etiquetado único.
3. **Agrega una nota sobre el ciclo de vida del token portador (bearer token) y la seguridad de transporte, y elimina las credenciales de cámara codificadas de forma fija (hardcoded) del ejemplo de §1.1** (§1.1/§1.7). El ejemplo de código de ONVIF previamente tenía codificados de forma fija un usuario y una contraseña; se reemplazaron por búsquedas de variables de entorno. La adición v4 en §1.7 decía que había que agregar una verificación de token portador pero no cómo generarlo, almacenarlo, rotarlo o transportarlo; agrega una guía concreta para cada uno, incluyendo poner la API detrás de TLS.

### v11 — Vitalidad fuera de proceso y tres correcciones de bloqueo (10 de sept. de 2026)

Un pase enfocado que cierra el único punto ciego que una auditoría de v10 calificó como más grave, más tres elementos que estaban documentados pero nunca bloqueados (gated). Esta revisión:

1. **Agrega monitoreo de vitalidad (liveness) fuera de proceso** (§1.13/§1.14/§3). El consumidor de `/health` de la v9 de §1.14 se ejecuta dentro del proceso del pipeline, por lo que no puede reportar una falla, un OOM kill, un systemd que agota `start_limit_burst: 5`, una pérdida de energía o una pérdida de red. Agrega dos mecanismos que cubren conjuntos de fallas disjuntos: una unidad oneshot `OnFailure=camera-alert@%n.service` (§1.13) que envía webhooks desde fuera del proceso caído ante una falla, un colgado, o el agotamiento del límite de reinicios; y un dead-man's switch (§1.14, `dead_mans_switch:` en `notify.yaml`) — un job de APScheduler que hace ping a un endpoint externo, el cual alerta cuando los pings se detienen — la única ruta que sobrevive a una pérdida de energía, una pérdida de red, o un kernel atascado. Ambos se verifican en el hito 11 (matar el proceso; apagar la Pi) y quedan bloqueados en la lista de verificación de preparación.
2. **Bloquea dos correcciones existentes que nada hacía cumplir** (solo §3). La prueba negativa `night_ir_insect_activity` ya presente en `eval.yaml` ahora debe pasar (cero eventos emitidos) en el hito 3 y en la lista de verificación; las credenciales de cámara por variable de entorno de §1.1 deben verificarse en el hito 2 y en la lista de verificación, sin ninguna credencial en ningún lugar del repositorio.
3. **Delimita honestamente dos afirmaciones** (§1.16/§1.18). §1.16 ahora indica que detecta obstrucción/reencuadre solo mientras la Pi está encendida y en funcionamiento, delegando el vector de energía/red al dead-man's switch de §1.14. El conteo de archivos YAML se corrige de ocho a once en todos los lugares donde aparece (§1.18, la adición v6 de §1.11, y la nota de revisión v6) — once es el número de bloques `config/*.yaml` que en realidad hay en este documento — y §1.18 ahora indica que la validación es solo en el arranque, sin ninguna ruta de recarga en caliente planeada para el MVP.
4. **Crea `docs/post-mvp-backlog.md`** para elementos diferidos (más invariantes de §1.18, recarga en caliente de configuración, una verificación de CI de trazabilidad), con un señalador de una línea al inicio de §3.

### v11.1 — Cuatro correcciones bloqueantes de la revisión del Model Council (10 de sept. de 2026)

Un pase de revisión independiente de tres modelos, contrainterrogatorio y adjudicación (Astra, Claude Fable 5.1, Gemini 3.1 Pro) encontró que cada una de las correcciones de vitalidad de v11 anteriores estaba socavada por un defecto pequeño y concreto en el texto literal. Las cuatro se corrigen aquí con ediciones de 1 a 3 líneas cada una; no se agregaron secciones ni componentes nuevos. Todo lo demás que el consejo consideró fue explícitamente diferido o descartado por no ser bloqueante para el MVP de una sola cámara.

1. **La línea `EnvironmentFile=` de `camera-alert@.service` llevaba un comentario al final** (§1.13) — systemd no soporta comentarios en línea en las directivas de unidad, por lo que el comentario se interpretaba como parte de la ruta y la unidad no podía activarse, anulando silenciosamente el webhook de alerta de falla de v11 ante una falla y reinicio ordinarios. Corregido moviendo el comentario arriba de la directiva.
2. **El notificador se especificó como un segundo consumidor de la propia cola de entrada del hilo de escritura de §1.6** (§1.14, y se repite en la tabla del hito 8 en §3) — dos consumidores compitiendo por una sola cola dividían los eventos entre ellos sin levantar ningún error. Corregido haciendo que el escritor publique los IDs de eventos confirmados en una cola de notificación separada y dedicada.
3. **`FreshFrameReader.get_latest()` no tenía ninguna señal de fotograma nuevo** (§1.10) — un consumidor más rápido que la tasa de fotogramas de la cámara podía reprocesar un mismo fotograma físico múltiples veces, permitiendo que un artefacto de un solo fotograma satisficiera `min_confirmed_hits` contra sí mismo y reintroduciendo el falso positivo por parpadeo (flicker) que el tracker fue construido para corregir. Corregido agregando una bandera `is_new` al valor de retorno de `get_latest()` y condicionando a ella las actualizaciones de MOG2/inferencia/hits del tracker.
4. **El latido (heartbeat) de `sd_notify` solo demostraba que el bucle principal estaba vivo, no los hilos del escritor de SQLite ni del notificador** (§1.13) — si cualquiera de los hilos trabajadores moría, el heartbeat, el dead-man's switch, y `/health` seguirían reportando estado saludable mientras los eventos se perdían silenciosamente. Corregido condicionando el heartbeat a `all(t.is_alive() for t in (reader, writer, notifier))`.

### v11.2 — Alcances de cooldown separados entre eventos y notificaciones; repositorio único (2 de oct. de 2026)

Decisiones del dueño del proyecto, registradas para que los agentes que escriben el código las sigan literalmente:

1. **Los cooldowns de eventos se quedan en la máquina de estados (§1.5, hito 6):** `same_track_same_zone_seconds: 30` y `rearm_after_track_lost_seconds: 5`. **Personas distintas producen eventos de entrada distintos:** dos `track_id` confirmados diferentes que entran a la misma zona son dos eventos, por cercanos que estén en el tiempo.
2. **El límite de 10 s por `(camera_id, event_type)` pasa al canal de notificación (§1.14, hito 8)** como `notify.throttle.same_camera_same_event_seconds: 10`. Limita *notificaciones*, nunca eventos: todo evento se persiste igual, y cada notificación suprimida se registra (fila + contador `notifications_throttled_total`), para que el historial de eventos quede completo.
3. **Validación separada.** La tarea RULE_REPLAY de `vva_contracts` evalúa sólo políticas de detección *a nivel fotograma* (confianza, fotogramas consecutivos, cooldown por cámara). Su limitación conocida — una alerta falsa abre el cooldown por cámara y puede suprimir un evento real inmediatamente después — queda fijada por una prueba de regresión, no "corregida". El comportamiento de eventos de producción (M6 + M8) se valida con el banco de reproducción del hito 3 (§1.11) ejecutando el código real de M6. Más adelante se podrá agregar a `vva_contracts` una tarea de reproducción basada en tracks.
4. **Repositorio único.** La aplicación vive en `src/vva_app/` junto a `src/vva_contracts/`; toda ruta de la guía `src/<x>/...` corresponde a `src/vva_app/<x>/...` (por ejemplo `src/events/state_machine.py` → `src/vva_app/events/state_machine.py`).
5. **Presupuesto por defecto de PIPELINE_BENCH = 400 ms**, igual a `end_to_end_latency_ms.alarm_p95` (§1.10).

---

## 0. Verificación previa al arranque: el presupuesto de píxeles

**Haz esto antes de escribir cualquier código.** Toma un fotograma y una cinta métrica, y su respuesta puede invalidar la elección de modelo, la resolución del substream, o la selección de lente — todo lo cual es costoso de descubrir en el hito 5.

**El problema.** Cada umbral en este documento asume que el detector realmente puede ver a una persona a la distancia que importa. El stride de detección más fino de YOLOv8n es de 8 píxeles, y la confiabilidad de la detección se degrada bruscamente una vez que la altura de un objeto en el fotograma de entrada del modelo cae por debajo de aproximadamente 30–40 px. Nada en v1 ni v2 verificaba si esa condición se cumplía para tus cámaras.

**El cálculo.** A partir del modelo de cámara estenopeica (pinhole), la altura en píxeles de un objeto de altura real \(H\) a una distancia \(D\) es:

\[
h_{px} = \frac{P_{h} \cdot f \cdot H}{D \cdot S_{h}}
\]

donde \(P_h\) es la altura de la imagen en píxeles, \(f\) la distancia focal del lente (mm), \(S_h\) la altura del sensor (mm, según la hoja de datos de la cámara), y \(H, D\) en las mismas unidades.

Ejemplo resuelto, asumiendo un sensor de 1/3" (\(S_h \approx 3.0\) mm), una persona de 1.7 m, y un substream de 640×360:

| Lente | Distancia | Altura de la persona (px) | Veredicto |
|---|---|---|---|
| 4.0 mm | 5 m | ~163 px | Cómoda |
| 4.0 mm | 15 m | ~54 px | Funcional |
| 4.0 mm | 25 m | ~33 px | Marginal — se esperan omisiones |
| 2.8 mm | 15 m | ~38 px | Marginal |
| 2.8 mm | 25 m | ~23 px | Por debajo del piso — fallará |

**Sustituye tus propios valores de hoja de datos.** La tabla es ilustrativa; la altura del sensor en particular varía de forma significativa entre las piezas de 1/3", 1/2.8" y 1/2.7".

**La consecuencia no obvia: un substream de mayor resolución no ayuda automáticamente.** Debido a que el detector aplica letterbox a una entrada fija de 640×640 (§1.3), un substream de 640×360 se escala por 1.0 (el ancho ya cabe, solo se agrega relleno vertical) mientras que un substream de 1280×720 se escala por 0.5. Una persona de 108 px de altura en 720p se convierte en 54 px después del letterbox — idéntico al caso de 640×360. Aumentar la resolución del substream no te da nada a menos que también cambies *cómo* el fotograma llega al modelo.

Si el presupuesto de píxeles resulta marginal, tienes tres opciones reales, en orden creciente de esfuerzo:

- **Campo de visión más corto.** Un lente de 4.0 mm en lugar de 2.8 mm pone aproximadamente 43% más píxeles sobre el objetivo a la misma distancia. Esta es también la base cuantitativa para resolver la pregunta aún abierta de 2.8 mm contra 4.0 mm para la cámara #3 en el documento de arquitectura — mide la distancia de la zona y calcula, en lugar de estimar "el ancho del patio trasero".
- **Recortar e inferir (crop-and-infer).** En lugar de aplicar letterbox a todo el fotograma, recorta el polígono de zona (más margen) a la resolución nativa del substream y alimenta *eso* a la entrada de 640×640. Una persona que ocupa 33 px de un fotograma completo puede ocupar 90+ px de un recorte ajustado. Te cuesta conciencia fuera del recorte, lo cual es aceptable cuando tu regla de zona de todos modos solo le importa una región.
- **Escalar el modelo.** YOLOv8s tiene un recall de objetos pequeños significativamente mejor que YOLOv8n, y el Hailo‑10H tiene amplio margen para ello (§1.10). Este es el mismo punto de decisión que alimenta la verificación INT8 contra FP32 en §1.3.

**Adición v5 — el presupuesto nocturno está limitado por el alcance del IR, no solo por la ecuación de lente anterior.** La tabla del modelo pinhole anterior asume suficiente luz ambiental para que el sensor resuelva detalle a la distancia calculada; después del anochecer, la mayoría de las cámaras PoE de consumo/prosumer cambian a iluminación IR con un alcance efectivo especificado — comúnmente entre 20 y 30 m para esta clase de cámara — más allá del cual la imagen simplemente está demasiado oscura para exponerse, sin importar cuántos píxeles coloque la geometría sobre el objetivo. Ejecuta la misma tabla de altura en píxeles usando el alcance real del iluminador IR de tu cámara en lugar de (o junto con) su línea de visión diurna, y trata el que sea menor — el piso óptico de altura en píxeles o el piso de iluminación IR — como el verdadero límite de la zona nocturna. Registra ambas distancias en `configs/`: una zona trazada solo a partir de la visibilidad diurna puede reclamar silenciosamente una cobertura nocturna que la cámara no puede ofrecer.

**Criterio de salida:** una altura en píxeles medida para una persona de pie en el límite de la zona de cada cámara, registrada en `configs/` junto con el polígono de zona, y una decisión escrita sobre lente/resolución/modelo antes de que inicie el hito 1 — cubriendo tanto el caso diurno como el caso nocturno limitado por el alcance del IR.

---

## 1. Referencia de componentes

Cada sección a continuación corresponde a un módulo en `src/`. Para cada componente obtienes: **Propósito** (el problema que existe para resolver), **Cómo funciona** (el mecanismo, con las matemáticas donde importa), y **Lecturas adicionales**.

### 1.1 Ingesta de cámara RTSP/ONVIF — `src/ingest/rtsp_reader.py`, `onvif_client.py`

**Propósito.** Convertir una cámara IP física en un flujo estable de fotogramas decodificados sobre los que tu código Python puede operar, sin codificar de forma fija (hard-coding) formatos de URL específicos de cada proveedor que se rompen en el momento en que se actualiza el firmware de una cámara.

**Cómo funciona.** ONVIF es un protocolo estandarizado de descubrimiento de dispositivos y perfiles de medios que implementan la mayoría de las cámaras IP comerciales (Uniarch, Tiandy, Hikvision‑OEM, etc.). En lugar de adivinar una ruta de stream como `/Streaming/Channels/102`, el código llama a los métodos SOAP `GetProfiles()` y `GetStreamUri()` de la cámara para obtener la URL RTSP real del **substream** de baja resolución — el stream más pequeño es el que quieres para la inferencia de IA continua, dejando libre el mainstream de resolución completa para los clips de evidencia. `cv2.VideoCapture(uri)` luego extrae fotogramas decodificados usando FFmpeg/GStreamer por debajo. Se requiere un bucle delgado de reconexión (capturar el `.read()` fallido, esperar, reintentar `VideoCapture`) porque el RTSP sobre una LAN ocasionalmente perderá una sesión TCP; sin él, un tropiezo de Wi‑Fi o un reinicio de cámara mata silenciosamente todo el pipeline.

```python
import os
from onvif import ONVIFCamera
cam = ONVIFCamera(
    os.environ['CAMERA_HOST'],
    80,
    os.environ['CAMERA_USER'],
    os.environ['CAMERA_PASSWORD'],
)
media = cam.create_media_service()
profiles = media.GetProfiles()
uri = media.GetStreamUri({
    'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
    'ProfileToken': profiles[1].token  # perfil de substream
}).Uri
```

**Corrección v10 — no codifiques de forma fija las credenciales de cámara en el código fuente.** El ejemplo anterior previamente escribía el usuario y la contraseña de ONVIF directamente en el código. En su lugar, léelos desde variables de entorno (o un mecanismo de secretos equivalente), ya que este archivo es del tipo que termina siendo confirmado (committed), pegado en un reporte, o compartido para depuración — y el modo de falla es una credencial de cámara filtrada, no solo una llave de API filtrada.

**Corrección (v2) — no dejes que `VideoCapture` se vuelva obsoleto silenciosamente.** `cv2.VideoCapture` almacena fotogramas en un búfer internamente (mediante FFmpeg/GStreamer), y si tu bucle de procesamiento alguna vez se atrasa respecto a la tasa de fotogramas real de la cámara, no descarta fotogramas — sirve fotogramas cada vez más viejos mientras `.read()` sigue retornando exitosamente. Nada da error, así que el sistema *parece* saludable mientras la latencia de alerta de extremo a extremo crece silenciosamente sin límite. Corrige esto con un hilo de ingesta dedicado que lee continuamente fotogramas hacia un "buzón" (mailbox) de una sola ranura (cada fotograma nuevo sobrescribe al anterior), de modo que el bucle de procesamiento siempre consuma el fotograma disponible más reciente y los obsoletos se descarten en lugar de encolarse. Donde el backend lo respeta, `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)` también ayuda, pero no está soportado de forma confiable en todas las compilaciones de FFmpeg, así que trata el patrón de buzón con hilos como la corrección confiable. Adjunta una **marca de tiempo de captura** a cada fotograma en el instante en que se lee (no cuando se procesa después) — todo evento río abajo debe llevar este tiempo de captura, ya que de lo contrario un pipeline rezagado representaría incorrectamente cuándo ocurrió realmente algo, lo cual destruye el valor forense. Ver §1.10 para el presupuesto de latencia completo y la disciplina de marcas de tiempo que esto alimenta.

**Configuración del lado de la cámara (v3).** Dos ajustes en la propia interfaz web de la cámara importan al software y son fáciles de olvidar porque viven fuera del repositorio:

- **Intervalo de I-frame / GOP → ~1 segundo.** El búfer de pre-grabación (§1.12) solo puede cortar un clip en un keyframe. Si la cámara emite un I-frame cada 4 segundos, tu "pre-grabación de 6 segundos" en realidad es "en algún punto entre 4 y 8 segundos, no decodificable al inicio".
- **La resolución y la tasa de fotogramas del substream** deben registrarse en `configs/` junto con el presupuesto de píxeles de §0, porque tanto el factor de escala del letterbox como el presupuesto de latencia se derivan de ellos.

**Lecturas adicionales:**
- [python-onvif-zeep on GitHub](https://github.com/FalkTannhaeuser/python-onvif-zeep) — la biblioteca cliente de ONVIF usada en este proyecto
- [ONVIF Core Specification](https://www.onvif.org/specs/core/ONVIF-Core-Specification.pdf) — el estándar subyacente, si quieres ver exactamente qué retorna `GetStreamUri` y por qué
- [OpenCV `VideoCapture` documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html) — la clase que hace la decodificación RTSP real
- [PyImageSearch — Faster video file FPS with cv2.VideoCapture and OpenCV](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/) — el patrón de lector con hilos que corrige el smear del buffering
- [Stack Overflow — OpenCV VideoCapture lag due to the capture buffer](https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer) — una explicación concisa de exactamente este modo de falla

---

### 1.2 Puerta de movimiento previa a la inferencia — `src/ingest/motion_gate.py`

**Propósito.** El presupuesto de 40 TOPS del Hailo‑10H es finito, y una cámara de seguridad estática pasa la abrumadora mayoría de su tiempo observando una escena sin cambios. Ejecutar YOLOv8n en cada fotograma decodificado desperdicia el rendimiento del NPU en pasillos vacíos y además invita falsos positivos por artefactos de compresión en un fotograma que, por lo demás, está inactivo. La puerta de movimiento es un pre‑filtro barato que solo usa CPU: únicamente los fotogramas con cambio real a nivel de píxel llegan al acelerador.

**Corrección (v2) — la diferenciación simple de fotogramas no sobrevive a una cámara exterior real.** El diseño original comparaba cada fotograma solo con el fotograma *anterior* (\(|G_t - G_{t-1}|\)). Eso no tiene ninguna noción persistente de "fondo", por lo que falla de tres formas que una cámara 24/7 encontrará constantemente: una persona caminando directamente hacia el lente cambia muy pocos píxeles de un fotograma a otro y puede pasar sin detección; árboles que se mecen, sombras en movimiento e insectos iluminados por IR en la noche cambian píxeles en cada fotograma y disparan movimiento falso constante; y cualquier cambio gradual de iluminación (nubes, atardecer) es indistinguible de movimiento real. La solución es un **modelo de fondo** adecuado en lugar de una comparación de dos fotogramas.

**Cómo funciona.** Reemplaza la diferencia de dos fotogramas con `BackgroundSubtractorMOG2` de OpenCV — un Modelo de Mezcla Gaussiana (GMM) por píxel que se actualiza continuamente con una tasa de aprendizaje, de modo que puede representar *múltiples* estados de fondo por píxel (por ejemplo, una rama que se mece entre dos posiciones cuenta como fondo en ambas, no como movimiento) y se adapta gradualmente a la deriva de iluminación en lugar de reaccionar a ella como primer plano. La llamada `apply()` de MOG2 devuelve una máscara de primer plano directamente — ya no calculas a mano un paso explícito de `absdiff`/umbral.

La segunda corrección es **dónde** mides el movimiento: calcular la proporción de movimiento sobre el fotograma *completo* desperdicia sensibilidad en regiones de fondo (el borde de una entrada de auto, el árbol de un vecino) que nunca contendrán un evento de zona real. En su lugar, deriva una máscara estática de región de interés \(Z(x,y)\) a partir de los polígonos de zona configurados más un margen de seguridad (para que los objetos sean detectados al acercarse a una zona, no solo cuando ya están dentro de ella), y restringe la proporción de movimiento a esa región:

\[
r_t = \frac{\sum_{x,y} F_t(x,y)\, Z(x,y)}{\sum_{x,y} Z(x,y)}
\]

Envía el fotograma a YOLOv8n cuando \(r_t \geq \tau_{\text{motion}}\). Un paso de apertura/dilatación morfológica sobre \(F_t\) antes de sumar sigue ayudando a fusionar píxeles de ruido dispersos en manchas coherentes. Dos reglas siguen siendo importantes en la práctica: mantener la inferencia corriendo al ritmo del tracker mientras un track ya está activo (para que una persona que se detiene a mitad de fotograma no desaparezca silenciosamente del pipeline de eventos), y forzar una inferencia de "latido" (heartbeat) cada pocos segundos sin importar el movimiento, para que una transmisión de cámara congelada se vea diferente de una que está inactiva pero sana.

**Corrección v3 — la máscara de MOG2 tiene tres valores, y `> 0` cuenta las sombras como movimiento.** Con `detectShadows=True`, `apply()` **no** devuelve una máscara binaria. Devuelve **0 para fondo, 255 para primer plano y 127 para sombra detectada**. El snippet de v2 probaba `fg_mask > 0`, lo cual clasifica cada píxel de sombra como movimiento — es decir, pagas el costo de CPU de la detección de sombras y luego descartas todo su beneficio. Dado que las sombras en movimiento sobre una entrada de auto a última hora de la tarde son exactamente una de las fuentes de falsos positivos que esta sección existe para suprimir, este único operador de comparación anula toda la mejora de MOG2. **Prueba `== 255`.**

Un detalle relacionado: el detector de sombras de MOG2 usa información cromática, así que alimentarlo con un **fotograma BGR en lugar de uno en escala de grises** da una supresión de sombras notablemente mejor. Cuesta aproximadamente 3× el ancho de banda de memoria en esa etapa, así que compara ambos contra tu presupuesto de latencia (§1.10) antes de decidirte.

**Adición v5 — la elección entre BGR y escala de grises es una configuración de modo diurno, y los insectos atraídos por IR necesitan su propia solución, no un arreglo del modelo de fondo.** La mayoría de las cámaras domésticas PoE cambian a video monocromático iluminado por IR después del anochecer (§1.3): cada canal lee R=G=B, así que la discriminación cromática de sombras de MOG2 no tiene nada con qué trabajar, y `input_color_space: bgr` no aporta nada de noche — trátalo como una configuración de `día` y recurre a escala de grises (o simplemente acepta que la elección BGR/escala de grises es irrelevante) una vez que el modo IR está activo, en lugar de suponer que un solo formato de entrada cubre ambos casos. Por separado, los insectos atraídos por el iluminador IR a corta distancia del lente son una fuente de falsos positivos distinta de los casos de árboles que se mecen/sombras en movimiento para los que la corrección v2 anterior nombró a MOG2 como solución: un modelo de fondo GMM solo aprende movimiento que se repite en un número pequeño de estados estables, pero la trayectoria de vuelo errática y no repetitiva de un insecto frente al lente nunca se asienta como "fondo", y como ocurre directamente en el campo de visión de la cámara, típicamente también cae dentro del margen de la ROI de zona, así que `roi_source: zone_polygons_plus_margin` tampoco lo filtra. La solución está río abajo de la puerta: deja pasar estos fotogramas al detector según lo diseñado, y confía en el filtro `classes: [person]` más el `min_confirmed_hits` (§1.4) del tracker para rechazar el ruido resultante en lugar de intentar suprimirlo en la etapa de la puerta de movimiento. Agrega un caso de prueba negativo explícito `night_ir` al banco de reproducción de §1.11 para que una regresión que deje pasar fotogramas disparados por insectos como envíos sostenidos al NPU aparezca como un número medido de falsos positivos/rendimiento, no como un Pi misteriosamente ocupado a las 2 a.m.

**Corrección v3 — `mog2_history` es un conteo de fotogramas, y el arreglo de Δt nunca se aplicó aquí.** §1.4 reemplaza correctamente `max_missed_frames` con una expiración de reloj de pared porque la puerta de movimiento hace que la llegada de fotogramas sea irregular. Pero `mog2_history: 500` tiene exactamente el mismo defecto y se dejó sin corregir: es un conteo de *fotogramas*, y MOG2 deriva de él su tasa de aprendizaje automática. Bajo el buzón de una sola ranura (§1.10), los fotogramas se descartan deliberadamente cada vez que el procesamiento se atrasa, así que 500 fotogramas es una cantidad desconocida y dependiente de la carga de tiempo de reloj de pared. La consecuencia es que el modelo de fondo se adapta más rápido o más lento dependiendo de cuán ocupado esté el Pi, lo cual cambia silenciosamente la sensibilidad de la puerta.

Solución: pasar una **`learningRate` explícita** a `apply()`, calculada a partir del tiempo transcurrido medido desde la última llamada, en lugar de dejar que OpenCV la derive de un conteo de fotogramas:

\[
\alpha = \min\left(1,\ \frac{\Delta t}{T_{\text{adapt}}}\right)
\]

donde \(T_{\text{adapt}}\) es la constante de tiempo de adaptación de fondo prevista, en segundos. Este es el mismo principio que el filtro de Kalman parametrizado en el tiempo de §1.4, aplicado de forma consistente.

```python
now = time.monotonic()
dt = now - last_gate_ts
learning_rate = min(1.0, dt / config.background_adaptation_seconds)

fg_mask = bg_subtractor.apply(bgr_frame, learningRate=learning_rate)
fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

# 255 = foreground, 127 = shadow, 0 = background.
# Test == 255, NOT > 0, or shadows are counted as motion.
motion_ratio = (fg_mask[zone_mask > 0] == 255).mean()

should_run_npu = (
    motion_ratio >= config.minimum_motion_ratio
    or has_active_tracks
    or now - last_inference_time >= config.maximum_idle_inference_interval_seconds
)
last_gate_ts = now
```

```yaml
# config/motion_gate.yaml (v3)
motion_gate:
  algorithm: mog2                        # v2: was previous_frame differencing
  input_color_space: bgr                 # v3: BGR improves MOG2 shadow discrimination vs. grayscale
  background_adaptation_seconds: 50.0    # v3: replaces mog2_history (a frame count) — learningRate = dt / this
  mog2_var_threshold: 16
  detect_shadows: true
  shadow_pixel_value: 127                # v3: documented so the == 255 test is never "simplified" back to > 0
  roi_source: zone_polygons_plus_margin  # restrict r_t to this mask, not the full frame
  roi_margin_px: 40
  minimum_motion_ratio: 0.01             # recalibrate against the ROI-only baseline via the §1.11 harness
  maximum_idle_inference_interval_seconds: 5.0
```

**Nota sobre la cláusula `has_active_tracks`.** Mientras cualquier track esté vivo, la puerta se omite y el NPU se ejecuta en cada fotograma. Eso es intencional y correcto, pero significa que tu peor caso de carga del NPU es el 100% de la tasa de fotogramas, no el promedio filtrado por la puerta. Presupuesta §1.10 para el caso sin filtro; la puerta es una optimización de energía y térmica, no un supuesto de capacidad.

**Deeper reading:**
- [OpenCV `BackgroundSubtractorMOG2` reference](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html) — incluyendo la firma `apply(image, fgmask, learningRate)` y la semántica de máscara 0/127/255
- [OpenCV background subtraction tutorial](https://docs.opencv.org/4.x/de/df4/tutorial_js_bg_subtraction.html) — panorama conceptual de MOG2 frente a métodos más simples
- [PyImageSearch — Basic motion detection and tracking with Python and OpenCV](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/) — contexto sobre el enfoque más simple de diferenciación de fotogramas y por qué es un punto de partida razonable pero no el diseño final

---

### 1.3 Detector YOLOv8n en el Hailo‑10H — `src/inference/detector.py`

**Propósito.** Esta es la etapa real de "qué hay en este fotograma" — el detector de objetos que convierte un fotograma filtrado por movimiento en una lista de cajas delimitadoras con etiquetas de clase y puntajes de confianza. YOLOv8n (Nano) es el modelo fijado para el MVP porque tiene una ruta madura y precompilada hacia hardware Hailo; YOLO26 no la tiene.

**Cómo funciona.** Los dispositivos Hailo no ejecutan ONNX ni PyTorch directamente — ejecutan un binario **HEF** (Hailo Executable Format) producido por el Hailo Dataflow Compiler, el cual cuantiza y compila un modelo entrenado para una arquitectura de chip específica (un HEF construido para Hailo‑8/8L no es intercambiable con un Hailo‑10H). El Hailo Model Zoo publica configuraciones de YOLOv8 y ya sea distribuye o documenta cómo obtener HEFs para los dispositivos objetivo soportados, y los datos de benchmark públicos del Model Zoo son la fuente confiable para el FPS/precisión esperados en este tamaño de modelo. En el momento de la inferencia, el código Python carga el `.hef` compilado, le alimenta un fotograma preprocesado de 640×640 a través de HailoRT, y recibe las salidas crudas del detector, las cuales luego pasan por supresión de no‑máximos (non-max suppression) usando el `nms_iou_threshold` configurado para eliminar cajas duplicadas/superpuestas del mismo objeto.

**Corrección (v2) — el detector no debe pre‑filtrar lo que el tracker necesita.** La configuración v1 fijaba `confidence_threshold: 0.45` en el detector mientras que la asociación de dos niveles del tracker (§1.4) espera que exista un nivel `low_confidence_threshold: 0.15`. Si el detector descarta cada caja por debajo de 0.45 antes de que el tracker la vea siquiera, el nivel de baja confianza no recibe nada — la asociación de dos niveles se vuelve imposible, no solo degradada. **El detector debe emitir al umbral bajo del tracker (0.15)** y dejar que el tracker (y, en última instancia, el `minimum_mean_confidence` de la máquina de estados de eventos) haga la clasificación por niveles. Cualquier corte más estricto pertenece río abajo, donde hay identidad y contexto temporal con los que razonar, no en el punto donde las cajas se descartan de forma irreversible.

**Corrección (v2) — letterbox, no estirar.** Es muy probable que el substream de tu cámara sea 16:9, pero la entrada del modelo es un cuadrado de 640×640. Redimensionar de forma naíf (estirando) un fotograma 16:9 a un cuadrado distorsiona la relación de aspecto con la que se entrenó el modelo (imágenes de COCO), lo cual reduce de forma medible el mAP — particularmente en personas, cuyas cajas delimitadoras son sensibles a la relación de aspecto. Usa en su lugar el preprocesamiento de **letterbox**: escala el fotograma para que quepa dentro de 640×640 preservando la relación de aspecto, y luego rellena el espacio restante (típicamente con gris neutro, 114/114/114) en lugar de estirar. Almacena el factor de escala y los desplazamientos de relleno usados para este fotograma, y aplica el **inverso** de esa misma transformación afín a cada caja devuelta antes de que llegue al tracker — de lo contrario tus cajas delimitadoras serán correctas en el "espacio de entrada del modelo" pero incorrectas en el espacio de coordenadas del fotograma original, lo cual corrompe silenciosamente la geometría de zona/línea río abajo.

Nota la interacción con §0: en un substream de 640×360 el factor de escala del letterbox es 1.0 (solo se agrega relleno vertical), así que la resolución del substream *es* la resolución efectiva de detección. Elevar el substream a 720p reduce el factor de escala a la mitad y produce la misma altura en píxeles del objetivo — por eso recortar‑e‑inferir (crop-and-infer), no un substream más grande, es la palanca a usar si el presupuesto de píxeles resulta insuficiente.

```yaml
# config/models.yaml (v3)
detector:
  name: yolov8n
  hef_path: models/yolov8n_h10h.hef
  input_size: [640, 640]
  preprocessing: letterbox        # v2: was naive stretch-to-square resize
  pad_value: [114, 114, 114]
  confidence_threshold: 0.15      # v2: was 0.45 — must match tracker.low_confidence_threshold
  nms_iou_threshold: 0.50
  classes: [person]
  on_device_nms: check            # v3: prefer an NMS-on-chip HEF variant if available; CPU-side NMS costs latency
  failure_policy: fail_loud       # v3: see "NPU failure handling" below
```

Trata `confidence_threshold` puramente como el *piso de emisión* del detector, no como una perilla de calibración — el tracker y la máquina de estados de eventos son los propietarios de la lógica real de confirmación, así que este valor debe mantenerse bajo y sesgado hacia el recall.

**Paso faltante (v2) — verifica la precisión INT8 antes de confiar en YOLOv8n.** El Dataflow Compiler cuantiza el modelo a INT8 para el Hailo‑10H, y la cuantización degrada la precisión de forma desproporcionada en modelos ya pequeños y en objetivos pequeños/distantes — exactamente el caso (una persona lejos de una cámara de puerta) que más le importa a este proyecto. Antes de comprometerte con YOLOv8n, corre el mismo conjunto de fotogramas grabados tanto por el HEF compilado como por el modelo original FP32 de Ultralytics, y compara las detecciones (cajas pequeñas perdidas, deriva de confianza, error de localización). Si la brecha de INT8 es inaceptable en el material real de tu cámara, esa es la señal para escalar a YOLOv8s en lugar de descubrirlo después como falsos negativos inexplicados en producción. Combina esta verificación con el presupuesto de píxeles de §0: la degradación de INT8 en objetos pequeños y la altura de píxeles marginal se agravan mutuamente, y ambas juntas determinan el modelo.

**Corrección v4 — la verificación INT8-contra-FP32 valida la cuantización, no el dominio de despliegue.** Pasar la verificación anterior solo demuestra que el HEF compilado es fiel a los pesos *originales* de Ultralytics — no dice nada sobre si esos pesos son buenos para la escena específica de este proyecto. Los artefactos publicados del model-zoo de Hailo, y la propia herramienta de exportación a Hailo de Ultralytics por defecto, están optimizados/calibrados contra COCO2017 ([Hailo Model Zoo — DATA.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/DATA.rst), [Ultralytics Hailo export guide](https://docs.ultralytics.com/integrations/hailo)), que recomienda al menos 1,024 imágenes de calibración *representativas* para un dominio personalizado. Las imágenes de personas de COCO son diurnas, bien iluminadas, a nivel del suelo, sin oclusiones, y nada parecidas a una cámara fija oblicua en una entrada a 15–25 m. Este es un modo de falla distinto al error de cuantización y no aparecerá en la verificación anterior — solo saldrá a la luz una vez que el banco de reproducción de §1.11 comience a reportar un recall más bajo del que predecía la aritmética del presupuesto de píxeles, momento en el cual es fácil diagnosticarlo erróneamente como un problema de umbral en lugar de un problema de dominio. Una vez que exista el conjunto de datos etiquetado (hito 3), calcula el recall crudo del detector por fotograma como su propio número, separado de la precisión/recall a nivel de evento, para que una brecha de dominio sistemática sea visible directamente en lugar de quedar enterrada dentro de las métricas del tracker/máquina de estados.

**Corrección v4 — el IR nocturno es, arquitectónicamente, el caso menos probado y de mayor valor.** La mayoría de las cámaras domésticas PoE cambian a video monocromático iluminado por IR de noche — un dominio aún más alejado del RGB diurno de COCO. Una revisión sistemática que compara YOLOv8 hasta YOLOv11 en el conjunto de datos de referencia de poca luz ExDark confirma que esto cuesta precisión real — la degradación se mantiene a lo largo de toda la familia de modelos, no solo en una versión ([systematic review of low-light detection, YOLOv8–v11 on ExDark](https://link.springer.com/article/10.1007/s42452-025-08051-5)). La aritmética del presupuesto de píxeles de §0 asume implícitamente video diurno en color; las alertas de mayor valor de una cámara de seguridad de puerta (alguien acercándose después del anochecer) son precisamente las que este proyecto aún no ha medido. Agrega una división explícita `day` / `night_ir` a `eval.yaml` (§1.11) en lugar de tratar el material de crepúsculo/amanecer como cobertura suficiente, y prepárate para que `confidence_threshold` necesite un valor condicional de día/noche en lugar de una sola constante global.

**Corrección v6 — el benchmark nocturno citado muestra un techo, no una línea base de pesos-COCO.** Un modelo de la familia YOLOv8 evaluado específicamente en material de vigilancia nocturna de objetos pequeños reporta Precisión 0.908 / Recall 0.819 / mAP@50 0.886 contra su conjunto de prueba de poca luz/noche ([YOLOv8 nighttime small-object surveillance benchmark](https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140611)) — pero ese modelo fue *entrenado* para la noche, y su mAP@50 está por encima de lo que YOLOv8n alcanza en COCO (~0.52), así que no puede leerse como evidencia de que los pesos calibrados en COCO de este proyecto se degradarán de noche; muestra lo que un modelo ajustado a un dominio logra una vez que ha sido entrenado en el dominio correcto. La revisión sistemática de ExDark mencionada arriba es la cita que realmente respalda la afirmación de degradación para pesos fuera de dominio (solo COCO). Lee el benchmark nocturno más bien como el objetivo que este proyecto puede alcanzar, no como la línea base desde la que parte. La división de etiqueta `night_ir` del hito 3 es a nivel de evento, así que solo puede revelar *si* existe una brecha de dominio nocturno a través del recall del detector por fotograma que ahora alimenta (§1.11) — no puede en sí misma usarse para ajustar finamente (fine-tune) el detector, ya que el ajuste fino necesita cajas delimitadoras por fotograma, no marcas de tiempo de eventos, y producir esas cajas es un esfuerzo de anotación considerablemente mayor que etiquetar ventanas de evento. Cerrar una brecha confirmada es, por lo tanto, un proyecto de fase 2 delimitado, no una continuación directa del hito 3: un paso separado de anotación de cajas delimitadoras, un reentrenamiento, y una nueva compilación de HEF a través del Hailo Dataflow Compiler usando un conjunto de calibración específico del dominio (el mismo mínimo de ~1,024 imágenes representativas que §1.3 ya cita para la deriva de COCO a la escena). Ese reentrenamiento se ejecuta a través de las herramientas de entrenamiento de Ultralytics, así que hereda la misma preocupación de licenciamiento AGPL-3.0 que la nota de licenciamiento de esta sección ya señala — vale la pena decidirlo antes de comprometerse con el ajuste fino, no solo antes de publicar el repositorio.

**Adición v10 — el reentrenamiento de fase 2 necesita una ruta de reversión y validación en modo sombra antes de la promoción, no solo una receta para producir un nuevo HEF.** La corrección v6 anterior describe cómo producir un nuevo HEF; no dice qué pasa si ese HEF resulta peor que el que ya está desplegado. Valida el modelo reentrenado contra la misma partición reservada (§1.11) usada para el YOLOv8n original antes de promoverlo — reporta precisión/recall/mAP en esa partición, no solo métricas del tiempo de entrenamiento, y trata un resultado peor en la partición reservada como una razón para conservar el HEF actual. Mantén el HEF anterior y sus valores exactos de `models.yaml` accesibles para que una regresión pueda revertirse cambiando un valor de configuración y una ruta de archivo, en lugar de volver a ejecutar el proyecto de fase 2 desde cero.

**Adición v5 — la brecha de IR nocturno tiene tres consecuencias concretas río abajo, no solo una advertencia para probarla.** Primero, el video iluminado por IR es monocromático (R=G=B en cada píxel), lo cual vuelve irrelevante la elección de BGR frente a escala de grises para la discriminación de sombras de MOG2 de noche y reabre a los insectos atraídos por IR como una fuente de falsos positivos distinta que la puerta de movimiento sola no puede filtrar — ver §1.2 para ambos casos. Segundo, el alcance del iluminador IR en esta clase de cámara típicamente es de 20–30 m sin importar la distancia focal del lente, lo cual limita el límite efectivo de la zona nocturna independientemente de la geometría de altura de píxeles de §0 — ver §0 para la variante específica de noche de ese cálculo.

**Adición v3 — manejo de fallas del NPU.** Nada en v1 ni v2 dice qué hace el pipeline cuando HailoRT lanza una excepción, el dispositivo se reinicia, o aparece una discordancia de firmware/driver después de un `apt upgrade`. Para un sistema de vigilancia, el peor modo de falla posible es aquel que cree que está observando cuando no lo está, así que la política debe ser explícita:

- **Al iniciar:** verifica el dispositivo y carga el HEF antes de que el hilo de ingestión inicie. Si cualquiera falla, niégate a iniciar en lugar de correr un pipeline que decodifica fotogramas que nunca analizará. Esto coincide con la filosofía de fallo temprano ya usada en otras partes de este proyecto.
- **Durante la ejecución:** captura las excepciones de HailoRT en el punto de la llamada de inferencia, registra en nivel ERROR, incrementa un contador `npu_failures_total` en `/metrics` (§1.10), y marca `/health` como **degradado** — nunca continúes silenciosamente.
- **Recuperación:** intenta una reinicialización acotada (unos pocos reintentos con backoff). Si la reinicialización falla, mantén viva la ingestión y el búfer de pre‑grabación (para que la evidencia siga siendo registrada) pero reporta el pipeline como caído. Grabar sin analizar es un estado degradado que vale la pena preservar; fingir que se analiza no lo es.

**Nota de licenciamiento.** Ultralytics YOLOv8 se distribuye bajo **AGPL‑3.0**. Eso es sin restricciones para uso privado y no distribuido exactamente como este proyecto, pero si el código llega a publicarse alguna vez como una pieza de portafolio de código abierto con código derivado construido sobre las herramientas de entrenamiento/exportación de Ultralytics, los términos de copyleft de AGPL se extienden hacia afuera a menos que se compre una licencia Ultralytics Enterprise. Decide esto temprano — no bloquea el MVP, pero sí restringe cómo se puede compartir el repositorio más adelante.

**Deeper reading:**
- [Hailo Model Zoo on GitHub](https://github.com/hailo-ai/hailo_model_zoo) — generación de HEF, números de benchmark, y configuraciones de modelo soportadas, incluyendo [`yolov8n.yaml`](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolov8n.yaml)
- [Hailo RPi5 Examples — object detection pipeline](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/basic-pipelines.md) — un pipeline de referencia funcional de GStreamer/Python exactamente para esta combinación de hardware
- [Hailo Application Code Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python) — ejemplos de inferencia HailoRT en Python puro (sin GStreamer), más cercanos a lo que necesita un pipeline embebido en FastAPI
- [Raspberry Pi AI accelerator documentation](https://www.raspberrypi.com/documentation/computers/ai.html) — el paquete oficial `hailo-h10-all` y la documentación del pipeline de modelos
- [Ultralytics — Hailo export integration](https://docs.ultralytics.com/integrations/hailo) — la ruta oficial de exportación y dónde verificar el comportamiento del HEF INT8 contra el modelo FP32
- [Ultralytics `data.augment` API reference (LetterBox)](https://docs.ultralytics.com/reference/data/augment) — la implementación de referencia de letterbox sobre la cual modelar tu propio preprocesamiento
- [Ultralytics License page](https://www.ultralytics.com/license) — los términos de licenciamiento AGPL-3.0 frente a Enterprise
- [Hailo Model Zoo — DATA.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/DATA.rst) — confirma COCO2017 como el conjunto de calibración/evaluación detrás de los artefactos del model-zoo distribuidos (v4)
- [YOLOv8 nighttime small-object surveillance benchmark](https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140611) — números concretos publicados de precisión/recall/mAP para detección en poca luz/nocturna (v5)
- [Systematic review of low-light object detection — YOLOv8–v11 on ExDark](https://link.springer.com/article/10.1007/s42452-025-08051-5) — confirma que la brecha de precisión en poca luz se mantiene en toda la familia de modelos YOLO (v5)

---

### 1.4 Tracker multiobjeto (Kalman + IoU) — `src/inference/tracker.py`

**Propósito.** Una caja de detección cruda no tiene memoria — puede parpadear de un fotograma a otro por oclusión, desenfoque de movimiento, o un fotograma saltado por la puerta de movimiento, y la vieja regla de "3 fotogramas consecutivos por encima de 0.60 de confianza" fallaba exactamente ahí. El trabajo de un tracker es darle a cada objeto detectado una identidad persistente (`track_id`) a través de los fotogramas, de modo que la lógica de eventos río abajo pueda razonar sobre *objetos estables*, no cajas ruidosas.

**Cómo funciona.** Este es un diseño de **tracking‑by‑detection**, de la misma familia que SORT y ByteTrack: el detector propone cajas en cada fotograma; el único trabajo del tracker es decidir a qué caja pertenece a qué track existente (o si es uno nuevo). Dos mecanismos hacen el trabajo:

1. **Predicción con filtro de Kalman** — cada track mantiene un vector de estado que estima posición, tamaño y velocidad, y predice dónde debería estar el objeto *antes* de ver las detecciones del siguiente fotograma:
   \[
   x_t = [c_x, c_y, w, h, v_x, v_y, v_w, v_h]^T
   \]
   Esto permite que un track sobreviva uno o dos fotogramas sin una detección coincidente (por ejemplo, una oclusión breve o un salto de la puerta de movimiento) porque el filtro sigue extrapolando el movimiento en lugar de que el track simplemente desaparezca.

2. **Asociación basada en IoU** — una caja de track predicha \(b_i\) se relaciona con una detección real \(d_j\) usando Intersección‑sobre‑Unión:
   \[
   \operatorname{IoU}(b_i, d_j) = \frac{\operatorname{area}(b_i \cap d_j)}{\operatorname{area}(b_i \cup d_j)}
   \]
   Una coincidencia se acepta solo cuando \(\operatorname{IoU}(b_i, d_j) \geq \tau_{\text{IoU}}\) (y opcionalmente una verificación de distancia de centroide para mayor robustez). Las detecciones sin coincidencia originan nuevos tracks tentativos; las predicciones sin coincidencia incrementan el "tiempo perdido" de un track, y un track se descarta una vez que excede la ventana de expiración.

La contribución específica de ByteTrack sobre un tracker de estilo SORT sencillo es asociar **cada** caja de detección, incluidas las de baja confianza, en lugar de descartarlas antes de la asociación — las cajas de puntaje bajo siguen siendo útiles para mantener viva a través de una oclusión parcial un track ya confirmado, aunque serían demasiado débiles para *iniciar* un nuevo track por sí solas ([FoundationVision/ByteTrack](https://github.com/FoundationVision/ByteTrack)). Esto solo funciona de extremo a extremo ahora que el detector (§1.3) emite a 0.15 en lugar de descartar ese nivel antes de que llegue al tracker.

**Corrección (v2) — el filtro de Kalman debe estar parametrizado en el tiempo, no en fotogramas.** Un filtro de Kalman estándar de estilo SORT asume un \(\Delta t = 1\) constante entre actualizaciones — algo aceptable cuando los fotogramas llegan con un reloj fijo, pero la puerta de movimiento (§1.2) deliberadamente hace irregular la llegada de fotogramas: un fotograma podría seguir al anterior por 100 ms, o por 4 segundos si nada se movió y solo se disparó el latido. Alimentar a un filtro de \(\Delta t\) fijo con tiempo real irregular hace que sobre‑ o subextrapole gravemente la posición de un objeto, rompiendo la propia asociación por IoU que se supone que debe soportar. La matriz de transición de estado \(F\) y la covarianza del ruido de proceso \(Q\) deben escalar ambas con el **tiempo real transcurrido en reloj de pared** desde la última actualización del track:

\[
F(\Delta t) = \begin{bmatrix} 1 & 0 & 0 & 0 & \Delta t & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & \Delta t & 0 & 0 \\ 0 & 0 & 1 & 0 & 0 & 0 & \Delta t & 0 \\ 0 & 0 & 0 & 1 & 0 & 0 & 0 & \Delta t \\ 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \end{bmatrix}
\]

En la práctica esto significa recalcular \(F\) (y reescalar \(Q\)) a partir del \(\Delta t\) medido en cada paso de predicción, usando la propia marca de tiempo de la última actualización del track en lugar de asumir un intervalo de fotograma fijo — la mayoría de las bibliotecas de Kalman (incluyendo `filterpy`) soportan esto reconstruyendo `F`/`Q` en cada llamada en lugar de una sola vez al inicializar.

**Refinamiento v3 — \(Q(\Delta t) \propto \Delta t^2\) es una aproximación, y la forma exacta importa para brechas largas.** v2 planteaba el escalamiento como \(Q(\Delta t) \propto \Delta t^2\), lo cual es conveniente pero no físicamente correcto. Bajo un modelo continuo de aceleración por ruido blanco, la incertidumbre en la *velocidad* crece linealmente con el tiempo transcurrido, pero la incertidumbre en la *posición* crece con el **cubo** del tiempo transcurrido — porque el error de posición acumula la integral de un error de velocidad que ya está creciendo. Para cada par independiente de posición/velocidad, el bloque correcto es:

\[
Q = \begin{bmatrix} \tfrac{1}{3}\Delta t^{3} & \tfrac{1}{2}\Delta t^{2} \\[2pt] \tfrac{1}{2}\Delta t^{2} & \Delta t \end{bmatrix} \sigma_a^{2}
\]

donde \(\sigma_a^2\) es la varianza de la aceleración del objeto — físicamente, cuán abruptamente esperas que una persona cambie de velocidad o dirección. Los términos fuera de la diagonal \(\tfrac{1}{2}\Delta t^2\) codifican la correlación entre el error de posición y el de velocidad, la cual el escalamiento naíf con \(\Delta t^2\) descarta por completo.

Por qué esto importa aquí específicamente: la puerta de movimiento puede producir valores de \(\Delta t\) de varios segundos. A \(\Delta t = 3\) s, el término cúbico es un orden de magnitud mayor que la aproximación cuadrática, así que la forma naíf deja al filtro **sobreconfiado** sobre dónde está una persona después de una brecha larga. Una predicción sobreconfiada produce una puerta demasiado estrecha, la asociación por IoU falla, y el track se pierde y se vuelve a crear con un nuevo `track_id` — que es precisamente la ruptura de identidad que el tracker existe para prevenir. Construye \(Q\) a partir del bloque anterior, y trata \(\sigma_a^2\) como un ajustable calibrado a través del banco de §1.11 en lugar de una constante adivinada una sola vez. Derivación completa y la elección discreto‑vs‑continuo: §4.1.

**Corrección (v2) — la expiración del track debe medirse en segundos, no en fotogramas.** `max_missed_frames: 12` no tiene sentido una vez que los fotogramas llegan a una tasa variable: 12 fotogramas *saltados* podrían abarcar medio segundo de inactividad real o, bajo un filtrado de movimiento agresivo, decenas de segundos — en este último caso el track expira mucho más tarde de lo previsto y las identidades obsoletas persisten. Reemplázalo con una expiración de reloj de pared, `max_missed_seconds`, evaluada contra el mismo reloj monótono usado para \(\Delta t\).

**Corrección (v2) — una única fuente de verdad para la confirmación.** v1 definía aquí `min_confirmed_hits: 3` *y* `required_track_hits: 3` en `events.yaml` — dos nombres para el mismo concepto, garantizados a divergir en el momento en que una configuración cambie y la otra no. **El tracker es el propietario exclusivo de esta variable.** Expone un booleano `track.is_confirmed` (verdadero una vez que se han acumulado `min_confirmed_hits` coincidencias exitosas), y la máquina de estados de eventos (§1.5) simplemente lee esa bandera en lugar de volver a contar coincidencias por sí misma.

```yaml
# config/tracker.yaml (v3)
tracker:
  algorithm: kalman_iou
  min_iou_match: 0.25
  max_centroid_distance_px: 120    # see §4.2 — depth-dependent; replace with metric distance in phase 2
  min_confirmed_hits: 3            # single source of truth — exposed downstream as track.is_confirmed
  max_missed_seconds: 1.5          # v2: was max_missed_frames: 12 — now wall-clock, not frame-count
  max_track_age_seconds: 3.0
  kalman_time_parameterized: true  # F(Δt) rebuilt from measured elapsed time each predict step
  kalman_q_model: continuous_white_noise   # v3: full (1/3 Δt³, 1/2 Δt², Δt) block, not the Δt² approximation
  kalman_sigma_accel_sq: 2000.0     # v7: was 30.0 (v6) — arithmetic error corrected, see correction below; units px²/s⁴, tune via the §1.11 harness
  low_confidence_threshold: 0.15   # matches detector.confidence_threshold exactly — no gap in the tiering
  high_confidence_threshold: 0.45
  class_aware_matching: true
  tracked_classes: [person]
```

Un track solo se vuelve `confirmed` (`track.is_confirmed = true`, elegible para disparar eventos) después de `min_confirmed_hits` coincidencias exitosas — esto es lo que reemplaza la vieja regla naíf de conteo de fotogramas con algo que efectivamente modela la identidad a lo largo del tiempo, y ahora es el *único* lugar donde se define este umbral.

**Corrección v6 — `kalman_sigma_accel` estaba nombrado como σ pero se valoraba y usaba como σ².** El nombre del campo implica una desviación estándar, pero el bloque de §4.1 usa \(\sigma_a^2\) directamente, y `Q_continuous_white_noise(..., spectral_density=...)` de `filterpy` espera esa misma cantidad al cuadrado, no su raíz cuadrada — el nombre de v3 era simplemente incorrecto para lo que la configuración contenía. Renombrado a **`kalman_sigma_accel_sq`**, con unidades declaradas explícitamente como **px²/s⁴**: el tracker opera en espacio de píxeles crudo, antes de la proyección de homografía de §4.2, así que esto es la varianza de aceleración en píxeles‑por‑segundo‑al‑cuadrado, elevada al cuadrado de nuevo — no en unidades métricas.

**Corrección v7 — el valor de v6 de `30.0` se derivó de una aritmética que contradecía su propia afirmación, y es aproximadamente dos órdenes de magnitud demasiado pequeño.** v6 afirmaba que una aceleración de píxeles "del orden de decenas de px/s²" implica una varianza "en las decenas de px²/s⁴" — pero elevar al cuadrado una cantidad en las decenas produce centenas a miles, no decenas; ese paso era simplemente incorrecto. Rederivar a partir de la propia tabla de presupuesto de píxeles de §0 da un rango de partida defendible en lugar de uno adivinado. El lente de 4.0 mm de §0 a 15 m ubica a una persona de 1.7 m en ~54 px, una escala de ~54 / 1.7 ≈ 32 px/m; una persona caminando que cambia de ritmo o dirección acelera aproximadamente a 1–2 m/s², así que a esa distancia la aceleración aparente en píxeles es del orden de ~32 × 1.5 ≈ 48 px/s², y \(\sigma_a^2 \approx 48^2 \approx 2{,}300\) px²/s⁴. A la distancia de 5 m del mismo lente, §0 da ~163 px, una escala de ~163 / 1.7 ≈ 96 px/m, así que la misma aceleración física de 1–2 m/s² es ~144 px/s² y \(\sigma_a^2 \approx 144^2 \approx 21{,}000\) px²/s⁴. La escala — y por lo tanto la varianza defendible — depende de la distancia a la cámara, así que trata **10³–10⁴ px²/s⁴** como el rango de partida que la geometría de zona de este proyecto realmente respalda, no una única constante. Fija **`kalman_sigma_accel_sq: 2000.0`** (configuración de §1.4) como punto de partida para que el banco de §1.11 lo ajuste — un valor de rango medio para una persona rastreada a distancia de zona moderada — en lugar del aritméticamente insostenible `30.0`. Un valor tan pequeño reproduce exactamente el fallo de covarianza sobreconfiada que describe §4.1: hace que el filtro esté *más* seguro de lo que debería sobre la posición después de un \(\Delta t\) largo, estrechando la puerta de asociación por IoU y disparando el mismo síntoma de ruptura de identidad/alerta duplicada que motivó en primer lugar la corrección de \(\Delta t^3\) — que es exactamente el modo de falla al que un valor dos órdenes de magnitud demasiado pequeño habría regresado directamente.

**Deeper reading:**
- [ByteTrack (ECCV 2022) on GitHub](https://github.com/FoundationVision/ByteTrack) — la idea de "asociar cada caja de detección" y sus números de precisión reportados en MOT17
- [SORT — Simple Online and Realtime Tracking](https://github.com/abewley/sort) — la línea base más simple de Kalman+IoU a la que este tracker del MVP más se parece
- [PyImageSearch — Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/) — IoU explicado con ejemplos resueltos
- [kalmanfilter.net](https://www.kalmanfilter.net/) — un recorrido accesible y centrado en las matemáticas de cómo funciona el ciclo de predicción/actualización del filtro de Kalman
- [Cross Validated — Kalman smoothing with irregular time steps](https://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps) — las matemáticas para escalar \(F\) y \(Q\) por el tiempo real transcurrido en lugar de un paso fijo
- [filterpy issue — handling variable dt](https://github.com/rlabbe/filterpy/issues/196) — una discusión concreta sobre reconstruir `F`/`Q` en cada actualización en una biblioteca de Kalman de Python ampliamente usada
- [`filterpy.common.Q_continuous_white_noise`](https://filterpy.readthedocs.io/en/latest/common/discretization.html) — una implementación lista para usar del bloque \(\tfrac{1}{3}\Delta t^3\) descrito arriba

---

### 1.5 Máquina de estados de eventos y zonas — `src/events/state_machine.py`

**Propósito.** Decidir, a partir de la trayectoria de un track confirmado, si realmente ocurrió algo que valga la pena alertar — una persona entrando a una zona de entrada (doorstep zone), cruzando una línea límite, o permaneciendo demasiado tiempo — evitando al mismo tiempo alertas duplicadas para la misma situación en curso.

**Cómo funciona.** Cada track avanza a través de una máquina de estados explícita:

```text
NEW → TENTATIVE → CONFIRMED → EVENT_EMITTED (cooldown) → LOST → REMOVED
```

La membresía de zona se evalúa geométricamente por cuadro (prueba de punto dentro de polígono para el centroide del track contra un polígono de zona configurado), y un evento de cruce de línea se detecta comprobando si la trayectoria del centroide del track cruza un segmento definido entre cuadros consecutivos — la misma idea subyacente que usan las funciones de "virtual tripwire" de los VMS comerciales. Un evento solo se emite cuando **todas** estas condiciones se cumplen a la vez:

\[
\text{event eligible} = \text{confirmed track} \land \text{valid class} \land \text{zone or crossing condition} \land \text{not in cooldown}
\]

El término de cooldown importa tanto como la lógica de detección: sin él, una persona parada cerca del límite de una zona durante 10 segundos podría generar un "evento" nuevo en cada cuadro. En la práctica son útiles dos alcances de cooldown — por `(track_id, zone_id)` para que la misma persona no vuelva a disparar el evento mientras permanece ahí, y por `(camera_id, event_type)` para que una ráfaga de personas distintas no inunde el registro en una ventana corta.

**Corrección v11.2 — el alcance `(camera_id, event_type)` pasó a las notificaciones (§1.14).** Suprimir la entrada de una persona *distinta* borraría historial real. La máquina de estados conserva sólo el cooldown por `(track_id, zone_id)` y el temporizador de re-armado, emite un evento de entrada por cada track confirmado distinto, y el límite de ráfaga de 10 s lo aplica el notificador como throttling que registra cada notificación suprimida.

**Corrección (v2) — lee `is_confirmed`, no vuelvas a contar hits.** v1 volvió a especificar aquí `required_track_hits: 3`, duplicando `min_confirmed_hits` de `tracker.yaml` (§1.4). Elimina ese campo por completo. La verificación de confirmación de la máquina de estados se convierte en una lectura directa del booleano del tracker:

```python
event_eligible = (
    track.is_confirmed          # owned and computed solely by the tracker (§1.4)
    and track.mean_confidence >= config.minimum_mean_confidence
    and track.age_seconds >= config.minimum_track_age_seconds
    and zone_or_crossing_condition
    and not in_cooldown
)
```

**Corrección de reducción de alcance (v2) — para el MVP, entrega solo la detección de entrada.** Entregar simultáneamente la detección de entrada, cruce y permanencia significa ajustar tres tasas de falsos positivos independientes a la vez, sin forma de aislar cuál regla se está comportando mal. El MVP debe habilitar **solo** la regla de zona de entrada, reducir su tasa de falsos positivos a un nivel aceptable usando el banco de reproducción fuera de línea (§1.11), y solo entonces activar el cruce y la permanencia como una adición de fase dos, una vez que la entrada sea confiable.

**Corrección v3 — los polígonos de zona deben crearse antes de calibrar la puerta de movimiento.** La puerta de movimiento ahora deriva su máscara de ROI de `zone_polygons_plus_margin` (§1.2), lo cual convierte al polígono de zona en una *entrada* para la calibración de la puerta, en lugar de un artefacto de esta sección. Los polígonos de zona son configuración pura (una lista de puntos en el espacio de la imagen por cámara), no código, así que no hay razón para esperar: **créalos durante el proceso de etiquetado de verdad de referencia (hito 3)**, mientras de todas formas estás revisando cuadros de esa cámara y puedes ver exactamente dónde cae el límite de la entrada. Guárdalos en `configs/zones/<camera_id>.json` junto con la medición de presupuesto de píxeles de §0 para la misma cámara.

```yaml
# config/events.yaml (v3)
events:
  confirmation:
    # required_track_hits removed (v2) — read track.is_confirmed from the tracker instead
    minimum_track_age_seconds: 0.5
    minimum_mean_confidence: 0.35
  zone_rules:
    zone_definitions: configs/zones/     # v3: authored at milestone 3, consumed by §1.2 and §1.5
    require_confirmed_track: true
    entry_event_enabled: true            # MVP scope: ship this rule first
    crossing_event_enabled: false        # phase 2 — enable only after entry's FP rate is validated
    dwell_event_enabled: false           # phase 2 — enable only after entry's FP rate is validated
    dwell_seconds: 2.0
  cooldowns:
    same_track_same_zone_seconds: 30
    # same_camera_same_event_seconds moved to notify.yaml as a notification throttle (v11.2)
    rearm_after_track_lost_seconds: 5
```

**Para profundizar:**
- [Hikvision — Line Crossing Detection](https://enpinfo.hikvision.com/hkwsen/unzip/20230410194813_20373_doc/GUID-246BF07A-3F33-48FC-99D9-DE1AFC3E9144.html) — cómo un sistema comercial define y evalúa el cruce de línea virtual, útil como referencia de especificación
- [yas-sim/object-tracking-line-crossing-area-intrusion](https://github.com/yas-sim/object-tracking-line-crossing-area-intrusion) — una implementación de referencia abierta de lógica de zona/línea sobre objetos rastreados
- [Debounce design pattern (community.openhab.org)](https://community.openhab.org/t/design-pattern-debounce/101566) — el patrón general de software detrás de la lógica de cooldown/deduplicación

---

### 1.6 Capa de persistencia — `src/events/store.py`

**Propósito.** Dar a cada evento emitido un registro duradero y consultable — qué pasó, en qué cámara, a qué hora, vinculado a qué track — sin requerir un servidor de base de datos en una computadora de placa única.

**Cómo funciona.** SQLite es una base de datos relacional sin servidor, basada en archivos, integrada en la biblioteca estándar de Python a través del módulo `sqlite3` — sin proceso separado que ejecutar, sin puerto de red, y todo el historial de eventos vive en un solo archivo trivial de respaldar o copiar fuera de la Pi. Una tabla `events` (camera_id, track_id, event_type, zone_id, first_seen, last_seen, confidence, metadata JSON) se escribe una vez por cada evento emitido, inmediatamente después de que la máquina de estados confirma la elegibilidad — antes de que se ejecute cualquier paso de enriquecimiento, de modo que la API siempre tenga disponible el evento crudo aunque el enriquecimiento sea lento o falle. Habilita el modo WAL (write-ahead log) si el proceso de FastAPI y algún worker de enriquecimiento en segundo plano van a leer/escribir concurrentemente; esto permite que los lectores y un único escritor avancen sin bloquearse entre sí, lo cual importa una vez que la API está sirviendo consultas de `/events` mientras siguen escribiéndose nuevos eventos.

**Corrección v3 — la escritura no pertenece a la ruta crítica por cuadro.** El presupuesto de latencia de v2 asignaba 15 ms por cuadro a `sqlite_write`, pero la gran mayoría de los cuadros no emite ningún evento en absoluto, así que esa asignación es a la vez un desperdicio en el presupuesto y peligrosa en el ciclo: un checkpoint de WAL o un bloqueo de fsync detendría directamente el procesamiento de cuadros. Coloca los eventos emitidos en una cola en memoria consumida por un **hilo escritor dedicado**, y deja que la única obligación del ciclo de detección sea un enqueue O(1).

Dos propiedades que hay que preservar: la cola debe estar acotada con una política de desborde explícita (bloquear brevemente, luego registrar y descartar con un contador `events_dropped_total` — nunca crecer sin límite), y el escritor debe completar la fila **antes** de que se despache el enriquecimiento, lo cual ocurre de forma natural ya que el enriquecimiento consume del registro persistido.

**Columnas de marca de tiempo (v3).** Guarda ambos relojes por evento, por las razones dadas en §1.10: `captured_at_utc` (reloj de pared, la respuesta forense a "¿cuándo pasó esto?") y `captured_at_monotonic` (para aritmética de intervalos y para correlacionar contra `/metrics` sin que los saltos de NTP corrompan las matemáticas).

**Para profundizar:**
- [Python `sqlite3` module documentation](https://docs.python.org/3/library/sqlite3.html) — la interfaz de la biblioteca estándar usada directamente en este proyecto
- [SQLite WAL mode documentation](https://sqlite.org/wal.html) — por qué y cómo habilitar el write-ahead logging para acceso concurrente de lectura/escritura

---

### 1.7 Capa de API — `src/api/`

**Propósito.** Exponer el estado de la cámara y el historial de eventos a cualquier cosa fuera de la Pi — un dashboard, una app de teléfono, un comando curl durante la depuración — sobre HTTP simple, sin construir un protocolo personalizado.

**Cómo funciona.** FastAPI es un framework web de Python que genera validación de solicitudes y documentación interactiva de la API directamente a partir de firmas de función con anotaciones de tipo y modelos de Pydantic, y corre detrás del servidor ASGI Uvicorn. Una superficie mínima de MVP es pequeña a propósito: `GET /health` (vitalidad de cámara/pipeline), `GET /events` (historial reciente de eventos, filtrable por cámara/tiempo/zona), `GET /events/{id}` (detalle de un evento individual, incluyendo el texto de enriquecimiento una vez disponible), y `GET /metrics` (§1.10). Como FastAPI es nativamente async, los endpoints de eventos pueden consultar SQLite sin bloquear el procesamiento de cuadros que corre en un hilo/proceso separado — esta separación (ingestión/inferencia en un ciclo, servicio de API en otro) es lo que evita que un cliente HTTP lento detenga alguna vez el pipeline de detección.

**`/health` debe ser tri-estado, no booleano (v3).** Dada la política de fallo del NPU (§1.3) y la detección de cuadros obsoletos (§1.10), `healthy` / `degraded` / `down` transmite mucha más información operativa que arriba/abajo. "Ingiriendo y almacenando evidencia pero sin analizar" es un estado real e importante.

**Adición v4 — la API necesita un límite declarado de red/autenticación.** §1.9 es explícito en que Hailo-Ollama se enlaza únicamente a `127.0.0.1`; esta sección nunca hace la afirmación equivalente para la API que en realidad sirve el historial de eventos, el texto de enriquecimiento y los enlaces a clips de visitantes reales. Incluso para un despliegue en LAN privada, enlaza Uvicorn a una interfaz específica en lugar de `0.0.0.0`, y añade una verificación mínima de bearer-token en las rutas que no sean `/health` — un registro de eventos de cámara es exactamente el tipo de dato que no debería ser accesible por "cualquier otra cosa en la LAN" por defecto.

**Adición v10 — ciclo de vida del bearer-token y seguridad de transporte.** La adición v4 anterior dice que hay que añadir una verificación de bearer-token, pero no cómo generar, almacenar o rotar el token, ni si la conexión está cifrada. Genera el token una sola vez con `secrets.token_urlsafe(32)` y guárdalo fuera del repositorio — una variable de entorno o un archivo con permisos `0600`, nunca un valor de configuración confirmado (committed) — ya que este token controla el acceso al historial de eventos, al texto de enriquecimiento y a los enlaces a clips de visitantes reales. Trata la rotación como un procedimiento manual y documentado para el MVP (regenera y redistribuye el token si se sospecha de un compromiso) en lugar de construir rotación automática en esta etapa temprana. Como la API es HTTP simple por defecto, colócala detrás de un proxy inverso mínimo con terminación TLS (por ejemplo, Caddy con certificados automáticos, o nginx con un certificado autofirmado para uso solo en LAN) en lugar de enviar bearer tokens en texto plano, incluso en una red de confianza.

**Para profundizar:**
- [FastAPI official tutorial](https://fastapi.tiangolo.com/tutorial/) — la guía canónica de introducción, incluyendo la generación automática de documentación
- [FastAPI project homepage](https://fastapi.tiangolo.com/) — resumen del framework y justificación de diseño

---

### 1.8 Programador / Heartbeat — APScheduler

**Propósito.** Ejecutar trabajos periódicos, no impulsados por solicitudes — la inferencia de heartbeat en reposo de la puerta de movimiento, la limpieza de tracks obsoletos, los barridos de expiración de cooldown, o una verificación nocturna de integridad de SQLite — sin construir a mano hilos de temporizador.

**Cómo funciona.** APScheduler ejecuta trabajos en horarios de intervalo, tipo cron, o de una sola vez, dentro del mismo proceso de Python, respaldado por un almacén de trabajos configurable (en memoria es suficiente para el MVP). Es el lugar natural para implementar `maximum_idle_inference_interval_seconds` de la puerta de movimiento y cualquier lógica de "barrer y expirar" que la máquina de estados de eventos necesite con una cadencia fija, en lugar de ser disparada por un cuadro nuevo.

**Para profundizar:**
- [APScheduler documentation](https://apscheduler.readthedocs.io/) — guía de usuario que cubre triggers, almacenes de trabajos y ejecutores

---

### 1.9 Enriquecimiento local con GenAI — `src/enrichment/` (Hailo-Ollama / HailoRT VLM)

**Propósito.** Convertir un registro de evento desnudo ("persona entró a la zona doorstep a las 14:20, track 42") en un resumen breve y legible para humanos, sin enviar ningún dato de evento, cuadro o metadato a una API de nube externa — el enriquecimiento ocurre **después** de que el evento ya fue persistido y servido, de modo que nunca puede ralentizar o bloquear la ruta de detección en vivo.

**Cómo funciona.** El paquete Hailo Model Zoo GenAI incluye **Hailo-Ollama**, un servidor REST (superficie de API compatible con Ollama, implementada en C++ sobre HailoRT) que ejecuta LLMs locales pequeños directamente en la memoria dedicada del Hailo-10H, independiente del propio CPU/RAM de la Pi. Se ejecuta `curl http://127.0.0.1:8000/hailo/v1/list` para ver qué modelos están disponibles localmente, y luego se hace un POST con metadatos de evento estructurados como prompt, para recibir de vuelta una descripción breve en lenguaje natural. Para el MVP, limita esto a la síntesis **solo de texto** de metadatos estructurados (cámara, zona, tiempo de permanencia, id de track) — ese es un uso bien soportado y de bajo riesgo de Hailo-Ollama. El análisis consciente del cuadro o del recorte (mirar realmente la imagen) es una ruta de capacidad materialmente distinta — corre a través de una aplicación VLM de HailoRT en lugar del endpoint de texto estilo Ollama — y debe permanecer deshabilitado en `config/enrichment.yaml` hasta que se pruebe por separado, ya que las características de corrección y latencia difieren de la ruta de texto.

**Nota v3 — el modelo de enriquecimiento comparte el NPU con el detector.** Cargar un LLM de clase 1B en el Hailo-10H consume memoria del dispositivo que YOLOv8n también necesita. Mide la latencia del detector con y sin el modelo de enriquecimiento residente antes de asumir que el margen de 40 TOPS absorbe a ambos, y dale a la cola de enriquecimiento una política de descarte explícita (`queue_max_size` con desalojo de más antiguo primero) para que un LLM lento o atascado nunca pueda aplicar contrapresión (backpressure) a la ruta de detección.

```yaml
# config/enrichment.yaml
enrichment:
  enabled: true
  execution_mode: asynchronous
  queue_max_size: 100
  queue_overflow_policy: drop_oldest   # v3: enrichment must never backpressure detection
  llm:
    provider: hailo_ollama
    base_url: http://127.0.0.1:8000
    model: <validated-local-model>
    timeout_seconds: 20
    max_tokens: 160
  vlm:
    enabled: false
    provider: hailort_vlm
  privacy:
    external_network_calls: false
    store_prompts: false
    store_raw_frames_in_prompt_logs: false
```

**Para profundizar:**
- [Hailo Model Zoo GenAI on GitHub](https://github.com/hailo-ai/hailo_model_zoo_genai) — servidor Hailo-Ollama, endpoints de pull/list/chat de modelos, y prerrequisitos de hardware/OS
- [Raspberry Pi AI HAT+ 2 — Hailo-10H local LLM walkthrough](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-local-llms) — números de rendimiento reales (aproximadamente 30–50 tokens/seg para un modelo de clase 1B) y un ejemplo trabajado de visión+LLM en este hardware exacto
- [Hailo GenAI Model Explorer — VLM models](https://hailo.ai/products/hailo-software/model-explorer/generative-ai/type/vlm/) — el catálogo separado de modelos de visión-lenguaje, para cuando se retome el enriquecimiento consciente de imagen

---

### 1.10 Backpressure de ingesta, presupuesto de latencia y disciplina de marcas de tiempo — `src/ingest/rtsp_reader.py`, `src/observability/metrics.py`

**Propósito.** Todo lo anterior (§1.1–§1.5) asume que el pipeline mantiene el ritmo de la cámara en tiempo real. Esta sección hace explícita, medible y exigida esa suposición — en lugar de descubrir meses después que las alertas se han vuelto silenciosamente minutos tarde.

**Cómo funciona.** Ejecuta la captura de cuadros en un **hilo dedicado** que lee continuamente de `cv2.VideoCapture` hacia un "buzón" (mailbox) de una sola ranura — cada cuadro nuevo simplemente sobrescribe lo que había ahí, y el ciclo de procesamiento siempre lee el cuadro más reciente disponible en lugar de vaciar una cola de cuadros obsoletos.

**Corrección v3 — la implementación de referencia de v2 era insegura de tres formas.** El fragmento entregado en v2 eliminaba la lógica de reconexión que §1.1 exige explícitamente, hacía girar un núcleo de CPU al 100% cuando `read()` devolvía `False`, y no le daba al consumidor ninguna forma de distinguir una cámara congelada de una inactiva pero saludable. Como §1.10 es el código que alguien realmente va a copiar, tiene que ser correcto. Las tres correcciones a continuación:

```python
import cv2, threading, time
from datetime import datetime, timezone


class FreshFrameReader:
    """Single-slot mailbox: always yields the newest frame, never a queued stale one."""

    def __init__(self, uri, stale_after_s=2.0, max_backoff_s=30.0):
        self._uri = uri
        self._stale_after_s = stale_after_s
        self._max_backoff_s = max_backoff_s
        self._lock = threading.Lock()
        self._latest = None            # (frame, monotonic_ts, utc_ts)
        self._stop = threading.Event()
        self._reconnects = 0
        self._last_served_mono_ts = None   # v11: lets get_latest() report whether this is a new physical frame
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self._uri)
        # Honored by some FFmpeg builds, ignored by others — the mailbox is the real fix.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _read_loop(self):
        cap = self._open()
        backoff = 0.5
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok:
                backoff = 0.5                       # v3: reset backoff on success
                with self._lock:
                    # Both clocks, captured at read time (see "timestamp semantics" below).
                    self._latest = (frame,
                                    time.monotonic(),
                                    datetime.now(timezone.utc))
                continue

            # v3: reconnect with exponential backoff instead of busy-looping forever.
            cap.release()
            time.sleep(backoff)
            backoff = min(backoff * 2, self._max_backoff_s)
            cap = self._open()
            self._reconnects += 1

    def get_latest(self):
        """Returns (frame, monotonic_ts, utc_ts, is_stale, is_new) or None before the first frame.

        v11 fix: the mailbox has no queue depth, so a consumer running faster than the
        camera's frame rate previously had no way to tell it had just re-read the same
        physical frame it already processed. `is_new` is False whenever `mono_ts` is
        unchanged since the last call — i.e. no new frame has arrived since this consumer
        last looked.
        """
        with self._lock:
            if self._latest is None:
                return None
            frame, mono_ts, utc_ts = self._latest
        # v3: a frozen camera returns successfully forever unless staleness is checked.
        is_stale = (time.monotonic() - mono_ts) > self._stale_after_s
        is_new = mono_ts != self._last_served_mono_ts   # v11: new-frame signal
        self._last_served_mono_ts = mono_ts
        return frame, mono_ts, utc_ts, is_stale, is_new
```

Una lectura obsoleta es una **señal de salud, no un error**: el consumidor debe dejar de enviar cuadros obsoletos al NPU, marcar `/health` como degraded (§1.7), e incrementar `frames_stale_total`. Una cámara que está encendida pero congelada es una falla común y, de otro modo, invisible.

**Corrección v11 — un consumidor rápido no debe reprocesar un cuadro físico como si fueran varios.** Nada regula el ritmo de este ciclo a la tasa de cuadros de la cámara, y una vez que `has_active_tracks` es verdadero, la puerta de movimiento (§1.2) se omite en cada iteración (nota propia de §1.2). Una iteración con puerta activada sin NPU y, especialmente, una iteración completa de inferencia con NPU pueden terminar cada una fácilmente dentro de un período de cuadro de 100 ms, así que el consumidor puede llamar a `get_latest()` y ejecutar la actualización de MOG2/inferencia/tracker contra el *mismo* cuadro varias veces antes de que el hilo lector produzca uno nuevo. Verifica `is_new` antes de hacer cualquiera de ese trabajo: omite la actualización de MOG2, el envío al NPU, y la actualización del conteo de hits del tracker siempre que `is_new` sea `False` (el manejo de obsolescencia y reconexión sigue corriendo en cada iteración sin importar esto). En consecuencia, `min_confirmed_hits` (§1.4) debe contar un hit por cada `mono_ts` distinto, nunca por iteración del ciclo — de lo contrario, un artefacto de detector de un solo cuadro (un glitch de compresión, un insecto atraído por el IR, un destello de faro) puede satisfacer todas las coincidencias de `min_confirmed_hits` contra sí mismo en Δt≈0, reintroduciendo el mismo falso positivo de parpadeo de un solo cuadro que el diseño de confirmación por hits del tracker estaba destinado a eliminar (§1.4). Nota que este modo de falla no aparece en el banco de reproducción de §1.11, que alimenta cada cuadro grabado exactamente una vez — así que las tasas de falsos positivos medidas por el banco no reflejarán este riesgo; hay que razonarlo directamente, no ajustarlo hasta que desaparezca.

**Corrección v3 — el rendimiento (throughput) y la latencia son presupuestos distintos, y v2 los confundió.** v2 definía un único `latency_budget_ms` de 100 ms cuyos componentes sumaban exactamente 100. Dos problemas separados:

1. **Miden cosas distintas.** 100 ms a 10 FPS es el **período entre cuadros** — una restricción de *throughput*, que significa que el procesamiento de cada cuadro debe terminar antes de que llegue el siguiente o el pipeline se atrasa. La **latencia** de extremo a extremo (captura → fila de evento escrita) es una cantidad distinta que legítimamente puede exceder un período de cuadro cuando las etapas están en pipeline (pipelined). Rastrea ambas, con objetivos separados y alarmas separadas. Confundirlas significa que un pipeline que va perfectamente al ritmo puede parecer que está fallando su presupuesto, y viceversa.
2. **Un presupuesto que suma el 100% del tiempo disponible ya está sobregirado.** No hay margen para pausas de GC, el worker de enriquecimiento, el proceso de la API, la E/S del búfer de pre-grabación, o una segunda cámara más adelante. Apunta a aproximadamente **60–70% de utilización** del período de cuadro, así que dirige la suma por etapa a 60–70 ms, no a 100.

También se elimina del presupuesto por cuadro: `sqlite_write`, que se traslada al hilo escritor dedicado (§1.6) ya que casi ningún cuadro emite eventos.

Instrumenta la duración de cada etapa y exponla en un endpoint `/metrics` (el formato de texto de Prometheus funciona bien con FastAPI vía `prometheus-fastapi-instrumentator`). En la práctica, en este hardware los 40 TOPS del Hailo-10H manejan YOLOv8n con margen de sobra — los cuellos de botella más probables son la carga de decodificación H.264 en el CPU de la Pi y la sobrecarga de copia de memoria de HailoRT por llamada, así que construye tus dashboards en torno a esas dos etapas primero.

**Corrección v5 — replanteado como un elemento de medición del hito 1, no como un techo estricto.** La Raspberry Pi 5 eliminó el bloque de decodificación H.264 por hardware que tenía la Pi 4; su único decodificador de video por hardware es H.265/HEVC ([Raspberry Pi Forums — no H.264 hardware decode on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=364180), [decode architecture comparison across Pi generations](https://salivity.github.io/ffmpeg/article/hardware-accelerated-video-decoding-on-raspberry-pi-with-ffmpeg)). Eso es cierto, pero el planteamiento de v4 exageraba su consecuencia: los ingenieros de Raspberry Pi reportan que los cuatro núcleos Cortex-A76 de la Pi 5 decodifican H.264 por software lo suficientemente rápido para superar directamente al viejo bloque de hardware de la Pi 4, incluso en resoluciones que ese bloque no podía manejar en absoluto ([Raspberry Pi Forums — Pi 5 software decode vs. Pi 4 hardware decode](https://forums.raspberrypi.com/viewtopic.php?t=391283), [Raspberry Pi Forums — NEON-optimised software H.264 decode](https://forums.raspberrypi.com/viewtopic.php?t=357870)), y un substream de 640×360@10fps se sitúa en apenas aproximadamente 1/25 a 1/40 de la tasa de píxeles de las cargas de trabajo 1080p30/4K que describen esos reportes. Eso es motivo para medir, no para asumir en ninguna dirección: **añade una medición de costo de decodificación a los criterios de salida del hito 1** (§3) — cronometra las lecturas de `cv2.VideoCapture` contra el `.mp4` grabado en la resolución/tasa de cuadros del substream objetivo y registra la duración real de decodificación por cuadro y la utilización de núcleo, en lugar de conservar el marcador de posición `decode: 25` ms de abajo como un peor caso asumido. El costo aún escala linealmente con cada cámara añadida, que es lo que en realidad importa para la generalización multi-cámara mencionada al final de §3 — un costo que es insignificante con una cámara puede seguir siendo el primer presupuesto que se agote con cuatro, así que mídelo una vez aquí y vuelve a revisarlo a medida que se añaden cámaras, en lugar de volver a derivar la suposición más adelante. **Corrección v6 — el cambio de codec a H.265/HEVC en sí se posterga al paso de generalización multi-cámara, no se recomienda aquí.** El ahorro medido al decodificar H.265 a través del bloque de hardware de la Pi 5 vía `-hwaccel drm` es modesto con una cámara (13% → 9% de un núcleo — aproximadamente 1% del CPU total) y el cambio también afecta la ruta de remux de PyAV de §1.12, así que no vale la pena asumirlo durante el hito 2 de una sola cámara; ver el párrafo de generalización multi-cámara al final de §3 para la recomendación y el detalle de `-hwaccel drm`, donde cuatro streams de cámara hacen que el mismo ahorro por núcleo valga la complejidad añadida.

**Corrección (v2) — semántica de marca de tiempo.** Todo registro de evento debe llevar la **marca de tiempo de captura** del cuadro (fijada en el instante en que el hilo de ingestión lee el cuadro) como su tiempo autoritativo, nunca la marca de tiempo de cualquier etapa posterior que llegue a procesarlo. Bajo carga, el tiempo de procesamiento puede legítimamente atrasarse respecto al tiempo de captura por cientos de milisegundos o más; si los eventos se marcan con el tiempo de procesamiento, la secuencia registrada de "qué pasó cuándo" se vuelve incorrecta justo cuando más la necesitas.

**Corrección v3 — `time.monotonic()` no puede ser la marca de tiempo del evento.** El lector de v2 marcaba los cuadros únicamente con `time.monotonic()`. El tiempo monotónico es un contador sin ancla desde un origen arbitrario: es exactamente correcto para calcular \(\Delta t\) (inmune a los saltos de NTP, lo cual importa muchísimo ahora que tanto el filtro de Kalman como la tasa de aprendizaje de MOG2 dependen de \(\Delta t\)), pero **no es una fecha** y no puede responder "¿cuándo pasó esto?". Captura **ambos** relojes al momento de la lectura, como hace el código anterior, y persiste ambos (§1.6).

**Nota de hardware relacionada:** la Raspberry Pi 5 tiene un RTC, pero requiere una **batería de botón (coin-cell)** para conservar la hora tras una pérdida de energía. Sin ella, cada evento registrado entre el arranque en frío y la sincronización de NTP lleva una marca de tiempo de reloj de pared incorrecta. Para un sistema cuyo valor entero es forense, instala la batería.

```yaml
# config/observability.yaml (v3)
observability:
  metrics_endpoint: /metrics

  # Throughput: per-frame processing must fit inside the frame period, with headroom.
  frame_period_ms: 100              # 10 FPS substream
  throughput_target_utilization: 0.65   # v3: aim the per-stage sum at ~65 ms, not 100 ms
  stage_budget_ms:
    decode: 25                     # v5: provisional — replace with milestone-1's measured value (§1.10)
    motion_gate: 10
    npu_inference: 25
    tracker_update: 5
    # sqlite_write removed (v3) — moved to the dedicated writer thread (§1.6)

  # Latency: a separate end-to-end measurement with its own alarm.
  end_to_end_latency_ms:
    target_p50: 150
    alarm_p95: 400

  staleness:
    frame_stale_after_seconds: 2.0
  counters:
    - frames_captured_total
    - frames_dropped_total
    - frames_stale_total
    - rtsp_reconnects_total
    - npu_submissions_total
    - npu_failures_total
    - events_emitted_total
    - events_dropped_total
  alert_on_budget_overrun: true
  overrun_log_level: warning
```

**Para profundizar:**
- [PyImageSearch — Increasing webcam FPS with a threaded video stream](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/) — el patrón de captura con hilos en el que se basa este diseño
- [Stack Overflow — OpenCV VideoCapture lag due to the capture buffer](https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer) — por qué el uso ingenuo de `VideoCapture` acumula latencia silenciosamente
- [prometheus-fastapi-instrumentator (GitHub)](https://github.com/trallnag/prometheus-fastapi-instrumentator) — métricas de Prometheus listas para usar en una app de FastAPI, usadas para el endpoint `/metrics`
- [Raspberry Pi 5 RTC documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) — el conector de batería de botón y el comportamiento del RTC tras una pérdida de energía
- [Raspberry Pi Forums — Pi 5 has no hardware H.264 decoder](https://forums.raspberrypi.com/viewtopic.php?t=364180) — confirma la brecha de codec/decodificación que motiva la recomendación de substream H.265 (v4)
- [Hardware-accelerated video decoding on Raspberry Pi with FFmpeg](https://salivity.github.io/ffmpeg/article/hardware-accelerated-video-decoding-on-raspberry-pi-with-ffmpeg) — comparación de capacidad de decodificación entre generaciones Pi 4/5 (v4)
- [Raspberry Pi Forums — Pi 5 software decode outperforms Pi 4 hardware decode](https://forums.raspberrypi.com/viewtopic.php?t=391283) — evidencia de que el costo de decodificación por software no es automáticamente un techo estricto (v5)
- [Raspberry Pi Forums — NEON-optimised software H.264 decode on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=357870) — el mismo punto, con detalle de implementación de por qué es rápido (v5)
- [Frigate GitHub discussion — Pi 5 `hwaccel drm` vs. `v4l2m2m`](https://github.com/blakeblackshear/frigate/discussions/18431) — la bandera correcta de decodificación para la Pi 5, con deltas de CPU medidos en el mundo real (v5)

---

### 1.11 Banco de reproducción y evaluación fuera de línea — `src/eval/replay_runner.py`

**Propósito.** Cada umbral de esta guía — `minimum_motion_ratio`, `confidence_threshold`, `min_iou_match`, `min_confirmed_hits`, `minimum_mean_confidence`, `kalman_sigma_accel_sq` — se ha descrito hasta ahora como "calibrado por escena de cámara", sin especificar calibrado *contra qué*. Sin verdad de referencia, ajustar es solo adivinar con pasos adicionales. Esta es la brecha individual más grande que identificó la revisión v2.

**Cómo funciona.** Graba en disco el substream crudo (e, idealmente, el mainstream) de video para cada cámara durante un periodo representativo — incluyendo casos límite como transiciones de iluminación al atardecer/amanecer, árboles agitados por el viento y visitas de vehículos de reparto. Etiqueta a mano un CSV de eventos de verdad de referencia (`camera_id, event_type, start_ts, end_ts, condition, notes`) contra ese material grabado. Luego construye un **banco de reproducción** que alimenta el video grabado a través del *mismo* código de la canalización (pipeline) de producción (puerta de movimiento → detector → tracker → máquina de estados) usando un **reloj falso** impulsado por las marcas de tiempo de los fotogramas grabados en lugar de `time.monotonic()`, de modo que el filtro de Kalman parametrizado en el tiempo (§1.4), la tasa de aprendizaje de MOG2 (§1.2) y la lógica de enfriamiento (§1.5) se comporten de forma idéntica a una ejecución en vivo.

```python
class FakeClock:
    def __init__(self, start_ts):
        self._t = start_ts
    def now(self):
        return self._t
    def advance_to(self, frame_ts):
        self._t = frame_ts   # driven by recorded frame timestamps, not wall-clock time
```

**Adición v3 — define la regla de coincidencia, o precisión y recall no son calculables.** v2 especificó puertas de CI (`fail_ci_below_precision: 0.85`) sin definir nunca qué cuenta como *coincidencia* entre un evento emitido y un evento de verdad de referencia etiquetado. Esa definición es el parámetro individual más determinante del banco de pruebas, y cada elección razonable produce un número materialmente distinto:

- **Ventana temporal.** ¿Un evento emitido coincide con una etiqueta si cae dentro de `[start_ts - δ, end_ts + δ]`? ¿Cuál es δ? Una δ generosa infla el recall; una estricta castiga a la canalización por un retraso de confirmación legítimo de medio segundo.
- **Cardinalidad.** Si el sistema dispara tres veces durante una sola visita real, ¿es eso un verdadero positivo, o un TP más dos falsos positivos? (Recomendado: **coincidencia voraz uno a uno por marca de tiempo más cercana** — cada etiqueta puede ser reclamada por, como máximo, un evento emitido, y toda emisión no reclamada es un falso positivo. Esta es la elección que realmente penaliza los errores de alertas duplicadas, que es el comportamiento que te interesa.)
- **Las etiquetas sin coincidencia** son falsos negativos por definición; asegúrate de que el banco de pruebas los reporte individualmente con marcas de tiempo, no solo como un conteo, para que puedas ir a revisar el material de lo que se perdió.

Escribe esto en `eval.yaml` como parámetros explícitos y versionados. Un número de precisión calculado bajo una regla de coincidencia no documentada no es reproducible y no pertenece a un reporte de portafolio.

**Adición v3 — reserva una partición de validación, y sé honesto sobre el tamaño de la muestra.** Realistamente etiquetarás a mano entre unas pocas decenas y un par de cientos de eventos. Ajustar cinco o más umbrales contra ese conjunto es una búsqueda de alta dimensionalidad sobre una muestra pequeña, y *sí* va a sobreajustar: los umbrales a los que llegues se verán excelentes en el material etiquetado y generalizarán peor de lo que sugieren las métricas.

La solución es la estándar y aquí no cuesta nada: **divide el material grabado por día.** Ajusta en los días de desarrollo y reporta los números finales en un día reservado que nunca se usó para el ajuste. Solo el número de la partición reservada va en el reporte o en el README.

Sobre las puertas de CI en específico: con 23 eventos etiquetados, una precisión observada de 0.87 conlleva un intervalo de confianza del 95% de aproximadamente ±0.14. Una puerta rígida en 0.85 contra esa muestra está midiendo ruido, no calidad. Registra `n_labeled_events` en la salida del banco de pruebas, reporta el intervalo junto con la estimación puntual, y trata las puertas como orientativas hasta que el conjunto de etiquetas sea suficientemente grande para respaldarlas.

**Adición v10 — una semana calendario de etiquetado del hito 3 no alcanzará por sí sola el umbral de puerta rígida.** Es poco probable que `minimum_labeled_events_for_hard_gate: 100` (abajo) se cumpla a partir de una sola semana de material a las tasas típicas de eventos de una cámara de puerta, lo que significa que las puertas de CI podrían quedarse en su estado orientativo indefinidamente en lugar de que el paso del hito 3 sea un paso único hacia ellas. Mantén el etiquetado como una tarea continua y progresiva más allá del hito 3 — sigue añadiendo eventos etiquetados a partir de la grabación 24/7 ya en marcha (§3, hito 2) a un ritmo regular — en lugar de tratar el único paso del hito 3 como el conjunto permanente de verdad de referencia del banco de pruebas.

**Corrección v4 — ese ±0.14 es un intervalo de Wald, y Wald es la herramienta equivocada a este tamaño de muestra.** El intervalo anterior usa la fórmula clásica de aproximación normal (Wald), que está documentada como subcubriente — e incluso puede producir límites fuera de [0, 1] — precisamente en el régimen de \(n\) pequeña y cercano al límite en el que este banco de pruebas operará durante meses ([análisis comparativo de la cobertura de Wald/Wilson/Jeffreys](https://arxiv.org/html/2508.10223v1), [Brown, Cai & DasGupta (2001) — Interval Estimation for a Binomial Proportion](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full)). Usa en su lugar un **intervalo de puntuación de Wilson** — un cambio de cinco líneas (`statsmodels.stats.proportion.proportion_confint(count, nobs, method="wilson")`), y honesto justo cuando el conjunto de etiquetas es más pequeño, que es cuando este proyecto más necesita que se confíe en sus números:

\[
\tilde p = \frac{\hat p + \dfrac{z^2}{2n} \pm z\sqrt{\dfrac{\hat p(1-\hat p)}{n} + \dfrac{z^2}{4n^2}}}{1 + \dfrac{z^2}{n}}
\]

**Adición v4 — el banco de pruebas también debería reportar trazas por etapa, no solo la lista final de eventos.** Tal como está especificado, una coincidencia fallida te dice *que* el sistema se perdió o duplicó una alerta, pero no *en dónde* de la canalización está la falla — puede que el detector nunca haya delimitado (boxed) a la persona, que el tracker haya delimitado pero nunca confirmado (o confirmado y luego perdido la identidad), o que la máquina de estados haya confirmado pero la geometría de zona o el enfriamiento se hayan comido el evento. Haz que `replay_runner.py` emita una traza intermedia por ventana etiquetada — fotogramas con una detección cruda, fotogramas con un track confirmado, y el ciclo de vida completo del track — junto con los números finales de precisión/recall. Sin esto, encontrar la causa raíz de un evento perdido todavía requiere reinstrumentar la canalización a mano cada vez; con esto, el banco de pruebas responde directamente "qué etapa falló".

**Adición v6 — registra qué configuración produjo cada número reportado.** A estas alturas once archivos YAML distintos alimentan la canalización, y nada actualmente vincula un número de precisión/recall reportado con la configuración exacta que lo produjo — volver a ejecutar el banco de pruebas después de cualquier cambio de ajuste invalida silenciosamente la procedencia del número anterior. Haz que `replay_runner.py` calcule un hash (por ejemplo, SHA-256 del contenido concatenado y serializado de forma canónica de cada archivo de configuración que carga) y lo emita junto con `n_labeled_events` en su salida, y agrega un campo `config_hash` correspondiente a `eval.yaml` para que el banco de pruebas lo llene en tiempo de ejecución. Esto convierte "la precisión fue 0.87" en "la precisión fue 0.87 bajo la configuración `a3f9e1...`", que es lo que hace que una regresión posterior o los números de un reporte de portafolio sean realmente reproducibles.

**Adición v7 — la división `day` / `night_ir` se especificó en cinco lugares (la nota de revisión v4 de esta sección, la corrección v4 de §1.3, la adición v5 de §1.2, los criterios de salida del hito 3, y la lista de verificación de preparación) pero nunca se implementó realmente en `eval.yaml`.** La lista `metrics:` de abajo era enteramente a nivel de evento, así que nada calculaba el recall del detector por fotograma que §1.3 necesita para distinguir una brecha de dominio genuina de un `confidence_threshold` mal ajustado — un número a nivel de evento mezcla el desempeño del detector con el comportamiento del tracker y la máquina de estados por encima de él. Se añadió un campo `conditions: [day, night_ir]` y un campo `condition_source` al bloque `split:` para que cada métrica se reporte por condición en lugar de agrupada entre ambas; se añadió una lista separada `detector_metrics: [per_frame_recall, per_frame_precision]` junto a `metrics:`, ya que estas se calculan directamente contra las detecciones crudas por fotograma en lugar de eventos emparejados; y se añadió una entrada `negative_test_cases:` para el caso de insectos con IR de §1.2, que espera cero eventos emitidos a pesar de una actividad sostenida en la etapa del detector. El esquema del CSV de verdad de referencia de arriba ahora incluye una columna `condition`, ya que de otro modo no había dónde registrar a cuál de `day` / `night_ir` pertenece un evento etiquetado — sin ella, el banco de pruebas no tiene forma de calcular nada por condición sin importar lo que pida `eval.yaml`.

```yaml
# config/eval.yaml (v8)
eval:
  recordings_dir: data/recordings/
  ground_truth_csv: data/ground_truth_events.csv

  # v3: the matching rule — without this, precision/recall are undefined
  matching:
    strategy: one_to_one_greedy_nearest
    temporal_window_seconds: 5.0      # emitted event must fall within [start - w, end + w]
    unmatched_emission: false_positive
    unmatched_label: false_negative
    report_unmatched_individually: true

  # v3: honest evaluation protocol
  split:
    tuning_days: [<tuning-day-1>, <tuning-day-2>, <tuning-day-3>]   # v8: placeholders, not literal dates — these must fall inside the week-2 (§3 milestone 2) recording window this harness actually labels from, and shift with the schedule rather than staying fixed
    heldout_days: [<heldout-day-1>]   # v8: same as above; never used for tuning, the only number that gets reported
    conditions: [day, night_ir]       # v7: report every metric per condition, not pooled
    condition_source: ground_truth_csv.condition   # v7: read from the CSV's condition column
    reject_unresolved_placeholders: true   # v9: replay_runner.py fails fast if tuning_days/heldout_days still match <...>
  report_sample_size: true
  report_confidence_intervals: true
  confidence_interval_method: wilson       # v4: was an implicit Wald/normal approximation
  emit_per_stage_traces: true              # v4: detection / confirmed-track / event-level, not just final matches
  config_hash: null                        # v6: populated at run time — SHA-256 over every loaded config file, reported alongside n_labeled_events

  metrics: [precision, recall, f1, mean_latency_to_event_ms]
  detector_metrics: [per_frame_recall, per_frame_precision]   # v7: per-frame, detector-stage only — isolates a domain gap from a threshold problem
  fail_ci_below_precision: 0.85       # advisory until n_labeled_events is large enough to support it
  fail_ci_below_recall: 0.80
  minimum_labeled_events_for_hard_gate: 100

  negative_test_cases:                # v7: cases that must emit zero events regardless of detector-stage activity
    - name: night_ir_insect_activity   # §1.2 v5 addition — IR-attracted insects at close range to the lens
      condition: night_ir
      expect: no_events
```

**Corrección v8 — las fechas de `split:` anteriores nombraban antes días calendario que aún no habían ocurrido.** `tuning_days` era del `2026-09-28` al `2026-09-30` y `heldout_days` era `2026-10-01`, pero los criterios de salida del hito 3 (§3) etiquetan el CSV de verdad de referencia a partir del **material de la semana 2** (la ventana de grabación del 21 al 27 de septiembre) — así que las fechas literales de la división caían enteramente dentro de la propia semana del hito 3, después de que comienza el etiquetado, no dentro del material del que realmente parte el etiquetado. Se reemplazaron con marcadores de posición (`<tuning-day-1>`, etc.) en lugar de un conjunto distinto de fechas fijas, ya que los valores reales dependen de dónde caiga el calendario cuando se ejecute el hito 3; ambos campos son ilustrativos y están pensados para moverse con el calendario, no para copiarse tal cual.

**Adición v9 — falla de forma temprana si los marcadores de posición de arriba nunca se llenan.** Antes nada impedía que `replay_runner.py` se ejecutara contra los marcadores de posición literales de estilo `<tuning-day-1>`, lo que produciría silenciosamente una ejecución sin sentido o con fallo en lugar de un error claro. Se añade `reject_unresolved_placeholders: true` arriba, y una verificación de arranque correspondiente en `replay_runner.py` — consistente con la filosofía de fallo temprano ya aplicada en §1.3 y §1.13 — que se niega a iniciar una ejecución del banco de pruebas mientras cualquier entrada de `split.tuning_days`/`heldout_days` todavía coincida con el patrón de marcador de posición `<...>`.

**Lecturas adicionales:**
- [Evaluating object detection models: methods and metrics (GeeksforGeeks)](https://www.geeksforgeeks.org/computer-vision/evaluating-object-detection-models-methods-and-metrics/) — definiciones de precisión/recall/F1 aplicadas a canalizaciones de detección
- [Object detection metrics explained (Label Your Data)](https://labelyourdata.com/articles/object-detection-metrics) — un recorrido práctico de las mismas métricas con ejemplos resueltos
- [Comparative analysis of Wald, Wilson, and other proportion CIs](https://arxiv.org/html/2508.10223v1) — comportamiento de cobertura con n pequeña y proporciones cercanas al límite (v4)
- [Brown, Cai & DasGupta (2001) — Interval Estimation for a Binomial Proportion](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) — el artículo canónico que establece la falla de cobertura de Wald con n pequeña/cerca del límite y recomienda Wilson (Statistical Science 16(2):101–133) (v5)

---

### 1.12 Búfer de evidencia de pre-grabación — `src/ingest/ring_buffer.py`

**Propósito.** Debido a que un track solo se vuelve `is_confirmed` después de `min_confirmed_hits` coincidencias y `minimum_track_age_seconds` (§1.4–§1.5), el evento en sí mismo siempre se dispara **después** de que el comportamiento interesante ya comenzó — por definición, la confirmación requiere haberlo observado ocurrir ya durante un momento. Sin un búfer, el clip guardado de un evento comienza a medio camino de la acción en lugar de mostrar el acercamiento.

**Cómo funciona.** Mantén un **ring buffer** continuo y de duración fija del mainstream (la transmisión de resolución completa, no el substream usado para la puerta de movimiento/detección), almacenando **paquetes codificados** en lugar de fotogramas decodificados para mantener bajo el consumo de CPU y memoria. Al disparar un evento, vacía una ventana que abarca una pre-grabación configurable (por ejemplo, 5–10 segundos antes de la primera detección del track) más una post-grabación (por ejemplo, 5–10 segundos después de que se dispara el evento o se pierde el track) en un clip guardado vinculado al registro del evento.

**Corrección v3 — `storage_mode: encoded_packets` no es alcanzable con OpenCV.** v2 especificó correctamente el almacenamiento en búfer de paquetes codificados, pero `cv2.VideoCapture` **decodifica**; no expone ningún acceso a los paquetes H.264 subyacentes. Nada en la lista de dependencias de v2 puede hacer lo que ese campo de configuración describe. Dos opciones viables:

- **PyAV** (bindings de Python para libav) — te permite desmultiplexar (demux) la transmisión RTSP y mantener `AVPacket`s crudos en un `collections.deque`, y luego remultiplexarlos (remux) en un MP4 al disparar, sin recodificar. Esta es la respuesta correcta y debería añadirse al stack.
- Un **subproceso de ffmpeg** que escribe segmentos rotativos (`-f segment -segment_time 2`) a disco, conservando los últimos N segmentos y concatenándolos al disparar. Más simple de implementar y depurar, a costa de escrituras continuas en disco y una granularidad de corte más gruesa.

**Corrección v3 — los clips solo se pueden cortar en keyframes.** Una transmisión H.264 solo es decodificable a partir de un IDR (keyframe) en adelante; comenzar un clip a mitad de un GOP produce basura hasta el siguiente. Dos consecuencias:

1. `buffer_seconds` debe exceder cómodamente el intervalo de keyframe de la cámara, o el búfer puede no contener ningún punto de corte en absoluto.
2. El **intervalo de I-frame / GOP de la cámara debe reducirse a aproximadamente 1 segundo** en su interfaz web (§1.1). Con un GOP predeterminado de 4 segundos, una "pre-grabación de 6 segundos" en realidad es "de 4 a 8 segundos, con una cabecera no decodificable". Este es un ajuste del lado de la cámara sin representación en el repositorio, que es exactamente por qué se descubre tarde.

La lógica de vaciado debe, por lo tanto, buscar hacia atrás desde el punto de pre-grabación solicitado hasta el **keyframe precedente más cercano**, y registrar la duración de pre-grabación realmente lograda en el registro del evento en lugar de asumir el valor configurado.

**Nota v3 — el búfer de pre-grabación es una segunda conexión RTSP continua.** Se ejecuta 24/7 contra el mainstream, independientemente del substream usado para la inferencia. Presupuesta su CPU (solo desmultiplexado, sin decodificación, así que es modesto) y su ancho de banda, y dale su propia lógica de reconexión y contadores de `/metrics`. También debería ser lo *último* en apagarse ante una falla: grabar evidencia sin análisis (§1.3) es un estado degradado útil.

```yaml
# config/pre_roll.yaml (v3)
pre_roll:
  enabled: true
  source_stream: mainstream
  backend: pyav                    # v3: OpenCV cannot expose encoded packets — PyAV or ffmpeg segments
  storage_mode: encoded_packets    # remux on trigger, no re-encode
  buffer_seconds: 15               # must exceed the camera GOP interval by a wide margin
  pre_roll_seconds: 6
  post_roll_seconds: 8
  cut_on_keyframe: true            # v3: search back to the nearest preceding IDR
  record_achieved_pre_roll: true   # v3: log what you actually got, not what you asked for
  expected_camera_gop_seconds: 1.0 # v3: set camera-side (§1.1); assert at startup if detectable
  clip_output_dir: /mnt/ssd/clips/
```

**Lecturas adicionales:**
- [PyAV documentation](https://pyav.org/docs/stable/) — desmultiplexado a paquetes crudos y remultiplexado sin recodificar, el mecanismo del que depende esta sección
- [picamera circular streams (deepwiki)](https://deepwiki.com/waveform80/picamera/4.2-circular-streams) — una implementación de referencia nativa de Raspberry Pi de exactamente este patrón circular de pre-grabación
- [VisioForge — Pre-event recording guide](https://www.visioforge.com/help/docs/dotnet/mediablocks/Guides/pre-event-recording/) — consideraciones de diseño generales para el almacenamiento en búfer de pre/post-grabación alrededor de un disparador
- [Battleroid/seccam (GitHub)](https://github.com/Battleroid/seccam) — una referencia abierta de cámara de seguridad que implementa la captura de clips de pre-grabación disparada por movimiento

---

### 1.13 Endurecimiento operativo y licenciamiento

**Propósito.** Una prueba de resistencia (soak test) de 24 horas (§3) solo es significativa si los modos de falla que tardan horas o días en manifestarse — agotamiento de disco, crecimiento excesivo de logs, desgaste del almacenamiento flash — se definen y se manejan de antemano, y si la posición legal del proyecto se resuelve antes de compartir públicamente cualquier código.

**Cómo funciona.**

- **Comportamiento con disco lleno.** Decide explícitamente qué ocurre cuando el volumen de clips/base de datos se llena: el comportamiento recomendado para el MVP es dejar de aceptar nuevos clips de pre-grabación (desalojo del más antiguo primero) mientras se siguen escribiendo las *filas* de eventos (pequeñas, económicas) para que el historial de alertas en sí nunca se rompa incluso si el almacenamiento de video se agota.
- **Rotación de logs.** Usa el `RotatingFileHandler` (o `TimedRotatingFileHandler`) integrado de Python para que los logs de depuración/puerta de movimiento/inferencia tengan un tope de tamaño total fijo en lugar de crecer sin límite durante una prueba de resistencia de varios días.
- **Desgaste de escritura de la tarjeta SD.** Las escrituras continuas de SQLite WAL (§1.6) más las escrituras frecuentes de logs son exactamente el patrón de acceso que acorta la vida de una tarjeta microSD. Antes de la prueba de resistencia, mueve el archivo de base de datos SQLite **y** el directorio de clips a un **SSD conectado por USB** en lugar de la microSD de arranque — los SSD tienen una resistencia de escritura mucho mayor y su modo de falla (degradación lenta, reportable por SMART) es mucho más seguro que una tarjeta SD que se corrompe silenciosamente. Con el búfer de pre-grabación (§1.12) escribiendo ahora clips continuamente al disparar, esto ya no es opcional.
- **Batería de RTC (v3).** Instala la pila de moneda (§1.10). Las marcas de tiempo de los eventos de un Pi que arrancó sin una son incorrectas hasta la sincronización NTP.
- **Salud del NPU (v3).** La política de fallas en §1.3 necesita una verificación de arranque correspondiente en la unidad de servicio: verifica que `hailortcli fw-control identify` tenga éxito antes de que la aplicación inicie, para que un desajuste de controlador/firmware después de un `apt upgrade` se manifieste como una falla de arranque limpia en lugar de una excepción en tiempo de ejecución a las 3 a.m.
- **Licenciamiento.** Como se señaló en §1.3, Ultralytics YOLOv8 se distribuye bajo AGPL-3.0. No es un impedimento para un MVP privado, pero resuélvelo *antes* de decidir hacer de código abierto el repositorio como pieza de portafolio — ya sea manteniendo el repositorio privado, aislando el código de entrenamiento/exportación derivado de Ultralytics de cualquier código de aplicación publicado, o presupuestando una licencia Ultralytics Enterprise si la redistribución amplia es una meta.
- **Supervisión de procesos (v4).** Todo lo anterior cubre la falla al *iniciar*; nada en v1–v3 cubre un cierre inesperado (crash), un bloqueo (hang) o un OOM una vez que la canalización ya está en ejecución — un segfault en los bindings nativos de `cv2`/HailoRT, el OOM-killer del kernel una vez que el LLM de enriquecimiento residente (§1.9) compite con el resto del proceso por RAM, o un hilo en segundo plano que se traba sin matar el proceso principal. Ejecuta la aplicación bajo una unidad de `systemd` con `Restart=on-failure`, un retroceso (backoff) `RestartSec`, y `StartLimitIntervalSec`/`StartLimitBurst` para que una falla persistente reinicie de forma limpia en lugar de entrar en un ciclo de fallas eterno ([patrones de reinicio/confiabilidad de systemd](https://forums.raspberrypi.com/viewtopic.php?t=376126), [guía práctica de confiabilidad de servicios en Raspberry Pi](https://www.dzombak.com/blog/2023/12/keep-your-software-up-and-running-on-the-raspberry-pi/)). Añade `WatchdogSec=` con un latido (heartbeat) periódico mediante `sd_notify` desde el bucle principal para que systemd también pueda detectar un proceso **bloqueado**, no solo uno que se cerró de forma inesperada — esta es la mejora de confiabilidad más económica disponible y sirve directamente al propio objetivo de la prueba de resistencia.
- **Corrección v11 — el latido debe verificar los hilos trabajadores (worker threads), no solo su propio pulso.** Tal como se especificó arriba, el latido solo demuestra que el hilo que llama a `sd_notify` sigue en su bucle; no dice nada sobre el hilo dedicado de escritura de SQLite (§1.6) ni sobre el hilo consumidor de notificaciones (§1.14). Si cualquiera de los dos muere por una excepción no capturada (un error de E/S de disco, una fila malformada, una excepción del cliente HTTP), el bucle principal sigue enviando `WATCHDOG=1`, el interruptor de hombre muerto (dead-man's switch) (§1.14) sigue recibiendo pings, y `/health` permanece saludable — porque ninguna de esas tres señales inspecciona nunca el estado de los hilos trabajadores, solo la actividad del proceso y el estado del NPU/desactualización (staleness) (§1.7). Los eventos entonces se acumulan en cola, alcanzan `queue_max_size` (§1.14), y se descartan silenciosamente vía `events_dropped_total`, un contador sobre el que nada genera alertas. Condiciona el latido a que los tres hilos estén vivos, de modo que un trabajador muerto detenga el latido, dispare el watchdog, y fuerce exactamente la ruta de recuperación de fallas + alerta fuera de proceso ya construida arriba:

  ```python
  # v11: heartbeat only fires if every required worker thread is still alive
  if all(t.is_alive() for t in (reader_thread, writer_thread, notifier_thread)):
      sd_notify("WATCHDOG=1")
  # else: skip the ping — WatchdogSec times out, systemd restarts the unit,
  # and (once EnvironmentFile above is fixed) OnFailure= fires the webhook.
  ```
- **Notificación de fallas fuera de proceso (v11).** `Restart=`/`WatchdogSec=` arriba *se recuperan* de un cierre inesperado o bloqueo; nada lo *reporta*, porque el único consumidor de notificaciones (§1.14, v9) vive dentro del proceso que acaba de morir — y una vez que se agota `StartLimitBurst`, la unidad se queda en estado `failed` indefinidamente sin que se le avise a nadie. Añade `OnFailure=camera-alert@%n.service` a la unidad de la canalización: systemd activa esa unidad de un solo disparo (oneshot) cada vez que la canalización entra en estado `failed` (un cierre inesperado, un bloqueo detectado por el watchdog, y el agotamiento final del límite de reinicios todos pasan por `failed` bajo el `RestartMode=normal` predeterminado), y la unidad oneshot entrega un webhook desde fuera del proceso fallido. Esto cubre "el proceso no se está ejecutando"; **no** cubre pérdida de energía, pérdida de red, o un kernel bloqueado, ya que systemd mismo desaparece en esos casos — ver el interruptor de hombre muerto en §1.14, que cubre exactamente ese conjunto disjunto.

  ```ini
  # /etc/systemd/system/camera-pipeline.service — [Unit] section addition (v11)
  [Unit]
  OnFailure=camera-alert@%n.service

  # /etc/systemd/system/camera-alert@.service (v11) — runs outside the failed process
  [Unit]
  Description=Out-of-process failure alert for %i

  [Service]
  Type=oneshot
  # NOTIFY_WEBHOOK_URL — same webhook as config/notify.yaml, never committed.
  # v11 fix: systemd does NOT support trailing/inline comments on a unit directive —
  # anything after `EnvironmentFile=<path>` on the same line becomes part of the path
  # value, so the line below must carry no comment of its own.
  EnvironmentFile=/etc/camera/notify.env
  ExecStart=/usr/bin/curl -fsS -d "%i entered failed state on %H" "${NOTIFY_WEBHOOK_URL}"
  ```

- **Presupuesto de memoria de todo el sistema (v4).** §1.9 ya señala la contención de memoria del *dispositivo* NPU entre el detector y el LLM de enriquecimiento residente; la misma disciplina aplica a la RAM del host. El conjunto concurrente en un solo Pi 5 es: el hilo del buzón (mailbox) RTSP, el hilo de desmultiplexado de pre-grabación de PyAV, el runtime de HailoRT, Uvicorn/FastAPI, APScheduler, SQLite (WAL), y el registro (logging) — sin ningún límite de RAM definido en ningún lugar. Haz un presupuesto aproximado durante el pico de exploración fuera de línea del hito 1 (con el modelo de enriquecimiento residente, ya que ese es el peor caso) y expón la memoria del host en `/metrics` junto con los contadores existentes de NPU/dispositivo, para que un crecimiento lento de memoria durante la prueba de resistencia sea visible antes de convertirse en un OOM kill.
- **Retención vs. capacidad (v4).** `disk_full_policy` arriba es una política de *capacidad* — solo se activa bajo presión de almacenamiento. No hay una purga independiente por edad máxima, ni una postura declarada sobre grabar incidentalmente a personas que no son del hogar (repartidores, carteros, vecinos que pasan por el límite de la zona). Incluso para un MVP residencial privado, define un `max_retention_days` explícito para los clips independiente del uso de disco, y decide de antemano por cuánto tiempo se conserva el material de personas ajenas al hogar.

```yaml
# config/hardening.yaml (v4)
hardening:
  disk_full_policy: evict_oldest_clips_keep_event_rows
  max_retention_days: 30                        # v4: age-based purge, independent of disk pressure
  log_rotation:
    handler: RotatingFileHandler
    max_bytes: 10485760      # 10 MB per file
    backup_count: 5
  storage:
    sqlite_db_path: /mnt/ssd/camera_events.db   # moved off the boot microSD before soak testing
    clips_dir: /mnt/ssd/clips/
    minimum_free_gb: 5                          # v3: eviction trigger, not "wait for ENOSPC"
  startup_checks:                               # v3: fail-fast, before the ingestion thread starts
    - hailo_device_identify
    - hef_load
    - rtc_time_plausible
    - storage_writable
  process_supervision:                          # v4: covers mid-run crash/hang, not just startup
    manager: systemd
    restart_policy: on-failure
    restart_sec: 5
    start_limit_interval_sec: 300
    start_limit_burst: 5
    watchdog_sec: 30
    watchdog_heartbeat: sd_notify
    on_failure_unit: camera-alert@%n.service   # v11: out-of-process webhook when the unit enters `failed`, incl. restart-limit exhaustion
  observability:
    host_memory_metric: process_rss_bytes       # v4: exposed on /metrics alongside NPU counters
  licensing:
    yolo_toolchain_license: AGPL-3.0
    safe_for_private_use: true
    requires_review_before_open_source: true
```

**Lecturas adicionales:**
- [Python logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html) — recetas prácticas incluyendo manejadores de archivos rotativos
- [Python `logging.handlers` reference](https://docs.python.org/3/library/logging.handlers.html) — detalles de la API de `RotatingFileHandler`/`TimedRotatingFileHandler`
- [SQLite on a Raspberry Pi (Atomic Object)](https://spin.atomicobject.com/sqlite-raspberry-pi/) — notas prácticas sobre los patrones de escritura de SQLite y consideraciones de tarjeta SD en esta clase exacta de hardware
- [SD card lifespan calculator (raspberry.tips)](https://raspberry.tips/en/sd-card-lifespan-calculator-how-long-will-your-storage-last) — una forma concreta de estimar el presupuesto de desgaste por escritura antes de mover el almacenamiento a SSD
- [Ultralytics License page](https://www.ultralytics.com/license) — términos de AGPL-3.0 vs. Enterprise
- [systemd restart/watchdog patterns for Raspberry Pi services](https://forums.raspberrypi.com/viewtopic.php?t=376126) — `Restart=`, `WatchdogSec=`, y la distinción entre cierre inesperado y bloqueo (v4)
- [Keeping software running on the Raspberry Pi (dzombak.com)](https://www.dzombak.com/blog/2023/12/keep-your-software-up-and-running-on-the-raspberry-pi/) — un recorrido práctico de confiabilidad con systemd (v4)

---

### 1.14 Canal de alertas y notificaciones — `src/notify/` *(adición v4)*

**Propósito.** Nada en §1.1–§1.13 llega jamás a un humano. Un evento emitido, persistido, e incluso enriquecido sigue siendo solo una fila en SQLite hasta que alguien consulta `/events` — y un sistema de seguridad del que nadie es notificado solo funciona como seguridad si alguien se acuerda de revisarlo. Esto es una brecha de producto, no un error de la canalización, pero es la única pieza faltante que convierte este proyecto de "una canalización funcional" en "una alerta real".

**Cómo funciona.** **Corrección v11 — el notificador no debe drenar la cola de entrada del escritor.** No adjuntes el notificador como un segundo consumidor `.get()` de la cola del hilo escritor de §1.6: la semántica estándar de las colas entrega cada elemento encolado a exactamente un consumidor, así que dos consumidores compitiendo por una sola `queue.Queue` dividen silenciosamente los eventos entre ellos — aproximadamente la mitad persistidos pero nunca alertados, la otra mitad alertados pero nunca persistidos, sin que se genere ninguna excepción. En su lugar, después de que el hilo escritor de SQLite confirma (commit) exitosamente una fila de evento, haz que publique el ID del evento confirmado en una cola de notificación separada y acotada que solo lee el notificador (o usa el observador de sondeo (polling) de `/events` ya ofrecido como alternativa, que evita por completo la cuestión de compartir la cola). Ese notificador entonces dispara una notificación push — un webhook a [ntfy](https://ntfy.sh/) o [Pushover](https://pushover.net/), o una publicación a Home Assistant/MQTT si el hogar ya tiene uno en ejecución. Mantén esto fuera de la ruta crítica de procesamiento de fotogramas de la misma forma en que ya lo están las escrituras de SQLite y el enriquecimiento: un extremo de notificación lento o inalcanzable nunca debe bloquear la ingesta.

**Adición v5 — esta es la bandeja de salida de alertas ya decidida previamente para el proyecto, no un diseño nuevo.** La forma de webhook-primero, segundo-canal-aditivo aquí ya se había resuelto para este proyecto antes de que existiera esta guía de MVP: entregar a través de un webhook genérico (ntfy/Pushover/Home Assistant satisfacen todos esto) como el canal de referencia siempre activo, y tratar una integración dedicada de bot de Telegram como un segundo canal superpuesto una vez que la ruta del webhook esté comprobada, no como un reemplazo de ella. Mantener el webhook como referencia evita acoplar la única ruta de notificación del MVP al tiempo de actividad y los límites de tasa de una sola API de terceros. Ahora que esta sección tiene su propio espacio de hito (§3), delimítala explícitamente a esa decisión en lugar de rederivarla desde cero.

**Adición v9 — el canal de notificación ahora también vigila las señales de salud del sistema, no solo los eventos emitidos.** `min_severity: entry_event` significaba que las transiciones de `/health` a degradado/caído (§1.3, §1.7, §1.10) y la señal `tamper_suspected` (§1.16) nunca llegaban a un humano — exactamente el punto ciego que esta sección existe para cerrar, solo que para fallas de infraestructura en lugar de un evento perdido. Se añade una segunda ruta de consumidor que se suscribe directamente a las transiciones de estado de `/health` y a la señal de manipulación, independiente de la cola de eventos del hilo escritor descrita arriba, y enruta ambas a través de los mismos canales de entrega (webhook primero, Telegram segundo) en su propio nivel de severidad.

**Adición v11 — un interruptor de hombre muerto, porque el consumidor v9 comparte destino con lo que vigila.** El consumidor de `/health` de arriba se ejecuta *dentro* del proceso de la canalización, así que puede reportar "degradado pero en ejecución" y nada más: un cierre inesperado, un OOM kill, systemd rindiéndose después de `start_limit_burst: 5`, un corte de energía, o una conexión de red perdida, todos lo silencian justo en el momento en que importa. Dos mecanismos cierran esto, y cubren conjuntos de fallas **disjuntos**: la unidad `OnFailure=` de §1.13 reporta un proceso que ha muerto, se ha bloqueado, o ha agotado su presupuesto de reinicios, pero necesita que systemd y la red estén vivos para hacerlo; el interruptor de hombre muerto de abajo no necesita ninguno de los dos — un trabajo de APScheduler (§1.8, ya en el stack) emite un ping HTTP a un extremo externo cada `interval_seconds`, y el servicio *externo* genera la alerta cuando los pings se detienen por más tiempo que `grace_period_seconds`. Es el único mecanismo en este documento que detecta pérdida de energía, pérdida de red, o un kernel bloqueado, precisamente porque la alerta se decide en algún lugar que el Pi no puede derribar consigo. Hacer ping es un trabajo, no una verificación de salud — no debe depender del bucle de fotogramas ni del hilo escritor, o heredaría sus modos de falla.

```yaml
# config/notify.yaml (v11)
notify:
  enabled: true
  provider: webhook             # v5: webhook is the baseline outbox channel — ntfy/pushover/mqtt are all webhook-shaped
  base_url: https://ntfy.sh/<topic>
  secondary_provider: telegram  # v5: additive second channel once the webhook path is proven, per the outbox decision above
  throttle:                     # v11.2: moved from events.yaml; throttles notifications, never events
    same_camera_same_event_seconds: 10
    record_suppressed: true     # every suppressed notification is persisted and counted
  include_enrichment_summary: true   # attach the §1.9 text summary once it's ready, don't block on it
  queue_max_size: 50
  queue_overflow_policy: drop_oldest
  min_severity: entry_event
  health_severity_tiers:               # v9: watches /health transitions and the §1.16 tamper signal, not just emitted events
    - health_degraded
    - health_down
    - tamper_suspected
  watch_health_endpoint: true          # v9: second consumer path, independent of the §1.6 writer-thread queue above
  dead_mans_switch:                    # v11: external service alerts when pings stop — the only path that survives power/network/kernel loss
    enabled: true
    ping_url: https://<dead-mans-switch-service>/ping/<check-id>   # secret-bearing URL: env var or 0600 file, never committed (same rule as §1.7)
    interval_seconds: 60
    grace_period_seconds: 180          # external alert fires after this long without a ping; ≥ 2× interval to tolerate one lost ping
```

**Lecturas adicionales:**
- [ntfy.sh documentation](https://docs.ntfy.sh/) — notificaciones push autoalojables sobre una API HTTP sencilla
- [Pushover API](https://pushover.net/api) — un servicio de notificaciones push alojado comúnmente usado para alertas de automatización del hogar

---

### 1.15 Interfaz mínima de operador — `src/ui/` *(adición v4)*

**Propósito.** La API (§1.7) es solo JSON. Ver lo que realmente ocurrió todavía significa extraer manualmente los archivos de clips del Pi — lo que hace que el MVP sea demostrable pero no genuinamente utilizable en el día a día.

**Cómo funciona.** Una sola página HTML estática, servida por la misma aplicación FastAPI, que lista los eventos recientes (miniatura o primer fotograma, marca de tiempo, zona, resumen de enriquecimiento) con una etiqueta `<video>` incrustada que apunta al clip de pre-grabación correspondiente (§1.12). Esto no necesita un framework de frontend, un paso de compilación (build), ni un esquema de autenticación más allá de lo que §1.7 ya especifica — es una vista delgada de solo lectura sobre `/events` y el directorio de clips, no una nueva capa arquitectónica.

**Lecturas adicionales:**
- [FastAPI — serving static files](https://fastapi.tiangolo.com/tutorial/static-files/) — el mecanismo mínimo necesario para servir esta página desde el mismo proceso

---

### 1.16 Detección de manipulación y obstrucción de cámara *(adición v4)*

**Propósito.** La puerta de movimiento (§1.2) no puede distinguir "nada se movió" de "alguien cubrió, dejó en negro, o reapuntó el lente" — un ataque bien conocido y específicamente relevante para la seguridad contra exactamente esta clase de sistema, y uno que el diseño actual no tiene forma de notar.

**Cómo funciona.** Añade una heurística ligera de manipulación (tamper) junto a la puerta de MOG2: una racha sostenida de fotogramas con varianza o entropía cercana a cero, especialmente combinada con un cambio abrupto de escalón en la luminancia media (lente cubierto) o un evento de movimiento de fotograma completo persistente y grande sin causa plausible (cámara reapuntada físicamente), debería levantar una señal distinta `tamper_suspected` en `/health` — separada de los estados de degradación existentes de NPU/desactualización (§1.3, §1.7, §1.10) — en lugar de leerse silenciosamente como "una escena muy tranquila".

**Corrección v5 — un cambio de modo día/noche por IR es indistinguible de la señal de manipulación de arriba a menos que se exente explícitamente.** Cada cámara de este proyecto cambia entre modos de color e iluminación IR al atardecer y al amanecer, y ese cambio en sí mismo es un cambio de escalón repentino y grande en la luminancia media — exactamente lo que `luminance_step_threshold` está diseñado para detectar. Dejado tal como se especificó, esta heurística levanta `tamper_suspected` dos veces al día, todos los días, que es la forma clásica en que un operador aprende a ignorar una señal de seguridad. Dos soluciones, cada una suficiente y más fuertes combinadas: (1) **verificación cruzada contra el propio modo día/noche reportado por la cámara** — la mayoría de las cámaras ONVIF exponen un ajuste o evento de filtro de corte IR/modo de imagen, así que un escalón de luminancia que coincide con un cambio de modo reportado es un cambio de rutina, no manipulación; (2) si esa señal no es expuesta de forma confiable por tu cámara específica, usa una **verificación de persistencia** en lugar de un umbral de escalón simple — un evento real de cobertura/oscurecimiento permanece con baja varianza *después* del escalón de luminancia (el lente está cubierto, así que la estructura nunca regresa), mientras que el escalón de un cambio de modo IR es seguido inmediatamente por que la imagen recupere varianza y detalle normales a medida que el iluminador se estabiliza. Exige `low_variance_frame_count` fotogramas consecutivos de baja varianza *siguiendo* al escalón, no solo el escalón en sí, antes de levantar `tamper_suspected`.

**Nota de alcance v11.** Esta heurística detecta la obstrucción y reapuntado del lente solo mientras el Pi permanece encendido y en ejecución — se ejecuta dentro del proceso sobre fotogramas decodificados, así que cortar la energía o la red al Pi deshabilita al detector junto con la cámara. Ese vector está cubierto en cambio por el interruptor de hombre muerto en §1.14 (v11), no por nada en esta sección.

```yaml
# config/tamper.yaml (v5)
tamper:
  enabled: true
  luminance_step_threshold: 40        # sudden mean-brightness jump between consecutive samples
  ir_cutover_grace_period_seconds: 10  # v5: suppress the step check around a reported/expected day-night mode change
  require_persistence_after_step: true # v5: step alone is not sufficient — variance must stay low afterward, not just spike momentarily
  low_variance_frame_count: 30        # consecutive near-zero-variance frames before flagging
  low_variance_threshold: 5.0
  action: mark_health_tamper_suspected
```

**Lecturas adicionales:**
- [OpenCV — image statistics and histogram basics](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html) — los bloques constructivos (media/varianza) a partir de los cuales se construye esta heurística

---

### 1.17 Pruebas y CI para componentes de función pura *(adición v4)*

**Propósito.** El banco de reproducción de §1.11 valida el sistema de extremo a extremo contra material real, lo cual es exactamente correcto para la calibración — pero es lento, requiere datos etiquetados, y es la herramienta equivocada para detectar una regresión en una función pequeña y pura. El IoU, el bloque \(Q(\Delta t)\) (§1.4/§4.1), la pertenencia de punto en polígono de zona, y el ciclo completo (round-trip) de homografía (§4.2) son todos independiente y económicamente verificables mediante pruebas unitarias, y un error en cualquiera de ellos de otro modo saldría a la luz semanas después como una caída de precisión inexplicada en el banco de pruebas en lugar de como una prueba fallida hoy.

**Corrección v6 — estas pruebas se escriben junto con el código que las necesita, no se posponen a una limpieza posterior.** Esta sección describe lo que cubre la suite de pruebas de función pura; ya no es un elemento independiente del backlog de fase 2 (ver §3). El IoU y el bloque \(Q(\Delta t)\) se escriben en el hito 5, cuando se construye el tracker; el punto en polígono se escribe en el hito 6, cuando se construye la lógica de zonas. Solo la prueba del ciclo completo de homografía sigue siendo genuinamente de fase 2, ya que §4.2 en sí es un refinamiento de fase 2 que aún no está en la ruta crítica del MVP.

**Cómo funciona.** Una pequeña suite de `pytest`, ejecutada en CI en cada commit, que cubre: el IoU contra casos calculados a mano (cajas idénticas, sin superposición, superposición parcial); el bloque \(Q(\Delta t)\) contra los valores de forma cerrada en §4.1 para unos cuantos \(\Delta t\) elegidos; el punto en polígono contra una forma de zona conocida y un puñado de puntos dentro/fuera/en el borde; y el ciclo completo de homografía (proyectar cuatro puntos de calibración a través de \(H\) y de vuelta, verificar un error de subpíxel/subcentímetro). Nada de esto reemplaza al banco de reproducción — detecta una clase de error distinta y más económica antes de que llegue siquiera al material real.

**Lecturas adicionales:**
- [pytest documentation](https://docs.pytest.org/) — el ejecutor de pruebas sobre el que se construye esta suite
- [GitHub Actions — Python CI quickstart](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) — conectando la suite de pytest a CI en cada push

---

### 1.18 Validación de configuración e invariantes de arranque — `src/config/validate.py` *(adición v9)*

**Propósito.** Cuatro de las ocho revisiones de este documento fueron causadas por el mismo problema subyacente: la configuración vive repartida en once archivos YAML independientes sin esquema y sin verificación de invariantes entre archivos, así que un valor puede desincronizarse de un valor relacionado en otro lugar — silenciosamente, hasta que sale a la luz como un error en tiempo de ejecución o una métrica engañosa. v1 duplicó un campo de conteo de confirmación bajo dos nombres distintos (§1.4); v6 descubrió que `kalman_sigma_accel` había sido mal nombrado para sus propias unidades; v7 encontró que el valor derivado de ese error de nombrado era en sí mismo aritméticamente incorrecto (§1.4); v8 encontró que las fechas de división de `eval.yaml` contradecían la tabla de hitos (§1.11/§3). Cada uno de estos solo se detectó porque un humano releyó varios archivos lado a lado.

**Cómo funciona.** Añade un paso de validación que se ejecuta antes de que inicie el hilo de ingesta (el mismo punto en la secuencia de arranque que las verificaciones del dispositivo NPU en §1.3 y la lista `startup_checks` en §1.13), verificando invariantes que este documento ya declara en prosa pero que nunca ha aplicado en código:

```python
# src/config/validate.py (v9)
class ConfigError(Exception):
    """Raised at startup when a cross-file config invariant is violated. Never a warning — refuse to start."""

def validate_config(cfg) -> None:
    if cfg.detector.confidence_threshold != cfg.tracker.low_confidence_threshold:
        raise ConfigError(
            f"detector.confidence_threshold ({cfg.detector.confidence_threshold}) must equal "
            f"tracker.low_confidence_threshold ({cfg.tracker.low_confidence_threshold}) — "
            "the two-tier association in §1.4 depends on the detector never discarding "
            "boxes below the tracker's low-confidence tier."
        )
    for camera_id in cfg.cameras:
        zone_path = cfg.events.zone_rules.zone_definitions / f"{camera_id}.json"
        if not zone_path.exists():
            raise ConfigError(f"missing zone polygon file for camera '{camera_id}': {zone_path}")
```

Esto verifica exactamente los dos invariantes nombrados arriba — `detector.confidence_threshold == tracker.low_confidence_threshold` (§1.3/§1.4) y que cada cámara configurada tenga un archivo de polígono de zona (§1.2/§1.5) — y genera una excepción en lugar de una advertencia, coincidiendo con la filosofía de fallo temprano ya usada para las verificaciones de NPU y almacenamiento en otras partes de este documento. No reemplaza la revisión cuidadosa de los once archivos YAML; detecta la clase específica de desincronización que ya ha causado errores reales en la propia historia de este proyecto. La validación se ejecuta solo en el arranque: un cambio de configuración solo surte efecto después de un reinicio del servicio, y no existe ni está planeada ninguna ruta de recarga en caliente (hot-reload) para el MVP (ver `docs/post-mvp-backlog.md`).

**Lecturas adicionales:**
- [Pydantic Settings documentation](https://pydantic.dev/docs/validation/latest/concepts/pydantic_settings/) — una capa de configuración tipada/validación ya lista sobre la que se puede construir este módulo

---

## 2. Cómo se conectan las etapas

| Etapa | Consume | Produce | Modo de falla contra el que protege |
|---|---|---|---|
| **Presupuesto de píxeles (§0)** | Especificación del lente + distancia de zona | Decisión de continuar o no sobre el modelo, el lente, la resolución | Construir todo el pipeline alrededor de una persona demasiado pequeña para que YOLOv8n la detecte |
| Ingesta (buzón con hilos, reconexión, detección de datos obsoletos) | IP de la cámara + credenciales ONVIF | Fotogramas decodificados, marcas de tiempo monotónicas + UTC | Adivinar la ruta RTSP, muerte silenciosa del stream, el "embarrado de OpenCV", cámaras congeladas pero "vivas" |
| Puerta de movimiento (MOG2, `== 255`, tasa de aprendizaje con Δt, máscara de zona) | Fotogramas decodificados | Decisión de continuar o descartar | Ciclos de NPU desperdiciados; disparos falsos por árboles/sombras/insectos IR; sensibilidad de la puerta dependiente de la carga |
| Detector YOLOv8n (con letterbox, piso de 0.15, fallo escandaloso) | Fotogramas que pasaron la puerta | Cuadros delimitadores + clase + confianza | Modelo no compatible, distorsión por estiramiento de aspecto, privar de datos al nivel de baja confianza del tracker, muerte silenciosa de la NPU |
| Tracker (Kalman parametrizado en el tiempo, `Q(Δt)` cúbica, IoU) | Cuadros por fotograma | `track_id` persistentes + `is_confirmed` | Alertas falsas provocadas por parpadeo; extrapolación incorrecta y pérdida de identidad tras saltos largos de la puerta |
| Máquina de estados de eventos (solo entrada para el MVP) | Tracks confirmados + polígonos de zona | Eventos emitidos | Alertas duplicadas, disparo prematuro, deriva de configuración en los umbrales de confirmación |
| Búfer circular de pre-grabación (PyAV, alineado a keyframes) | Paquetes del mainstream | Clips de pre/post-grabación | Clips que inician después de que comenzó la acción; cabezas de clip indecodificables a mitad de GOP |
| Almacén SQLite (hilo escritor asíncrono) | Eventos emitidos | Filas durables, ambas marcas de tiempo | Pérdida de eventos; un checkpoint de WAL que paraliza el bucle de detección |
| FastAPI + `/metrics` + `/health` de tres estados | Eventos almacenados | Respuestas HTTP, métricas de latencia, salud | La API bloqueando la detección; excesos de presupuesto invisibles; un "activo" que en realidad no está analizando |
| Enriquecimiento (cola acotada, descarta el más antiguo) | Metadatos de eventos persistidos | Resumen en lenguaje natural | Dependencia de la nube; un LLM atascado generando backpressure sobre la detección |
| Banco de reproducción sin conexión (regla de coincidencia, partición reservada, IC de Wilson) | Video grabado + CSV etiquetado | Precisión/recall + IC honestos + trazas por etapa | Calibrar umbrales sin verdad de referencia; sobreajustar un conjunto de etiquetas diminuto; un IC con subcobertura que oculta ruido real |
| Canal de notificación (v4) | Evento persistido + enriquecimiento opcional | Alerta push a un humano | Un historial de alertas del que nunca se le informa a nadie |
| UI del operador (v4) | Eventos almacenados + clips | Lista de eventos/clips visible para humanos | Un MVP que es demostrable pero no realmente usable en el día a día |
| Detección de manipulación/obstrucción (v4) | Luminancia/varianza de los fotogramas decodificados | Señal de salud `tamper_suspected` | Un lente cubierto o reapuntado que se lee como "una escena tranquila" |
| Supervisión de proceso + presupuesto de memoria (v4) | Unidad de systemd + RAM del host en `/metrics` | Reinicio automático, detección de bloqueo, advertencia de OOM | Un proceso caído o atascado que detiene silenciosamente todo el sistema |
| Pruebas unitarias / CI (v4; escritas por hito a partir de v6) | Funciones puras (IoU, `Q(Δt)`, polígono, homografía) | Aprobado/fallido en cada commit | Una regresión barata que aparece semanas después como una caída inexplicable de una métrica del banco de reproducción |

---

## 3. Lista de verificación de hitos y cronograma propuesto

El ritmo propuesto asume trabajo incremental de medio tiempo (tardes/fines de semana), comenzando la semana del **14 de septiembre de 2026**. Los elementos deliberadamente diferidos más allá del MVP se rastrean en `docs/post-mvp-backlog.md` (v11), no en esta guía.

**Reordenado por prioridad en v2:** el elemento de mayor riesgo (compatibilidad del HEF de Hailo) se ejecuta primero como un spike aislado sin conexión, la ingesta RTSP se mueve a la semana 2, el banco de reproducción/evaluación se construye *antes* de ajustar cualquier umbral, y la máquina de estados de eventos se entrega solo con entrada.

**Corregido aún más en v3:** un presupuesto de píxeles preliminar (§0) se ejecuta antes del hito 1 porque puede invalidar la elección del modelo o del lente; **la grabación continua de video comienza pasivamente al final del hito 2**, ya que acumular metraje de anochecer/amanecer/viento/entregas se mide en *semanas calendario* y no es apenas parte de tu tiempo de trabajo; **los polígonos de zona se elaboran en el hito 3**, porque la máscara ROI de la puerta de movimiento (§1.2) los consume y el hito 4 de otra forma no puede avanzar; y el hito 8 de v2 se divide, porque agrupaba los dos elementos restantes de mayor riesgo (pre-grabación y enriquecimiento) en una sola semana justo antes de la prueba de resistencia (soak test).

**Refinado aún más en v5:** el hito 1 ahora mide el costo real de decodificación de §1.10 contra metraje grabado en lugar de arrastrar un presupuesto asumido sin verificar; y **el hito 7 se divide** — la persistencia/API/`/metrics` y el canal de notificación se agrupaban en una sola semana en v4, lo cual la sobrecargaba, así que la notificación ahora tiene su propio espacio (hito 8), lo que retrasa una semana cada hito posterior.

| # | Hito | Ventana objetivo | Criterios de salida |
|---|---|---|---|
| **0** | **Presupuesto de píxeles y decisión de lente** | 8 sep – 13 sep, 2026 | Altura en píxeles medida de una persona en el límite de zona de cada cámara; decisión escrita sobre el lente (2.8 vs 4.0 mm), resolución del substream, inferencia con recorte vs. fotograma completo, y YOLOv8n vs. YOLOv8s (§0) |
| 1 | YOLOv8n en Hailo-10H — spike sin conexión | 14 sep – 20 sep, 2026 | `yolov8n_h10h.hef` carga y devuelve detecciones de personas contra un `.mp4` **grabado** (no se necesita cámara en vivo); salida HEF INT8 verificada contra Ultralytics FP32 en los mismos fotogramas (§1.3); implementadas la política de fallo temprano en el arranque y de fallo mid-run de la NPU; **v5: costo de decodificación por software H.264 del substream objetivo medido directamente contra el `.mp4` grabado (duración por fotograma y utilización del núcleo), reemplazando el presupuesto asumido `decode: 25 ms` por un valor medido (§1.10)** |
| 2 | Ingesta RTSP + captura con hilos | 21 sep – 27 sep, 2026 | Substream en vivo a través del buzón de una sola ranura (§1.10) con reconexión de backoff exponencial y detección de datos obsoletos; cada fotograma lleva marcas de tiempo monotónicas y UTC; GOP de la cámara configurado a ~1 s (§1.1); **v11: credenciales ONVIF cargadas desde variables de entorno; ninguna credencial presente en ningún lugar del repositorio (§1.1)**. **Iniciar la grabación continua 24/7 al final de esta semana y dejarla corriendo hasta el hito 6** |
| 3 | Polígonos de zona, etiquetado y banco de reproducción | 28 sep – 4 oct, 2026 | Polígonos de zona elaborados en `configs/zones/` (§1.5); CSV de verdad de referencia etiquetado a partir del metraje de la semana 2, **explícitamente dividido en `day` / `night_ir` (§1.3, v4)**; `replay_runner.py` (§1.11) reproduce a través del pipeline real con un reloj simulado y reporta precisión/recall **con la regla de coincidencia y la partición reservada configuradas explícitamente**, usando un **intervalo de puntaje de Wilson y trazas por etapa en lugar de un IC de Wald (§1.11, v4)** — construido antes de ajustar cualquier umbral; **v11: la prueba negativa `night_ir_insect` pasa — cero eventos emitidos en las ventanas etiquetadas de actividad de insectos (§1.11)** |
| 4 | Puerta de movimiento calibrada mediante el banco | 5 oct – 11 oct, 2026 | MOG2 + máscara derivada de zona (§1.2) con la prueba de sombra `== 255` y `learningRate` escalado por Δt; `minimum_motion_ratio` ajustado reejecutando el banco, no observando a simple vista un feed en vivo |
| 5 | Tracker Kalman/IoU consciente del tiempo | 12 oct – 18 oct, 2026 | El tracker posee `min_confirmed_hits` y expone `track.is_confirmed`; `F(Δt)` se reconstruye en cada paso de predicción y `Q(Δt)` usa el bloque de ruido blanco continuo (§1.4/§4.1); la expiración es `max_missed_seconds`; `kalman_sigma_accel_sq` ajustado mediante el banco; **v6: pruebas unitarias de función pura de `IoU` y `Q(Δt)` escritas contra los valores de forma cerrada de §4.1** |
| 6 | Máquina de estados de eventos — solo entrada | 19 oct – 25 oct, 2026 | La máquina de estados lee `track.is_confirmed` (sin conteo duplicado de hits); solo `entry_event_enabled` está activo; tasa de FP validada en el día **reservado** antes de iniciar el cruce/permanencia (§1.5); **v6: prueba unitaria de pertenencia a zona por punto-en-polígono escrita contra una forma de zona conocida** |
| 7 | Persistencia + FastAPI + `/metrics` | 26 oct – 1 nov, 2026 | El hilo escritor asíncrono mantiene SQLite fuera de la ruta de fotogramas (§1.6); `/health` es de tres estados; `/metrics` reporta la utilización de throughput por etapa *y* la latencia end-to-end por separado contra los presupuestos divididos de §1.10; **v4: Uvicorn vinculado a una interfaz específica con verificación de token portador en rutas que no son `/health` (§1.7)** |
| 8 | Alertas y canal de notificación (v5) | 2 nov – 8 nov, 2026 | Consumidor de notificaciones conectado a una cola de notificación dedicada post-commit alimentada por el hilo escritor de §1.6 — nunca un segundo consumidor de la propia cola de entrada del escritor — disparando una vez que cada evento se confirma (§1.14); la entrega pasa primero por webhook, según la decisión de outbox del proyecto, con Telegram conectado como un segundo canal adicional; los fallos de entrega se registran y reintentan con backoff sin bloquear la ingesta; evento sintético end-to-end entregado a través de ambos canales |
| 9 | Búfer de evidencia de pre-grabación | 9 nov – 15 nov, 2026 | El búfer circular de paquetes de PyAV vacía clips de pre/post-grabación alineados a keyframes vinculados a las filas de eventos; se registra la duración de pre-grabación lograda; la segunda conexión RTSP tiene su propia reconexión y contadores (§1.12) |
| 10 | Enriquecimiento local (Hailo-Ollama) | 16 nov – 22 nov, 2026 | Resúmenes solo de texto generados de forma asíncrona; `/hailo/v1/list` accesible solo en localhost; latencia del detector remedida con el LLM residente en la NPU (§1.9) |
| 11 | Endurecimiento, prueba de resistencia y demo | 23 nov – 29 nov, 2026 | BD/clips en SSD USB, batería RTC instalada, rotación de logs y desalojo por disco lleno verificados, verificaciones de arranque en la unidad de systemd (§1.13); **v4: `Restart=on-failure` + `RestartSec` + `StartLimitBurst` + latido de `WatchdogSec`/`sd_notify` configurado, y presupuesto de memoria del host expuesto en `/metrics` (§1.13)**; **v11: `OnFailure=camera-alert@%n.service` configurado y verificado matando el proceso del pipeline y recibiendo el webhook (§1.13); dead-man's switch verificado apagando la Pi y confirmando que llega la alerta externa (§1.14)**; prueba de resistencia de 24 horas sin caídas, bloqueos ni crecimiento de memoria; pipeline completo demostrado en una cámara |

**Backlog de fase 2 (v4):** la detección de manipulación/obstrucción de cámara (§1.16) y la UI mínima de operador (§1.15) son valiosas pero no están en la ruta crítica hacia una demo de MVP de una cámara — pertenecen junto a los refinamientos matemáticos existentes de §4 como endurecimiento post-MVP, no como bloqueadores para el hito 11 (renumerado en v5).

**Corrección v6 — la suite de pruebas unitarias de función pura/CI (§1.17) ya no forma parte de este backlog.** Diferir estas pruebas a una pasada de limpieza post-MVP significó que cada función pura en la que habrían detectado errores — incluyendo la aproximación de \(\Delta t^2\) (§1.4/§4.1) y el error de unidades \(\sigma\) contra \(\sigma^2\) recién corregido en §1.4 — se entregó y corrió sin ser detectada durante varios hitos primero. Cada prueba de función pura ahora se escribe en el hito que toca por primera vez esa función, en lugar de agruparse en una única semana posterior: IoU y el bloque \(Q(\Delta t)\) se prueban en el hito 5 (§1.4/§4.1, donde se construye el tracker), y la pertenencia a zona por punto-en-polígono se prueba en el hito 6 (§1.5, donde se construye la lógica de zona). Una prueba de \(Q(\Delta t)\) contra los valores de forma cerrada de §4.1 cuesta alrededor de una hora escribirla y habría detectado tanto la aproximación de \(\Delta t^2\) como el error de unidades en el momento en que se introdujo cualquiera de los dos, en lugar de semanas después como una caída inexplicable de una métrica del banco.

**Lista de verificación de preparación para la demo del MVP:**

*Preliminar*
- [ ] Altura en píxeles de una persona medida en el límite de zona de cada cámara; decisión de lente/resolución/modelo por escrito (§0)

*Detección*
- [ ] `yolov8n_h10h.hef` corre de extremo a extremo a la latencia esperada, con precisión INT8-vs-FP32 verificada en metraje grabado
- [ ] El detector emite con `confidence_threshold: 0.15`, coincidiendo exactamente con `low_confidence_threshold` del tracker
- [ ] Relleno de letterbox con la transformación afín inversa aplicada a los cuadros devueltos, no estiramiento a cuadrado
- [ ] Política de fallo de la NPU: fallo temprano en el arranque, fallo escandaloso a mitad de ejecución, `/health` degradado, nunca continuar silenciosamente

*Puerta de movimiento*
- [ ] Máscara de MOG2 probada con `== 255`, **no** `> 0`, para que `detectShadows: true` realmente suprima sombras
- [ ] `learningRate` pasado explícitamente y escalado por el Δt medido, no derivado de un historial de conteo de fotogramas
- [ ] Proporción de movimiento restringida a una máscara ROI derivada de zona, y umbrales calibrados mediante el banco

*Seguimiento*
- [ ] `min_confirmed_hits` definido una sola vez, en `tracker.yaml`, leído en todas partes como `track.is_confirmed`
- [ ] `F(Δt)` reconstruido a partir del tiempo transcurrido de reloj real medido en cada paso de predicción
- [ ] `Q(Δt)` usa el bloque de ruido blanco continuo (⅓Δt³ / ½Δt² / Δt), no la aproximación de Δt²
- [ ] Expiración de track evaluada en segundos (`max_missed_seconds`), nunca en conteos de fotogramas
- [ ] Un track confirmado sobrevive breves huecos de detección y oclusiones cortas sin perder identidad

*Eventos*
- [ ] Eventos de zona emitidos solo desde `track_id` confirmados; solo la regla de zona de entrada está habilitada
- [ ] El enfriamiento de mismo-track/misma-zona previene alertas duplicadas durante una sola visita
- [ ] Tracks confirmados distintos que entran a la misma zona producen eventos de entrada distintos (v11.2)
- [ ] El notificador limita misma cámara/tipo de evento dentro de 10 s y registra cada notificación suprimida (v11.2)

*Ingesta y temporización*
- [ ] Buzón de una sola ranura con reconexión de backoff exponencial y una señal de datos obsoletos, no un bucle activo
- [ ] Ambas marcas de tiempo, monotónica y UTC, capturadas en el momento de lectura y persistidas; pila de moneda RTC instalada
- [ ] Throughput y latencia end-to-end rastreados como presupuestos **separados**, a ~65% de utilización del periodo de fotograma
- [ ] Escrituras de SQLite en un hilo dedicado, fuera de la ruta por fotograma
- [ ] Credenciales ONVIF cargadas desde variables de entorno; ninguna credencial presente en ningún lugar del repositorio (v11)

*Evaluación*
- [ ] Regla de coincidencia del banco (ventana, cardinalidad, manejo de no coincidencias) configurada explícitamente en `eval.yaml`
- [ ] El día reservado nunca se usa para ajustar; los números reportados provienen de él, con tamaño de muestra e IC
- [ ] Intervalos de confianza calculados con un intervalo de puntaje de Wilson, no una aproximación de Wald/normal (v4)
- [ ] Etiquetas de verdad de referencia divididas en `day` / `night_ir`, con el recall del detector por fotograma reportado por separado de las métricas a nivel de evento (v4)
- [ ] El banco de reproducción emite trazas por etapa (detección / track-confirmado / evento) junto con la precisión/recall final (v4)
- [ ] La prueba negativa `night_ir_insect` pasa — cero eventos emitidos en las ventanas etiquetadas de actividad de insectos (v11)

*Evidencia y operaciones*
- [ ] Los clips de pre-grabación están alineados a keyframes mediante PyAV e inician antes de la primera detección del track confirmado
- [ ] Intervalo de GOP de la cámara reducido a ~1 s; duración de pre-grabación lograda registrada por evento
- [ ] BD y clips en SSD USB; rotación de logs y desalojo por disco lleno verificados antes de la prueba de resistencia
- [ ] `hailo-ollama` accesible solo en `127.0.0.1`; el enriquecimiento VLM permanece deshabilitado hasta probarse por separado
- [ ] Implicaciones de AGPL-3.0 revisadas antes de cualquier decisión de hacer el repositorio de código abierto
- [ ] Unidad de systemd configurada con `Restart=on-failure`, `RestartSec`, `StartLimitIntervalSec`/`StartLimitBurst`, y latido de `WatchdogSec`/`sd_notify` (v4)
- [ ] Presupuesto de memoria del host estimado con el modelo de enriquecimiento residente; expuesto en `/metrics` (v4)
- [ ] Purga de `max_retention_days` definida independientemente del desalojo por disco lleno; postura de privacidad sobre visitantes ajenos al hogar decidida (v4)
- [ ] API vinculada a una interfaz específica con verificación de token portador en rutas que no son `/health` (v4)
- [ ] Canal de notificación dispara al confirmarse el evento sin bloquear la ingesta (v4)
- [ ] Unidad de alerta `OnFailure=` verificada matando el proceso; dead-man's switch verificado apagando la Pi y recibiendo la alerta externa (v11)

Una vez que esta lista de verificación esté completamente en verde para una cámara, la siguiente fase es generalizar a múltiples cámaras con aislamiento acotado de workers por cámara, verificaciones de salud y manejo de backpressure.

**Corrección v6 — el cambio de decodificación H.265/HEVC pertenece aquí, no en los criterios de salida del hito 2 ni en la lista de verificación de preparación de una sola cámara.** La propia medición citada de la guía para la bandera `-hwaccel drm` en la Pi 5 es un ahorro de 13% → 9% de un núcleo de CPU (§1.10) — aproximadamente 1% de la CPU total en un pipeline de una sola cámara, no un cambio que valga la pena planear desde ahora, y uno que también afecta la ruta de remux de PyAV de §1.12, ya que el manejo de paquetes del búfer de pre-grabación asume el codec que sea que el mainstream esté entregando. Ese cálculo cambia una vez que las cámaras se generalizan: con cuatro streams de cámara, el mismo ahorro por núcleo se multiplica por cuatro, momento en el cual puede ser la diferencia entre encajar cómodamente en el presupuesto de CPU de la Pi 5 y no hacerlo. Si las cámaras soportan streaming de substreams H.265/HEVC (la mayoría de las cámaras PoE modernas lo hacen), reconfigura el codec del substream y decodifica a través del bloque HEVC de hardware real de la Pi 5 en lugar de H.264 por software como parte de este trabajo de generalización multi-cámara — un cambio de configuración del lado de la cámara, no un cambio de código. Configura correctamente la bandera: usa **`-hwaccel drm`**, no `v4l2m2m` — la Pi 5 no usa la misma ruta de decodificación V4L2 M2M que usaba la Pi 4, y el backend predeterminado FFmpeg/GStreamer de OpenCV no elige automáticamente un decodificador de hardware solo porque exista uno para el codec negociado, así que esto todavía debe configurarse explícitamente.

**Adición v10 — el etiquetado continúa después del hito 3 de forma continua.** Los criterios de salida del hito 3 etiquetan un CSV de verdad de referencia a partir de una semana de metraje, lo cual por sí solo es poco probable que alcance `minimum_labeled_events_for_hard_gate: 100` (§1.11). Sigue agregando eventos etiquetados a partir de la grabación continua 24/7 en curso (iniciada al final del hito 2) a un ritmo regular después de que se entregue el hito 3, en lugar de tratar esa única pasada de etiquetado de una semana como el conjunto final de verdad de referencia del banco.

---

## 4. Refinamientos matemáticos de fase 2

Dos optimizaciones más profundas para tener en reserva mientras el sistema escala. Ninguna bloquea el MVP; ambas se vuelven valiosas una vez que tienes en marcha el banco de §1.11 para demostrar que ayudaron.

### 4.1 La física de la covarianza de ruido del proceso \(Q(\Delta t)\)

§1.4 señala que la covarianza de ruido del proceso debe escalar con el tiempo transcurrido. El atajo conveniente \(Q(\Delta t) \propto \Delta t^2\) es una aproximación; la forma físicamente fundamentada proviene de un modelo de aceleración de ruido blanco continuo, en el cual la aceleración no modelada se trata como un proceso de ruido de media cero integrado sobre el intervalo.

Bajo ese modelo, los dos componentes crecen a tasas diferentes. La incertidumbre de velocidad acumula la integral del ruido de aceleración, así que crece **linealmente** con \(\Delta t\). La incertidumbre de posición acumula la integral de un error de velocidad que ya está creciendo, así que crece con el **cubo** de \(\Delta t\). Para cada par independiente de posición/velocidad, el bloque exacto es:

\[
Q = \begin{bmatrix} \tfrac{1}{3}\Delta t^{3} & \tfrac{1}{2}\Delta t^{2} \\[2pt] \tfrac{1}{2}\Delta t^{2} & \Delta t \end{bmatrix} \sigma_a^{2}
\]

donde \(\sigma_a^2\) es la varianza de la aceleración del objeto — físicamente, qué tan abruptamente esperas que una persona cambie de velocidad o dirección. Un corredor que podría desviarse lateralmente justifica un \(\sigma_a^2\) mayor que un repartidor caminando en línea recta hacia una puerta.

Los términos fuera de la diagonal \(\tfrac{1}{2}\Delta t^2\) no son decoración: codifican la **correlación** entre el error de posición y de velocidad, la cual el escalamiento ingenuo de \(\Delta t^2\) descarta por completo. Un filtro que ignora esa correlación juzga mal la forma de su propia elipse de incertidumbre, no solo su tamaño.

**Por qué esto importa específicamente en este pipeline.** La puerta de movimiento (§1.2) puede producir valores de \(\Delta t\) de varios segundos — una inferencia de latido tras un intervalo silencioso, o un track que se reanuda tras una oclusión larga. Con \(\Delta t = 3\) s el término cúbico es aproximadamente un orden de magnitud mayor que la aproximación cuadrática. Usar la aproximación deja al filtro **sobreconfiado** sobre dónde está la persona: reporta una covarianza estrecha, la puerta de asociación de IoU es correspondientemente estrecha, la detección real cae fuera de ella, la coincidencia falla, y el track muere y renace con un `track_id` nuevo. Eso es precisamente la ruptura de identidad que el tracker existe para prevenir, y se presentará como un misterioso bug de alerta duplicada en lugar de como un problema del filtro.

**Implementación.** `filterpy.common.Q_continuous_white_noise(dim, dt, spectral_density)` produce este bloque directamente, así que el cambio práctico es una bandera de configuración y un solo punto de llamada; `spectral_density` es exactamente \(\sigma_a^2\), que es lo que `kalman_sigma_accel_sq` (§1.4, v6) ahora nombra con precisión — pásalo directamente, no su raíz cuadrada. Trata `kalman_sigma_accel_sq` como un ajustable calibrado a través del banco de §1.11 en lugar de una constante adivinada una sola vez — es exactamente el tipo de parámetro que parece arbitrario hasta que puedes medir la persistencia de identidad del track contra metraje etiquetado, y debido a que el tracker corre en espacio de píxeles (§4.2), sus unidades son px²/s⁴, no una varianza de aceleración métrica.

La variante de ruido blanco discreto (`Q_discrete_white_noise`) asume que la aceleración es constante *dentro* de cada intervalo y cambia solo entre ellos, lo cual es el mejor modelo cuando \(\Delta t\) es pequeño y uniforme. Dados los intervalos deliberadamente irregulares y a veces largos de este pipeline, la forma continua es el mejor ajuste.

### 4.2 Invariancia de profundidad espacial mediante homografía (mapeo de perspectiva inversa)

`tracker.yaml` especifica `max_centroid_distance_px: 120` como una verificación de sensatez de asociación. El problema es que una imagen de cámara 2D es una **proyección en perspectiva**, así que un píxel no es una unidad de distancia: 120 px cerca del horizonte podrían abarcar 10 metros de suelo real, mientras que 120 px en el primer plano podrían abarcar medio metro. Un solo umbral de píxeles es, por lo tanto, simultáneamente demasiado permisivo para objetos lejanos (permitiendo que el tracker intercambie identidades entre dos personas lejanas) y demasiado restrictivo para los cercanos (rompiendo el track de alguien que camina rápidamente frente a la cámara). La misma distorsión afecta las estimaciones de velocidad del filtro de Kalman: una persona caminando a velocidad constante parece acelerar a medida que se acerca al lente.

**El mapeo de perspectiva inversa (IPM)** elimina la distorsión proyectando las coordenadas de la imagen sobre el plano de suelo real. Debido a que un mapeo de plano a plano bajo proyección en perspectiva es una homografía, la transformación es una única matriz 3×3 \(H\) aplicada en coordenadas homogéneas:

\[
\begin{bmatrix} X' \\ Y' \\ w \end{bmatrix} = H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \qquad (X, Y) = \left(\tfrac{X'}{w},\ \tfrac{Y'}{w}\right)
\]

**Calibrar \(H\) es un procedimiento único y de baja tecnología.** Elige cuatro puntos en una superficie plana dentro del campo de visión de la cámara cuyas posiciones en el mundo real puedas medir — esquinas de una entrada vehicular, losas de pavimento, o cuatro marcadores que coloques con una cinta métrica. Anota sus coordenadas en píxeles \((u_i, v_i)\) en un fotograma capturado y sus coordenadas métricas de suelo \((X_i, Y_i)\). Cuatro correspondencias determinan completamente \(H\):

```python
import cv2, numpy as np

image_pts  = np.float32([[412, 688], [905, 690], [1102, 431], [246, 428]])   # pixels
ground_pts = np.float32([[0.0, 0.0], [3.5, 0.0],  [3.5, 8.0],  [0.0, 8.0]])  # metres

H, _ = cv2.findHomography(image_pts, ground_pts)

def to_ground(u, v, H=H):
    p = H @ np.array([u, v, 1.0])
    return p[0] / p[2], p[1] / p[2]      # (X, Y) in metres
```

Usa el punto **inferior-central** del track (los pies, donde la persona toca el plano de suelo) en lugar del centroide del cuadro delimitador — la homografía solo es válida para puntos sobre el plano al que fue calibrada, y un centroide de torso flota por encima de él.

**Qué te da esto.** La puerta de asociación, la velocidad y la distancia de permanencia se calculan todas en metros en lugar de píxeles, así que el tracker se vuelve invariante a qué tan cerca está una persona del lente. `max_centroid_distance_px: 120` se convierte en algo como `max_association_distance_m: 1.5`, un umbral con un significado físico sobre el que puedes razonar y que se transfiere sin cambios entre cámaras y ubicaciones — lo cual importa directamente para el objetivo de portabilidad del documento de arquitectura, ya que un umbral de píxeles ajustado en una casa no tiene sentido en la siguiente.

La misma dependencia de profundidad aplica a `kalman_sigma_accel_sq` (§1.4): la derivación ahí muestra que \(\sigma_a^2\) varía aproximadamente 9× entre 5 m y 15 m en el mismo fotograma, precisamente porque es una cantidad en espacio de píxeles y los píxeles-por-metro cambian con la distancia. Una vez que el filtro de Kalman opera sobre coordenadas del plano de suelo provenientes de esta homografía, \(\sigma_a^2\) se convierte en una varianza de aceleración física en m\(^2\)/s\(^4\) — aproximadamente 2–3 m\(^2\)/s\(^4\) para una persona caminando — una única constante para toda la zona en lugar de una que debe rederivarse por banda de distancia. Conlleva el mismo beneficio de portabilidad que `max_association_distance_m`: se transfiere sin cambios entre cámaras y ubicaciones, en lugar de estar ligada a una sola combinación de lente/distancia.

También habilita reglas de eventos que los píxeles no pueden expresar honestamente: la velocidad real de caminata (útil para separar una visita de entrega de alguien que está deambulando), la distancia real a la puerta, y áreas de zona en metros cuadrados.

**Advertencias.** La homografía solo es válida para el plano sobre el que fue calibrada, así que una entrada vehicular inclinada, escalones, o un patio elevado necesitan cada uno su propia \(H\) o un tratamiento por partes. También debe recalibrarse cada vez que la cámara se mueve o reapunta, lo cual la convierte en una compañera natural de un paquete de configuración por ubicación en lugar de algo integrado en el código.

**Lectura más profunda:**
- [OpenCV — `findHomography` and perspective transforms](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) — la llamada de calibración y la geometría subyacente
- [`filterpy.common` discretization helpers](https://filterpy.readthedocs.io/en/latest/common/discretization.html) — `Q_continuous_white_noise` vs `Q_discrete_white_noise`

---

## Fuentes

**Núcleo (v1):**

- [python-onvif-zeep](https://github.com/FalkTannhaeuser/python-onvif-zeep)
- [ONVIF Core Specification](https://www.onvif.org/specs/core/ONVIF-Core-Specification.pdf)
- [OpenCV VideoCapture documentation](https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html)
- [PyImageSearch — Basic motion detection and tracking with Python and OpenCV](https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/)
- [OpenCV absdiff / core array operations](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)
- [Detecting movement with pairwise frame subtraction](https://sam-low.com/opencv/frame-differencing.html)
- [Hailo Model Zoo (GitHub)](https://github.com/hailo-ai/hailo_model_zoo)
- [Hailo Model Zoo — yolov8n.yaml config](https://github.com/hailo-ai/hailo_model_zoo/blob/master/hailo_model_zoo/cfg/networks/yolov8n.yaml)
- [Hailo RPi5 Examples — object detection pipeline](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/basic-pipelines.md)
- [Hailo Application Code Examples (Python runtime)](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python)
- [Raspberry Pi AI accelerator documentation](https://www.raspberrypi.com/documentation/computers/ai.html)
- [ByteTrack (ECCV 2022, GitHub)](https://github.com/FoundationVision/ByteTrack)
- [SORT — Simple Online and Realtime Tracking (GitHub)](https://github.com/abewley/sort)
- [PyImageSearch — Intersection over Union (IoU) for object detection](https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/)
- [kalmanfilter.net — Kalman Filter Explained Through Examples](https://www.kalmanfilter.net/)
- [Hikvision — Line Crossing Detection](https://enpinfo.hikvision.com/hkwsen/unzip/20230410194813_20373_doc/GUID-246BF07A-3F33-48FC-99D9-DE1AFC3E9144.html)
- [yas-sim/object-tracking-line-crossing-area-intrusion](https://github.com/yas-sim/object-tracking-line-crossing-area-intrusion)
- [Debounce design pattern (community.openhab.org)](https://community.openhab.org/t/design-pattern-debounce/101566)
- [Python sqlite3 module documentation](https://docs.python.org/3/library/sqlite3.html)
- [SQLite WAL mode documentation](https://sqlite.org/wal.html)
- [FastAPI official tutorial](https://fastapi.tiangolo.com/tutorial/)
- [FastAPI project homepage](https://fastapi.tiangolo.com/)
- [APScheduler documentation](https://apscheduler.readthedocs.io/)
- [Hailo Model Zoo GenAI (GitHub) — Hailo-Ollama](https://github.com/hailo-ai/hailo_model_zoo_genai)
- [Raspberry Pi AI HAT+ 2 — Hailo-10H local LLM walkthrough](https://raspberry.tips/en/raspberrypi-tutorials/raspberry-pi-ai-hat-2-hailo-10h-40-tops-local-llms)
- [Hailo GenAI Model Explorer — VLM models](https://hailo.ai/products/hailo-software/model-explorer/generative-ai/type/vlm/)

**Añadidas en la revisión arquitectónica v2:**

- [OpenCV `BackgroundSubtractorMOG2` reference](https://docs.opencv.org/4.x/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
- [OpenCV background subtraction tutorial](https://docs.opencv.org/4.x/de/df4/tutorial_js_bg_subtraction.html)
- [Ultralytics `data.augment` API reference (LetterBox)](https://docs.ultralytics.com/reference/data/augment)
- [Ultralytics letterbox preprocessing discussion (GitHub issue)](https://github.com/ultralytics/ultralytics/issues/2580)
- [YOLOv5 letterbox PR reference (GitHub)](https://github.com/ultralytics/yolov5/pull/9213)
- [Stack Overflow — OpenCV VideoCapture lag due to the capture buffer](https://stackoverflow.com/questions/30032063/opencv-videocapture-lag-due-to-the-capture-buffer)
- [PyImageSearch — Faster video file FPS with cv2.VideoCapture and OpenCV](https://pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/)
- [PyImageSearch — Increasing webcam FPS with a threaded video stream](https://pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/)
- [Cross Validated — Kalman smoothing with irregular time steps](https://stats.stackexchange.com/questions/49300/how-does-one-apply-kalman-smoothing-with-irregular-time-steps)
- [filterpy issue — handling variable dt](https://github.com/rlabbe/filterpy/issues/196)
- [Ultralytics License page](https://www.ultralytics.com/license)
- [Ultralytics AGPL licensing discussion (GitHub issue)](https://github.com/ultralytics/ultralytics/issues/5691)
- [Ultralytics YOLOv8 model docs](https://docs.ultralytics.com/models/yolov8)
- [Ultralytics — Hailo export integration](https://docs.ultralytics.com/integrations/hailo)
- [VisioForge — Pre-event recording guide](https://www.visioforge.com/help/docs/dotnet/mediablocks/Guides/pre-event-recording/)
- [picamera circular streams (deepwiki)](https://deepwiki.com/waveform80/picamera/4.2-circular-streams)
- [Battleroid/seccam (GitHub)](https://github.com/Battleroid/seccam)
- [prometheus-fastapi-instrumentator (GitHub)](https://github.com/trallnag/prometheus-fastapi-instrumentator)
- [SQLite on a Raspberry Pi (Atomic Object)](https://spin.atomicobject.com/sqlite-raspberry-pi/)
- [SD card lifespan calculator (raspberry.tips)](https://raspberry.tips/en/sd-card-lifespan-calculator-how-long-will-your-storage-last)
- [Python logging cookbook](https://docs.python.org/3/howto/logging-cookbook.html)
- [Python `logging.handlers` reference](https://docs.python.org/3/library/logging.handlers.html)
- [Evaluating object detection models: methods and metrics (GeeksforGeeks)](https://www.geeksforgeeks.org/computer-vision/evaluating-object-detection-models-methods-and-metrics/)
- [Object detection metrics explained (Label Your Data)](https://labelyourdata.com/articles/object-detection-metrics)

**Añadidas en la revisión v3:**

- [PyAV documentation](https://pyav.org/docs/stable/) — demux/remux a nivel de paquete para el búfer circular de pre-grabación
- [`filterpy.common` discretization helpers](https://filterpy.readthedocs.io/en/latest/common/discretization.html) — `Q_continuous_white_noise` vs `Q_discrete_white_noise`
- [OpenCV — homography and perspective transform tutorial](https://docs.opencv.org/4.x/d9/dab/tutorial_homography.html) — `findHomography` para la proyección del plano de suelo del IPM
- [Raspberry Pi hardware documentation](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html) — RTC de la Pi 5 y conector de batería de pila de moneda

**Añadidas en la revisión v4:**

- [Hailo Model Zoo — DATA.rst](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/DATA.rst) — confirma COCO2017 como el conjunto de calibración/evaluación detrás de los artefactos entregados del model-zoo
- [Raspberry Pi Forums — Pi 5 has no hardware H.264 decoder](https://forums.raspberrypi.com/viewtopic.php?t=364180) — confirma la brecha de codec/decodificación que impulsa la recomendación del substream H.265
- [Hardware-accelerated video decoding on Raspberry Pi with FFmpeg](https://salivity.github.io/ffmpeg/article/hardware-accelerated-video-decoding-on-raspberry-pi-with-ffmpeg) — comparación de capacidad de decodificación entre las generaciones Pi 4/5
- [Comparative analysis of Wald, Wilson, and other proportion CIs](https://arxiv.org/html/2508.10223v1) — comportamiento de cobertura con n pequeña y proporciones límite
- [systemd restart/watchdog patterns for Raspberry Pi services](https://forums.raspberrypi.com/viewtopic.php?t=376126) — `Restart=`, `WatchdogSec=`, y la distinción entre caída y bloqueo
- [Keeping software running on the Raspberry Pi (dzombak.com)](https://www.dzombak.com/blog/2023/12/keep-your-software-up-and-running-on-the-raspberry-pi/) — un recorrido práctico de confiabilidad con systemd
- [ntfy.sh documentation](https://docs.ntfy.sh/) — notificaciones push autohospedables sobre una API HTTP simple
- [Pushover API](https://pushover.net/api) — un servicio de notificaciones push hospedado comúnmente usado para alertas de automatización del hogar
- [FastAPI — serving static files](https://fastapi.tiangolo.com/tutorial/static-files/) — el mecanismo mínimo para la UI del operador
- [OpenCV — image statistics and histogram basics](https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html) — los bloques constructivos detrás de la heurística de manipulación
- [pytest documentation](https://docs.pytest.org/) — el ejecutor de pruebas para la suite de pruebas unitarias de función pura
- [GitHub Actions — Python CI quickstart](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python) — conectando la suite de pytest al CI

**Añadidas en la revisión v5:**

- [Brown, Cai & DasGupta (2001) — Interval Estimation for a Binomial Proportion](https://projecteuclid.org/journals/statistical-science/volume-16/issue-2/Interval-Estimation-for-a-Binomial-Proportion/10.1214/ss/1009213286.full) — reemplaza la cita CASRAI de v4 con el artículo canónico de cobertura Wald-vs-Wilson (Statistical Science 16(2):101–133)
- [YOLOv8 nighttime small-object surveillance benchmark](https://www.iieta.org/journals/ijsse/paper/10.18280/ijsse.140611) — reemplaza las citas de poca luz de v4 con números concretos publicados de precisión/recall/mAP
- [Systematic review of low-light object detection — YOLOv8–v11 on ExDark](https://link.springer.com/article/10.1007/s42452-025-08051-5) — confirma la brecha de precisión en poca luz a través de toda la familia de modelos YOLO, no solo una versión
- [Raspberry Pi Forums — Pi 5 software decode outperforms Pi 4 hardware decode](https://forums.raspberrypi.com/viewtopic.php?t=391283) — evidencia de que el costo de decodificación de §1.10 no es automáticamente un techo duro
- [Raspberry Pi Forums — NEON-optimised software H.264 decode on Pi 5](https://forums.raspberrypi.com/viewtopic.php?t=357870) — el mismo punto, con detalle de implementación de por qué es rápido
- [Frigate GitHub discussion — Pi 5 `hwaccel drm` vs. `v4l2m2m`](https://github.com/blakeblackshear/frigate/discussions/18431) — la bandera de decodificación correcta para la Pi 5, con deltas de CPU medidos en el mundo real

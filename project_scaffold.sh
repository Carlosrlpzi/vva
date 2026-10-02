#!/usr/bin/env bash
# Fase 8 — Project Scaffold
# Ejecutar dentro de ~/projects/smart-camera-system (carpeta ya creada)

set -euo pipefail  # -e: aborta al primer error | -u: error si usas variable no definida
                    # -o pipefail: un fallo en medio de un pipe también cuenta como error

# --- Guard: evita re-inicializar un repo que ya existe ---
if [ -d ".git" ]; then
    echo "Ya existe un repositorio git en $(pwd). Nada que hacer."
    exit 0
fi

echo "Inicializando repo en $(pwd)..."
git init

echo "Creando estructura de carpetas..."
mkdir -p src/{ingest,inference,events,api} docs models tests

echo "Escribiendo .gitignore..."
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
*.db
.env
models/*.hef
EOF

echo "Escribiendo README.md..."
cat > README.md << 'EOF'
# Smart Camera System
Edge AI home security system: Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H),
PoE IP cameras (RTSP/ONVIF), event-driven person/object detection.
EOF

echo "Creando commit inicial..."
git add -A
git commit -m "Initial project scaffold"

echo "Listo. Estructura creada y commiteada."

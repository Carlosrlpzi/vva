# Guía: instalar OpenCode en Ubuntu con acceso mínimo

**Objetivo:** dejar OpenCode listo en tu laptop para construir el sistema de la Pi 5, conectado a DeepSeek, y con acceso **solo** a lo que necesita: la carpeta del proyecto y la API del modelo. Nada de tus llaves SSH, nada de tu cuenta de GitHub, nada de la Pi.

**Tiempo estimado:** 30–45 minutos.

---

## 0. La decisión de aislamiento (y por qué)

Te recomiendo **dos capas**:

1. **Un usuario dedicado de Ubuntu (`opencode-dev`) sin `sudo`.** Esta es la barrera real. La hace cumplir el kernel de Linux, no OpenCode.
2. **Permisos en `opencode.json`.** Controlan qué herramientas usa el agente y cuándo te pide aprobación.

**¿Por qué no basta con la capa 2?** Porque cualquier comando que el agente ejecute corre con los permisos del usuario que lanzó OpenCode. Si lo corres con tu usuario, un `python script.py` que el agente escribió podría leer `~/.ssh` o tus credenciales de GitHub, aunque `opencode.json` diga otra cosa. Las reglas de `bash` comparan texto de comandos, no son un sandbox: un `bash -c "..."` o un script de Python las puede rodear.

**¿Por qué no Docker?** Aísla más, pero agrega fricción que no necesitas ahorita: permisos de archivos entre host y contenedor, la terminal interactiva, y más adelante el paso de GPU. El usuario dedicado te da el 90% de la protección con el 20% del esfuerzo. Si algún día le das al agente tareas más riesgosas (instalar paquetes de terceros sin revisar, por ejemplo), ahí sí vale la pena migrar a contenedor.

**Reparto de responsabilidades resultante:**

| Quién | Puede | No puede / no debe |
|---|---|---|
| `carlosr` (repo de confianza `~/src/vva`) | Revisar diffs de forma estática, `git merge --ff-only`, `git commit`, `git push` (auditado), correr lo que requiere `root`, desplegar a la Pi | Correr pruebas o código del agente, ejecutar git dentro del taller |
| `opencode-dev` (taller `/srv/pi-vision`) | Leer y editar archivos del taller, correr pruebas, crear bundles en `/srv/vva-xfer/to-trusted` | Leer tu home, usar `sudo`, hacer push a GitHub, entrar a la Pi, tocar el auditor |
| `vva-audit` (auditoría) | Auditar un clon nuevo hecho desde un bundle, con config y llave de `root` | Editar archivos, ejecutar código, leer `/home/opencode-dev` |

Detalle de las tres zonas y del flujo de bundles: «Tres zonas (diseño A6)».

---

## 1. Prerrequisitos

```bash
sudo apt update
sudo apt install -y git curl
```

Checa que tu home no sea legible por otros usuarios (en Ubuntu 21.04+ ya viene así):

```bash
ls -ld $HOME
# Debe verse: drwxr-x---  (750). Si ves drwxr-xr-x (755), corrígelo:
chmod 750 $HOME
```

**Por qué:** si tu home es 755, el usuario `opencode-dev` podría listar y leer muchos de tus archivos. Con 750, no entra.

---

## 2. Crear el usuario dedicado

```bash
sudo adduser --disabled-password --gecos "" opencode-dev
```

- `--disabled-password`: nadie inicia sesión con contraseña; tú entras a él con `sudo -iu opencode-dev`.
- **No** lo agregues al grupo `sudo`.

Verifica:

```bash
groups opencode-dev
# Debe mostrar solo: opencode-dev : opencode-dev
```

---

## 3. Carpeta compartida del proyecto

El taller vivirá en `/srv/pi-vision`, fuera de tu home, compartido por el grupo `pivision`. `opencode-dev` es el dueño y escribe en el taller. Por regla de trabajo, `carlosr` solo lo lee y nunca ejecuta git ahí. Es una regla, no una barrera: por el `umask 002` y `core.sharedRepository group` los archivos y subcarpetas del taller son escribibles por el grupo. Eso no expone tu lado; lo que importa es que tú nunca ejecutes código ni git del taller.

```bash
sudo groupadd pivision
sudo usermod -aG pivision $USER
sudo usermod -aG pivision opencode-dev

sudo install -d -o opencode-dev -g pivision -m 2750 /srv/pi-vision

# Buzones de bundles
sudo install -d -o root -g pivision -m 0750 /srv/vva-xfer
sudo install -d -o opencode-dev -g pivision -m 2750 /srv/vva-xfer/to-trusted
sudo install -d -o carlosr -g pivision -m 2750 /srv/vva-xfer/to-taller
```

- `2750` en la raíz del taller: `opencode-dev` lee, escribe y ejecuta; el grupo `pivision` lee y ejecuta; nadie más entra. El `2` (setgid) hace que todo lo nuevo que se cree adentro herede el grupo `pivision`.

**Importante:** cierra sesión y vuelve a entrar (o reinicia) para que tu usuario quede en el grupo nuevo. Verifica con:

```bash
groups   # debe incluir pivision
```

El `umask 002` de `opencode-dev` (ver §5) hace que los archivos nuevos del taller nazcan editables por el grupo; en Ubuntu es seguro porque cada usuario tiene su propio grupo privado.

---

## 4. Clonar el repo de confianza desde GitHub (como `carlosr`)

El repo de confianza vive en `~/src/vva` y es el **único** clon con remoto de GitHub y el único lugar desde el que se hace `push`. No clones en `/srv/pi-vision`: esa ruta es el taller de `opencode-dev` (ver «Tres zonas (diseño A6)»).

Lo clonas **tú**, con tus credenciales. El agente nunca las toca.

```bash
git clone git@github.com:TU_USUARIO_GITHUB/TU_REPO.git ~/src/vva
cd ~/src/vva
git remote -v    # origin -> GitHub
```

Si clonas por HTTPS en vez de SSH, funciona igual; lo importante es que las credenciales se queden en `carlosr`.

**Activa el hook de auditoría.** Sin este paso los push salen **sin auditar y sin aviso**. Revisa antes la copia de referencia (`.githooks/pre-push`) y después instálala fuera del árbol:

```bash
install -D -m 0700 .githooks/pre-push ~/.config/vva/hooks/pre-push
git config core.hooksPath ~/.config/vva/hooks
git remote set-head origin -a          # el hook necesita origin/HEAD
git config --get core.hooksPath        # debe imprimir /home/<tu-usuario>/.config/vva/hooks
```

El hook falla cerrado hasta que el auditor esté instalado (usuario `vva-audit`, wrapper, sudoers y `/etc/vva-audit`; ver «Tres zonas (diseño A6)»): mientras tanto, ningún push sale.

---

## 5. Preparar el usuario `opencode-dev`

Entra como ese usuario:

```bash
sudo -iu opencode-dev
```

A partir de aquí, todo lo que corras es como `opencode-dev` hasta que escribas `exit`.

```bash
# Archivos nuevos editables por el grupo
echo 'umask 002' >> ~/.bashrc

# Identidad distinguible, por si algún día le permites commits locales
git config --global user.name "OpenCode (agente)"
git config --global user.email "opencode-dev@localhost"
```

Ya no hace falta `git config --global --add safe.directory /srv/pi-vision`: el taller pertenece a `opencode-dev`, así que git no ve «dubious ownership», y `carlosr` nunca ejecuta git ahí.

Nota: este usuario **no** tiene llaves SSH ni token de GitHub. Aunque intente `git push`, va a fallar. Esa es la idea.

### Crear el taller desde un bundle

`carlosr`, en `~/src/vva`:

```bash
git bundle create /srv/vva-xfer/to-taller/trusted.bundle main
```

`opencode-dev`, en `/srv/pi-vision`:

```bash
git clone -q /srv/vva-xfer/to-taller/trusted.bundle . -b main
git remote rename origin trusted
git config core.sharedRepository group
```

Esperado: `git remote -v` muestra solo `trusted` apuntando al bundle; `git config --get core.hooksPath` no imprime nada.

Después, en el taller:

```bash
python3 -m venv .venv --system-site-packages && . .venv/bin/activate && pip install -e ".[dev]"
npm --prefix .opencode ci
```

Nota: `pip` puede conservar el Pillow del sistema; si `mypy` reporta `PIL import-untyped`, instala la versión fijada en `uv.lock`.

### Migrar desde el diseño anterior

Si `/srv/pi-vision` ya existe como clon de GitHub de `carlosr` (diseño anterior, modo `2770`), no lo reutilices: `install -d` solo cambia la carpeta, no su contenido, y `git clone … .` falla si no está vacía. Así se migró la segunda PC el 2026-11-25:

1. Respaldar, como `carlosr`, todo lo que no esté publicado: `git bundle create ~/<respaldo>.bundle <ramas> refs/stash` y `git diff > ~/<respaldo>.patch`.
2. Retirar el auditor viejo: `git -C /srv/pi-vision config --unset core.hooksPath`, borrar `/usr/local/lib/vva/hooks` y la llave del auditor viejo en el home de `opencode-dev`, y reemplazar `/etc/sudoers.d/vva-audit` con la regla nueva.
3. Sacar del taller lo que no va en git: `.env` va a `~/src/vva/.env` (de `carlosr`, `0600`); los informes viejos de `.vva/audits/` se copian a `~/src/vva/.vva/audits/`.
4. Mover el taller viejo: `sudo mv /srv/pi-vision /srv/pi-vision.old`, y crear el nuevo con §3 y «Crear el taller desde un bundle».
5. Como `opencode-dev`, quitar el `safe.directory` viejo: `git config --global --unset-all safe.directory /srv/pi-vision`.
6. Borrar `/srv/pi-vision.old` solo cuando confirmes que no falta nada.

---

## 6. Instalar OpenCode (como `opencode-dev`)

Buena práctica: revisa el script antes de ejecutarlo en vez de mandarlo directo a `bash`.

```bash
curl -fsSL https://opencode.ai/install -o /tmp/opencode-install.sh
less /tmp/opencode-install.sh      # léelo; sal con q
bash /tmp/opencode-install.sh
exec bash                          # recarga el PATH
opencode --version
```

Como lo instalas con este usuario, el binario queda en su home y no requiere `sudo`.

### 6.1 Copia del auditor en una ruta de root

El binario de `~/.opencode/bin` es el que usas de forma interactiva, y `opencode-dev` puede escribirlo: se autoactualiza y, como el código generado **no** corre en sandbox, un agente de build también podría reemplazarlo. Por eso el auditor (`vva-audit-run`) **no** usa ese binario, sino una copia en `/opt/opencode/<versión>/`, de `root` y sin permiso de escritura para nadie más. Ahí no puede autoactualizarse ni ser sustituido, y el wrapper lo comprueba en cada push (sale con código 7 si la ruta o cualquier carpeta superior no es de `root` o tiene escritura de grupo u otros).

Hazlo con **tu** usuario, no como `opencode-dev`:

```bash
VER=1.19.5
sudo install -d -o root -g root -m 0755 /opt/opencode /opt/opencode/$VER
# Usa un binario que opencode-dev no haya podido tocar: descarga el release oficial de esa
# versión desde la página de releases de OpenCode en GitHub y compara su checksum con el
# publicado. No copies el de /home/opencode-dev/.opencode/bin: es justo el que no es confiable.
sudo install -o root -g root -m 0755 ./opencode /opt/opencode/$VER/opencode
sudo -u opencode-dev /opt/opencode/$VER/opencode --version   # debe imprimir 1.19.5
```

Para actualizar el auditor: revisa la versión nueva, instálala en `/opt/opencode/<nueva>/`, cambia `OPENCODE_VERSION` en `install/vva-audit-run` y reinstala el wrapper (`sudo install -o root -g root -m 0755 install/vva-audit-run /usr/local/bin/vva-audit-run`). La carpeta anterior se queda como respaldo para volver atrás.

---

## Tres zonas (diseño A6)

### Por qué

El auditor anterior corría como `opencode-dev`, el mismo usuario que los agentes auditados. Por eso un agente podía modificar el auditor antes de que revisara su propio push. A6 separa el trabajo en tres zonas con permisos distintos.

### Las tres zonas

**Taller: `/srv/pi-vision`.**

- Dueño y grupo: `opencode-dev:pivision`; modo: `2750`.
- Lo clona `opencode-dev` a partir de un bundle.
- Su único remoto se llama `trusted` y apunta a una ruta de bundle; no tiene remoto de GitHub ni `core.hooksPath`.
- Aquí corren los agentes de OpenCode.

**Repo de confianza: `~/src/vva`.**

- Dueño: `carlosr`.
- Es el único lugar desde el que se hace `push` a GitHub.
- Hook activo: `~/.config/vva/hooks/pre-push` (`0700`), configurado con `core.hooksPath`; `.githooks/pre-push` es solo la copia de referencia.

**Auditoría.**

- Usuario de sistema: `vva-audit`.
- Home: `/var/lib/vva-audit`.
- Shell: `/usr/sbin/nologin`.
- Directorio de trabajo: `/var/lib/vva-audit/work`, `vva-audit:vva-audit 0700`.

### Archivos del auditor

Los directorios son `root:root 0755`; los archivos son `root:root 0644` salvo que se indique otro modo.

- `/etc/vva-audit/opencode/opencode.json`, `/etc/vva-audit/opencode/AGENTS.md`, `/etc/vva-audit/opencode/prompts/vva-audit.md` y `/etc/vva-audit/opencode/commands/audit.md`: copias de `install/vva-audit/opencode/`.
- `/etc/vva-audit/xdg/opencode`: vacío; debe existir y ser de `root`. OpenCode intenta crearlo al arrancar; como `/etc/vva-audit/xdg` es de `root`, ese intento fallaría, así que debe crearse antes como `root`.
- `/etc/vva-audit/anthropic.env`: `root:vva-audit 0640`, exactamente una línea `ANTHROPIC_API_KEY=<value>`, una llave de Anthropic usada solo por el auditor. Nunca muestres su contenido.
- `/opt/opencode/1.19.5/opencode`: `root:root 0755`, del release oficial `anomalyco/opencode` v1.19.5, asset `opencode-linux-x64.tar.gz`.
  - **Referencia de integridad:** la única fuente válida es el checksum que GitHub publica para ese asset (el `digest` del asset en la página o en la API del release), obtenido por `carlosr` en su propia sesión. **No uses los valores de este documento como referencia** para una instalación nueva ni para una actualización: este archivo lo puede editar el lado auditado.
  - Registro, solo como constancia: el 2026-11-25 `carlosr` comprobó que el `.tar.gz` coincide con el digest del release (sha256 `588310a336fa3b9df36b3f0225b26f679d453efaad0cf3d66c3651ef3ec1647a`) y obtuvo con `sudo sha256sum /opt/opencode/1.19.5/opencode` el sha256 del binario instalado (`7df4e4136301f16589b0f30bf91a61380fca5f8ef22dd36570f5ebe1d4ec802c`).
- `/usr/local/bin/vva-audit-run`: `root:root 0755`.
- `/etc/sudoers.d/vva-audit`: `root:root 0440`; después, `sudo visudo -c`.

Instalación, como `carlosr` desde `~/src/vva` (el binario de OpenCode se instala antes, según §6.1):

```bash
sudo useradd --system --home-dir /var/lib/vva-audit --create-home --shell /usr/sbin/nologin vva-audit
sudo install -d -o vva-audit -g vva-audit -m 0700 /var/lib/vva-audit/work

sudo install -d -o root -g root -m 0755 /etc/vva-audit /etc/vva-audit/opencode \
  /etc/vva-audit/opencode/commands /etc/vva-audit/opencode/prompts \
  /etc/vva-audit/xdg /etc/vva-audit/xdg/opencode
SRC=~/src/vva/install/vva-audit/opencode
sudo install -o root -g root -m 0644 $SRC/opencode.json $SRC/AGENTS.md /etc/vva-audit/opencode/
sudo install -o root -g root -m 0644 $SRC/commands/audit.md /etc/vva-audit/opencode/commands/
sudo install -o root -g root -m 0644 $SRC/prompts/vva-audit.md /etc/vva-audit/opencode/prompts/

# La llave: créala vacía y edítala tú; nunca la pegues en un chat ni en un comando.
sudo install -o root -g vva-audit -m 0640 /dev/null /etc/vva-audit/anthropic.env
sudoedit /etc/vva-audit/anthropic.env

sudo install -o root -g root -m 0755 install/vva-audit-run /usr/local/bin/vva-audit-run
sudo install -o root -g root -m 0440 install/vva-audit.sudoers /etc/sudoers.d/vva-audit
sudo visudo -c
```

Cada vez que cambie `install/vva-audit-run`, `install/vva-audit/opencode/` o `.githooks/pre-push`, reinstala la copia correspondiente con el mismo comando.

Códigos de salida de `vva-audit-run` (copiados del encabezado de `install/vva-audit-run`):

```text
2  range is not <40-hex-sha>..<40-hex-sha>
3  API key file missing, unreadable or malformed
4  not running as vva-audit
5  opencode binary not found
6  opencode version differs from the pin
7  opencode binary path not root-owned or group/other-writable
8  /etc/vva-audit config or key file not root-owned or group/other-writable
9  bundle empty or invalid (rejected by git clone)
10 a SHA of the range is not in the bundle
11 an excluded path is present in the checkout
Any other non-zero code comes from opencode itself.
```

### Buzones

Los bundles son la única transferencia; `carlosr` nunca ejecuta git dentro del taller.

- `/srv/vva-xfer`: `root:pivision 0750`.
- `/srv/vva-xfer/to-trusted`: `opencode-dev:pivision 2750`; lo escribe el builder.
- `/srv/vva-xfer/to-taller`: `carlosr:pivision 2750`; lo escribe `carlosr`.

**Taller -> repo de confianza.** `opencode-dev` ejecuta en `/srv/pi-vision`:

```bash
git bundle create /srv/vva-xfer/to-trusted/<name>.bundle trusted/main..<branch>
```

`carlosr` ejecuta en `~/src/vva`:

```bash
git bundle verify <bundle>
git fetch <bundle> 'refs/heads/*:refs/remotes/taller/*'
git diff main...taller/<branch>
git merge --ff-only taller/<branch>
git push
```

El `git diff` es la revisión estática; las pruebas las corre `opencode-dev`, nunca `carlosr`. El `git push` lo audita el hook.

**Repo de confianza -> taller.** `carlosr` ejecuta en `~/src/vva`:

```bash
git bundle create /srv/vva-xfer/to-taller/trusted.bundle main
```

`opencode-dev` ejecuta en `/srv/pi-vision`:

```bash
git fetch /srv/vva-xfer/to-taller/trusted.bundle main:refs/remotes/trusted/main && git merge --ff-only trusted/main
```

**Por qué bundles.** Un bundle lleva solo objetos y refs (ni config ni hooks), así que `carlosr` nunca confía en el `.git/config` del taller (`safe.directory`), como exigiría un `git fetch` directo.

### Reglas del hook

- Bloquea `REQUEST_CHANGES` y también los informes sin una última línea `VERDICT` parseable.
- Un push no fast-forward a la rama por defecto siempre se bloquea.
- En otras ramas solo se audita `merge-base..local`.
- Solo se salta a propósito: `VVA_AUDIT=0` o `--no-verify`.
- Los informes van a `.vva/audits/` en `~/src/vva`, nunca a `/tmp`.
- Contexto: `docs/security-decisions.md` SD-2. El hook es un control del lado del cliente; la mitigación del lado del servidor es la protección de ramas de GitHub.

### Verificación (resultados esperados)

Comandos ejecutados por `carlosr`:

```bash
# El rango no tiene el formato <40-hex-sha>..<40-hex-sha>
sudo -n -u vva-audit /usr/local/bin/vva-audit-run bad
# Esperado: exit 2

# Rango válido con stdin vacío: pasan todas las comprobaciones previas
sudo -n -u vva-audit /usr/local/bin/vva-audit-run \
  0000000000000000000000000000000000000000..0000000000000000000000000000000000000000 </dev/null
# Esperado: exit 9 (todas las comprobaciones previas pasaron)

# Un archivo del auditor escribible por el grupo
sudo chmod g+w /etc/vva-audit/opencode/AGENTS.md
sudo -n -u vva-audit /usr/local/bin/vva-audit-run \
  0000000000000000000000000000000000000000..0000000000000000000000000000000000000000 </dev/null
# Esperado: exit 8
sudo chmod g-w /etc/vva-audit/opencode/AGENTS.md

# El auditor no puede leer el home del usuario auditado
sudo -u vva-audit ls /home/opencode-dev
# Esperado: Permission denied

# El hook está activo en el repo de confianza
git -C ~/src/vva config --get core.hooksPath
# Esperado: /home/<tu-usuario>/.config/vva/hooks
# Y el primer push real deja un informe nuevo en ~/src/vva/.vva/audits/

# Durante una auditoría (lánzala en otra terminal), con el PID de opencode.
# Se comparan solo NOMBRES con la allowlist de reset_env en install/vva-audit-run;
# `_` lo añade bash al ejecutar el binario.
pid=$(pgrep -u vva-audit -f 'opencode run' | head -1)
sudo cat /proc/$pid/environ | tr '\0' '\n' | cut -d= -f1 | sort > ~/vva-env-names.txt
printf '%s\n' ANTHROPIC_API_KEY GIT_CONFIG_GLOBAL GIT_CONFIG_NOSYSTEM HOME LANG OPENCODE_CONFIG \
  OPENCODE_CONFIG_DIR PATH TMPDIR XDG_CONFIG_HOME _ | sort | diff - ~/vva-env-names.txt
# Esperado: diff no imprime nada. Cualquier nombre de más es una fuga del entorno del llamador.
ps -u vva-audit -o args= | grep -c 'sk-ant'
# Esperado: 0 (la llave no aparece en ninguna línea de comandos)

# El directorio de trabajo queda vacío
sudo ls -A /var/lib/vva-audit/work
# Esperado: vacío después de una auditoría
```

### Actualizar el OpenCode del auditor

La versión vigente es `1.19.5` en `/opt/opencode/1.19.5/opencode`. Para actualizar, sigue §6.1: revisa el release nuevo, instálalo en `/opt/opencode/<nueva>/`, cambia `OPENCODE_VERSION` en `install/vva-audit-run` y reinstala el wrapper:

```bash
sudo install -o root -g root -m 0755 install/vva-audit-run /usr/local/bin/vva-audit-run
```

La carpeta anterior se queda como respaldo para volver atrás. No uses la copia de `~/.opencode/bin` ni la de `/home/opencode-dev/.opencode/bin`: el auditor solo usa la ruta fijada por el wrapper.

---

## 7. Conectar DeepSeek

**Recomendación:** crea en la consola de DeepSeek una **API key exclusiva para OpenCode**, distinta de la que usará la Pi 5, y ponle un límite de gasto si la consola lo permite.

**Por qué:** si algo sale mal, revocas solo esa llave sin tumbar el sistema de la Pi, y puedes ver por separado cuánto gasta el desarrollo contra la producción.

Todavía como `opencode-dev`:

```bash
cd /srv/pi-vision
opencode
```

Dentro de la interfaz:

```
/connect      → busca DeepSeek → pega la API key
/models       → elige el modelo de DeepSeek y anota su ID exacto
```

La llave se guarda en `~/.local/share/opencode/auth.json` del usuario `opencode-dev`. Protégela:

```bash
chmod 600 ~/.local/share/opencode/auth.json
```

Sal de OpenCode (Ctrl+C) para el siguiente paso.

---

## 8. Configurar permisos: `opencode.json`

Este archivo va en la raíz del repo de confianza y lo editas, commiteas y pusheas **tú** (`carlosr`), nunca el agente. Hazlo en `~/src/vva`, no en `/srv/pi-vision`:

```bash
cd ~/src/vva
nano opencode.json
```

Contenido (reemplaza `ID-DEL-MODELO` con el ID que anotaste en `/models`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "deepseek/ID-DEL-MODELO",
  "small_model": "deepseek/ID-DEL-MODELO",
  "share": "disabled",
  "permission": {
    "*": "ask",
    "read": {
      "*": "allow",
      "*.env": "deny",
      "*.env.*": "deny",
      "*.env.example": "allow"
    },
    "glob": "allow",
    "grep": "allow",
    "lsp": "allow",
    "edit": {
      "*": "ask",
      "opencode.json": "deny"
    },
    "bash": {
      "*": "ask",
      "git status*": "allow",
      "git diff*": "allow",
      "git log*": "allow",
      "ls*": "allow",
      "pytest*": "allow",
      "python -m pytest*": "allow",
      "ruff*": "allow",
      "git commit*": "deny",
      "git push*": "deny",
      "git remote*": "deny",
      "ssh*": "deny",
      "scp*": "deny",
      "rsync*": "deny",
      "sudo*": "deny",
      "curl*": "deny",
      "wget*": "deny",
      "rm -rf*": "deny"
    },
    "webfetch": "ask",
    "websearch": "ask",
    "external_directory": "deny",
    "doom_loop": "ask"
  }
}
```

### Qué hace cada decisión

- **`"*": "ask"` primero.** Todo lo que no esté listado te pide aprobación. En OpenCode gana **la última regla que coincide**, por eso el comodín va arriba y las reglas específicas abajo.
- **`small_model` en DeepSeek.** OpenCode usa un modelo pequeño para tareas internas como titular sesiones; por defecto es uno alojado por el servicio de OpenCode. Fijarlo en DeepSeek hace que tu código solo viaje a un proveedor.
- **`share: disabled`.** Evita que una sesión se publique como enlace por accidente.
- **`read` permitido, `.env` bloqueado.** El agente necesita leer el código para trabajar, pero nunca tus secretos.
- **`edit` en `ask` y `opencode.json` en `deny`.** Apruebas cada cambio al inicio, y el agente no puede aflojar sus propios permisos.
- **`bash`: lectura y pruebas permitidas; red, SSH, `sudo` y git remoto bloqueados.** El ciclo "editar → correr pruebas → corregir" fluye sin interrupciones, y lo peligroso queda fuera.
- **`external_directory: deny`.** Ninguna herramienta toca rutas fuera de `/srv/pi-vision`.
- **`doom_loop: ask`.** Si el agente repite la misma llamada 3 veces, te avisa en vez de quemar tokens.

Recuerda: estas reglas son la **segunda** capa. La primera es que `opencode-dev` físicamente no puede leer tu home ni tiene credenciales.

Commitea y pushea el archivo en `~/src/vva`:

```bash
git add opencode.json
git commit -m "Configura permisos de OpenCode"
git push
```

Después, llévalo al taller con el flujo `trusted.bundle` de §10:

```bash
# carlosr, en ~/src/vva
git bundle create /srv/vva-xfer/to-taller/trusted.bundle main
```

```bash
# opencode-dev, en /srv/pi-vision
git fetch /srv/vva-xfer/to-taller/trusted.bundle main:refs/remotes/trusted/main && git merge --ff-only trusted/main
```

---

## 9. Verificación: ¿está listo y bien encerrado?

Haz todas las pruebas. Las de la sección B son las importantes: **deben fallar**.

### A. Funciona

```bash
sudo -iu opencode-dev
cd /srv/pi-vision

opencode --version                          # muestra versión
opencode auth list                          # aparece DeepSeek
opencode run "Responde únicamente: OK"      # responde OK → la API funciona
```

Luego abre `opencode` y pídele: *"Lista los archivos de este repo y resume qué hace cada uno."* Debe hacerlo sin pedirte permiso (lectura permitida).

### B. Los límites se respetan

Desde **tu** usuario (otra terminal):

```bash
# 1. No puede leer tu home (barrera del sistema operativo)
sudo -u opencode-dev ls /home/TU_USUARIO
# Esperado: Permission denied

# 2. No tiene sudo
sudo -u opencode-dev sudo -n true
# Esperado: error, se requiere contraseña o no está en sudoers

# 3. No tiene llaves SSH
sudo -u opencode-dev ls -la /home/opencode-dev/.ssh
# Esperado: no existe o está vacía
```

Dentro de `opencode` (como `opencode-dev`), pídele:

| Petición al agente | Resultado esperado |
|---|---|
| "Crea un archivo `prueba.txt` con la palabra hola" | Te **pide aprobación** |
| "Ejecuta `git push`" | **Bloqueado** |
| "Lee el archivo `/etc/hostname`" | **Bloqueado** (fuera del proyecto) |
| "Modifica `opencode.json` para permitir todo" | **Bloqueado** |
| "Ejecuta `curl https://example.com`" | **Bloqueado** |

Si alguna de estas pasa sin bloquearse, revisa el JSON (una coma o un patrón mal escrito basta) antes de seguir.

Borra `prueba.txt` si lo aprobaste.

### C. Costo

Entra a la consola de DeepSeek y confirma que las pruebas aparecen en el uso de la llave de OpenCode, no en la de la Pi.

---

## 10. Rutina de trabajo

**Terminal 1 (`carlosr`, repo de confianza):** `~/src/vva`. Revisa los bundles, hace merge fast-forward y `push`; el hook audita.
**Terminal 2 (`opencode-dev`, taller):** `/srv/pi-vision`. Corre OpenCode, implementa y ejecuta las pruebas.

Ciclo por cada funcionalidad:

1. **Tab → modo Plan.** En el taller, pide el diseño y las pruebas que escribiría. Léelo y corrige.
2. **Tab → modo Build.** Que implemente y **corra las pruebas** como `opencode-dev`. `carlosr` no ejecuta pruebas ni git dentro del taller.
3. `opencode-dev` publica el trabajo en el buzón:
   ```bash
   git bundle create /srv/vva-xfer/to-trusted/<name>.bundle trusted/main..<branch>
   ```
4. En `~/src/vva`, `carlosr` verifica el bundle y revisa el diff de forma estática:
   ```bash
   git bundle verify /srv/vva-xfer/to-trusted/<name>.bundle
   git fetch /srv/vva-xfer/to-trusted/<name>.bundle 'refs/heads/*:refs/remotes/taller/*'
   git diff main...taller/<branch>
   ```
   Revisa también que el agente no haya tocado archivos tuyos ni del auditor. Es la comprobación determinista; el auditor es un LLM y no debe ser el único control:
   ```bash
   git diff --name-only main...taller/<branch> -- opencode.json AGENTS.md prompts .opencode \
     tests/test_skills_consistency.py install .githooks
   # Esperado: no imprime nada. Si imprime algo, revisa cada archivo antes de seguir.
   ```
   Si el cambio sí tenía que tocar uno de esos archivos (por ejemplo, un parche del auditor), lo redacta y lo commitea `carlosr` en `~/src/vva`, nunca el builder.
5. Si todo bien: `git merge --ff-only taller/<branch>` y `git push` desde `carlosr`. El hook `~/.config/vva/hooks/pre-push` audita el rango y guarda el informe en `.vva/audits/`.
6. Devuelve `main` al taller:
   ```bash
   # carlosr, en ~/src/vva
   git bundle create /srv/vva-xfer/to-taller/trusted.bundle main
   ```
   ```bash
   # opencode-dev, en /srv/pi-vision
   git fetch /srv/vva-xfer/to-taller/trusted.bundle main:refs/remotes/trusted/main && git merge --ff-only trusted/main
   ```

Si un cambio no te convence, `/undo` dentro de OpenCode lo revierte.

**Nunca uses `opencode --auto`** con esta configuración: aprueba automáticamente todo lo que esté en `ask`, que es justo la capa que te protege.

---

## Siguientes pasos (fuera de esta guía)

- Escribir el `AGENTS.md` del proyecto (puedes generar un borrador con `/init` y luego reemplazar la sección de hardware con las reglas de la Pi 5).
- Clonar el repo en la Pi 5 con una **deploy key de solo lectura** de GitHub, para que la Pi pueda hacer `git pull` pero nunca `push`.

> Los comandos de OpenCode cambian seguido. Si algo no coincide con lo que ves, compáralo con la documentación oficial en `opencode.ai/docs`, en especial las páginas de *Providers* y *Permissions*.

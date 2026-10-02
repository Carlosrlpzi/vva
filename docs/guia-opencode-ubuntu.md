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

| Quién | Puede | No puede |
|---|---|---|
| Tu usuario | Revisar diffs, correr pruebas, `git commit`, `git push`, desplegar a la Pi | — |
| `opencode-dev` | Leer y editar archivos en `/srv/pi-vision`, correr pruebas | Leer tu home, usar `sudo`, hacer push a GitHub, entrar a la Pi |

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

El repo vivirá en `/srv/pi-vision`, fuera de tu home, compartido por un grupo que incluye a tu usuario y a `opencode-dev`.

```bash
sudo groupadd pivision
sudo usermod -aG pivision $USER
sudo usermod -aG pivision opencode-dev

sudo mkdir -p /srv/pi-vision
sudo chown $USER:pivision /srv/pi-vision
sudo chmod 2770 /srv/pi-vision
```

- `2770`: tú y el grupo leen y escriben, nadie más entra. El `2` (setgid) hace que todo lo nuevo que se cree adentro herede el grupo `pivision`.

**Importante:** cierra sesión y vuelve a entrar (o reinicia) para que tu usuario quede en el grupo nuevo. Verifica con:

```bash
groups   # debe incluir pivision
```

Para que los archivos nuevos sean editables por ambos usuarios, agrega `umask 002` al final del `~/.bashrc` **de tu usuario** (en Ubuntu esto es seguro porque cada usuario tiene su propio grupo privado):

```bash
echo 'umask 002' >> ~/.bashrc
```

---

## 4. Clonar tu repo de GitHub (como tu usuario)

Lo clonas **tú**, con tus credenciales. El agente nunca las toca.

```bash
git clone git@github.com:TU_USUARIO_GITHUB/TU_REPO.git /srv/pi-vision
cd /srv/pi-vision

# Que git respete los permisos de grupo en .git
git config core.sharedRepository group

# Ajuste inicial de permisos (una sola vez)
chmod -R g+rwX /srv/pi-vision
find /srv/pi-vision -type d -exec chmod g+s {} +
```

Si clonas por HTTPS en vez de SSH, funciona igual; lo importante es que las credenciales se queden en tu usuario.

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

# El repo pertenece a tu usuario; sin esto git se queja de "dubious ownership"
git config --global --add safe.directory /srv/pi-vision

# Identidad distinguible, por si algún día le permites commits locales
git config --global user.name "OpenCode (agente)"
git config --global user.email "opencode-dev@localhost"
```

Nota: este usuario **no** tiene llaves SSH ni token de GitHub. Aunque intente `git push`, va a fallar. Esa es la idea.

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

Este archivo va en la raíz del repo y lo creas y commiteas **tú** (tu usuario, no el agente). Sal de `opencode-dev` con `exit` y, como tu usuario:

```bash
cd /srv/pi-vision
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

Commitea el archivo:

```bash
git add opencode.json
git commit -m "Configura permisos de OpenCode"
git push
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

**Terminal 1 (tu usuario):** revisas, pruebas, commiteas, despliegas.
**Terminal 2 (agente):**

```bash
sudo -iu opencode-dev
cd /srv/pi-vision
opencode
```

Ciclo por cada funcionalidad:

1. **Tab → modo Plan.** Pide el diseño y las pruebas que escribiría. Léelo y corrige.
2. **Tab → modo Build.** Que implemente.
3. En la terminal 1: `git diff`, corre las pruebas tú mismo, y revisa que `opencode.json` no haya cambiado:
   ```bash
   git diff --quiet opencode.json || echo "⚠️ opencode.json cambió, revisa antes de seguir"
   ```
4. Si todo bien: `git commit` y `git push` desde tu usuario.

Si un cambio no te convence, `/undo` dentro de OpenCode lo revierte.

**Nunca uses `opencode --auto`** con esta configuración: aprueba automáticamente todo lo que esté en `ask`, que es justo la capa que te protege.

---

## Siguientes pasos (fuera de esta guía)

- Escribir el `AGENTS.md` del proyecto (puedes generar un borrador con `/init` y luego reemplazar la sección de hardware con las reglas de la Pi 5).
- Clonar el repo en la Pi 5 con una **deploy key de solo lectura** de GitHub, para que la Pi pueda hacer `git pull` pero nunca `push`.

> Los comandos de OpenCode cambian seguido. Si algo no coincide con lo que ves, compáralo con la documentación oficial en `opencode.ai/docs`, en especial las páginas de *Providers* y *Permissions*.

# Smart Camera System — Hardware & OS Setup Guide

This guide takes you from an unboxed Raspberry Pi 5 to a fully working development machine (OS installed, network connected, AI accelerator verified, Python environment ready) — everything needed **before** you start writing the detection/tracking code. It's built around your specific stack: Raspberry Pi 5, AI HAT+ 2 (Hailo‑10H, 40 TOPS), a Ruijie RG‑ES209GC‑P PoE switch, and PoE IP cameras (Uniarch / Tiandy) speaking RTSP/ONVIF.

Each phase ends with a verification step, and Section 9 is a single troubleshooting table for every error you're likely to hit.

---

## 0. Target Architecture (recap)

```
                    [Modem / Router]  ← internet
                            │
                            │ Cat6 (uplink)
                            ▼
   [Uniarch/Tiandy PoE cameras] ──Cat6/PoE── [Ruijie RG-ES209GC-P PoE switch]
                                                        │
                                                        │ Cat6 (non-PoE uplink port)
                                                        ▼
                                            [Raspberry Pi 5] + [AI HAT+ 2 / Hailo-10H]
                                                        │
                                            Python: pull RTSP substreams → decode →
                                            Hailo inference (YOLO) → event logic
                                                        │
                                                        ▼
                                        [Your Ubuntu 24 machine] — SSH / VS Code Remote
```

### Two things worth being clear about

**1. The Pi needs only ONE network connection.** It has a single Ethernet port, and because the switch is uplinked to your modem/router, that one cable gives the Pi both local camera access (RTSP on the same subnet) and internet access (for `apt`, package installs, and remote alerts). Wi‑Fi is **not** required.

You'd only add Wi‑Fi in two cases:
- As a **fallback management path**, so you can still SSH in if the cable or switch fails.
- If you later **isolate the cameras** on a subnet with no router uplink (a good security practice, since it prevents cameras from phoning home). In that design the Pi genuinely needs both: Ethernet for cameras, Wi‑Fi for internet — and you'd need to set route metrics so each type of traffic uses the right interface.

**2. No CSI ribbon camera is involved.** Your cameras are network (IP) cameras reached over RTSP/ONVIF through the PoE switch. The AI HAT+ 2 is a co‑processor board on the GPIO header + PCIe FPC connector. The "install the camera ribbon cable" steps in most Raspberry Pi AI tutorials do **not** apply to this build — skip them.

### No monitor required

This entire build is headless. You never need an HDMI cable, monitor, keyboard, or mouse: Imager writes your SSH credentials into the image, and you do all the work from your Ubuntu machine over SSH and VS Code Remote‑SSH.

---

## 1. Parts & Tools Checklist

Hardware:
- [ ] Raspberry Pi 5 (4GB or 8GB)
- [ ] AI HAT+ 2 (Hailo‑10H) — ships with 4 threaded spacers, 4 long screws, 4 short screws, GPIO stacking header, PCIe ribbon cable, and a heatsink for the HAT itself
- [ ] Official Raspberry Pi 5 Active Cooler (strongly recommended — official docs pair it with the AI HAT+ 2)
- [ ] Official Raspberry Pi 5 27W USB‑C power supply (undervoltage is the #1 cause of random reboots/instability — see Section 9)
- [ ] microSD card, A1/A2 rated, 32GB+ (you already confirmed A1 is fine for this workload)
- [ ] Ruijie RG‑ES209GC‑P PoE switch + uplink to your router
- [ ] Ethernet cables (solid bare‑copper Cat6, as you already sourced)
- [ ] Your Ubuntu 24 desktop/laptop with an SD card reader, to flash the OS
- [ ] Phillips crosshead screwdriver
- [ ] **Not needed** — HDMI cable, monitor, keyboard, mouse. This build is fully headless via SSH.

Software (on your Ubuntu 24 machine):
- [ ] Raspberry Pi Imager — **install the official `.deb`, not the Ubuntu repo version** (see Section 3)
- [ ] VS Code + "Remote - SSH" extension ([raspberrypi.com](https://www.raspberrypi.com/news/coding-on-raspberry-pi-remotely-with-visual-studio-code/))
- [ ] OpenSSH client (`ssh`) — preinstalled on Ubuntu
- [ ] `libnss-mdns` for `.local` hostname resolution — usually already present on Ubuntu Desktop

---

## 2. Phase 1 — Physical Assembly

**Power off / unplug everything before you start.** Never seat or remove the HAT with power connected.

1. **Mount the Active Cooler on the Pi 5 first.**
   - Peel the protective film off the two thermal pads on the cooler.
   - Align the two white push‑pins with the two heatsink mounting holes on the Pi 5 (they land on the SoC, PMIC, and wireless radio).
   - Press down evenly until both pins click.
   - Plug the fan's small JST connector into the fan header next to the GPIO pins. ([raspberrypi.com](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html))
2. **Attach the 4 threaded spacers** to the Pi 5's mounting holes using the 4 long screws.
3. **Seat the GPIO stacking header** onto the Pi 5's 40‑pin GPIO connector — this is what lets the HAT sit above the Active Cooler with clearance.
4. **Connect the PCIe FPC ribbon cable**: one end to the Pi 5's PCIe FFC connector, the other to the HAT's PCIe connector. **Orientation matters** — the small "1" printed near the connector on the cable must line up with the "1" printed on the board at both ends; the ribbon's contacts face down toward the board. Getting this backwards is the most common reason the HAT is never detected later. ([community.hailo.ai](https://community.hailo.ai/t/issue-with-connecting-ai-hat-on-raspberry-pi-5-kernel-and-driver-errors/13212))
5. **Lower the AI HAT+ 2 onto the stacking header and spacers**, then secure it with the 4 short screws.
6. **Attach the HAT's own heatsink** on top of the AI HAT+ 2 (peel its adhesive film first).
7. Leave the microSD card out for now — you'll flash it separately in Phase 2.

Verification: everything should sit flat, no ribbon cable pinched, fan spins freely, no metal contact between the HAT's underside and the Active Cooler's fan housing.

Full official walkthrough with photos: [raspberrypi.com/documentation/accessories/ai-hat-plus.html](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)

---

## 3. Phase 2 — Flash Raspberry Pi OS (from Ubuntu 24)

### 3.1 Install the *current* Raspberry Pi Imager

**Do not run `sudo apt install rpi-imager`.** Ubuntu 24.04's repository ships Imager 1.8.5 ([Launchpad](https://launchpad.net/ubuntu/noble/+package/rpi-imager)), while the current release is 2.0.11 ([GitHub](https://github.com/raspberrypi/rpi-imager)). Version 1.8.5 does not support the cloud-init customisation method used by current Raspberry Pi OS images — which is precisely the headless-setup feature this guide depends on ([Raspberry Pi Forums](https://forums.raspberrypi.com/viewtopic.php?t=397018)).

Install the official `.deb` instead:

```bash
cd ~/Downloads
wget https://downloads.raspberrypi.com/imager/imager_latest_amd64.deb
sudo apt install ./imager_latest_amd64.deb
rpi-imager --version     # confirm 2.x, not 1.8.5
```

Alternative (AppImage) if you prefer not to install a package — note it needs `libfuse2` and must be launched with sudo on Ubuntu 24.04 ([raspberrytips](https://raspberrytips.com/install-raspberry-pi-imager-ubuntu/), [Raspberry Pi docs](https://www.raspberrypi.com/documentation/computers/getting-started.html)):

```bash
sudo apt install libfuse2
wget https://downloads.raspberrypi.com/imager/imager_latest_amd64.AppImage
chmod +x imager_latest_amd64.AppImage
sudo ./imager_latest_amd64.AppImage
```

### 3.2 Generate an SSH key first (recommended)

Since you're on Ubuntu, set up key-based auth now and skip passwords entirely:

```bash
ssh-keygen -t ed25519 -C "camerapi"     # press Enter to accept defaults
cat ~/.ssh/id_ed25519.pub               # copy this, you'll paste it into Imager
```

### 3.3 Flash the card

1. Insert the microSD into your card reader and launch Imager.
2. **Device**: choose "Raspberry Pi 5".
3. **Operating System**: choose "Raspberry Pi OS (64‑bit)" — you need 64‑bit for Hailo/TAPPAS and modern OpenCV wheels.
4. **Storage**: select your microSD card (double‑check you didn't select your own system disk — keep "Exclude system drives" ticked).
5. Work through the **OS Customisation** step before writing — this is what lets you boot headless straight into a working, networked Pi:
   - **Hostname**: e.g. `camerapi` (so you can reach it at `camerapi.local`)
   - **Username/password**: set your own (don't leave the default `pi`/`raspberry`)
   - **Wi‑Fi**: **skip it** — you're using wired Ethernet into the switch. Only fill this in if you specifically want a backup route into the Pi for when the cable or switch is down.
   - **Locale/timezone/keyboard**: set `America/Mexico_City` and your keyboard layout
   - **Services → Enable SSH**: choose "Allow public-key authentication only" and paste the `id_ed25519.pub` contents from step 3.2 (or pick password auth if you'd rather)
6. Save, confirm "Yes" to apply customisation, and write the image. ([raspberrypi.com](https://www.raspberrypi.com/documentation/computers/getting-started.html), [raspberry.tips](https://raspberry.tips/en/raspberrypi-tutorials/how-to-install-and-setup-raspberry-pi))

Verification: Imager reports "Write Successful". Eject the card safely.

---

## 4. Phase 3 — First Boot & Base OS Configuration

1. Insert the microSD into the Pi 5, connect Ethernet to the Ruijie switch's **non‑PoE uplink port** (see Phase 4 for the full cabling plan — the switch must already be uplinked to your modem/router so the Pi gets internet), then finally connect power. First boot takes ~60–90 seconds (it resizes the filesystem and may reboot once).
2. From your Ubuntu machine, connect over SSH:
   ```bash
   ssh <your-username>@camerapi.local
   ```
   Ubuntu resolves `.local` names natively via Avahi/mDNS, so this should work immediately. If it doesn't, confirm the resolver module is present and then find the Pi on the network:
   ```bash
   sudo apt install libnss-mdns          # if .local resolution fails
   avahi-resolve -n camerapi.local       # should print the Pi's IP
   avahi-browse -art | grep -i workstation   # lists mDNS hosts on the LAN
   ```
   Or scan the subnet directly (adjust the range to match your network):
   ```bash
   sudo apt install nmap
   nmap -sn 192.168.1.0/24
   ```
   As a last resort, check your modem/router's DHCP client list for the Pi's IP and SSH to that address directly. ([Avahi on Ubuntu](https://oneuptime.com/blog/post/2026-01-15-configure-mdns-avahi-ubuntu/view))
3. Update the OS and firmware (do this before anything else):
   ```bash
   sudo apt update && sudo apt full-upgrade -y
   sudo rpi-eeprom-update -a
   sudo reboot
   ```
4. After reboot, reconnect and open the config tool:
   ```bash
   sudo raspi-config
   ```
   - **6 Advanced Options → A8 PCIe Speed → Yes** (enables PCIe Gen 3, required for full Hailo throughput) ([github.com/giladnahor](https://github.com/giladnahor/hailo_rpi5_examples/blob/main/doc/install-raspberry-pi5.md))
   - Confirm **Localisation Options** (locale, timezone, keyboard) match what you expect
   - **Finish** and reboot when prompted.
5. (Optional, if you ever want a screen) enable VNC under **3 Interface Options → VNC**.

Verification:
```bash
uname -a                 # kernel should be 6.6.31+ (required for the Hailo driver)
cat /boot/firmware/config.txt | grep pciex1   # should show dtparam=pciex1_gen=3
```

---

## 5. Phase 4 — Network Setup (Ruijie Switch + Cameras)

**Cabling order (do this first):**

The RG‑ES209GC‑P gives you 8 PoE ports plus 1 non‑PoE uplink port — so only one port is truly non‑PoE, and you have two non‑powered devices to connect (the Pi and the router). That's fine:

1. **Raspberry Pi 5 → the non‑PoE uplink port.** The Pi runs off its own 27W USB‑C supply, and this is the port you already planned for it.
2. **Modem/router LAN port → any PoE port.** This is safe: 802.3af/at PoE switches only energise a port after the connected device negotiates for power, so a router or PC plugged into a PoE port simply gets normal Gigabit Ethernet with no voltage applied.
3. **Each camera → a PoE port.** They negotiate power and come up automatically.

Everything then sits on one subnet handed out by your router's DHCP, which is what you want for RTSP discovery.

If you'd rather keep the router on the dedicated uplink port for clarity, just swap steps 1 and 2 — put the Pi on a PoE port instead. Electrically it makes no difference.

**Then configure addresses:**

1. **Give the Pi a static/reserved IP** so it never changes (critical once you start opening RTSP connections by IP). Easiest approach: reserve the Pi's MAC address a fixed IP in your router's DHCP settings. Alternatively, set it directly on the Pi with NetworkManager:
   ```bash
   sudo nmcli connection modify "Wired connection 1" ipv4.addresses 192.168.1.50/24 ipv4.gateway 192.168.1.1 ipv4.dns "192.168.1.1" ipv4.method manual
   sudo nmcli connection up "Wired connection 1"
   ```
2. **Log into the Ruijie switch's cloud/local management UI** and confirm the Pi's uplink port and each camera's PoE port show link-up and the expected power draw.
3. **Assign static IPs to each camera** (via the camera's own web UI or ONVIF device tool) on the same subnet, e.g. `192.168.1.101`, `.102`, `.103` — do this now so your Python code never has to re-discover cameras via DHCP.
4. **Connect VS Code to the Pi** (do this once and you'll code from your normal desktop for the rest of the project):
   - VS Code → Extensions → install "Remote - SSH" (Microsoft)
   - `Ctrl+Shift+P` → "Remote-SSH: Connect to Host…" → `<username>@camerapi.local` (or the static IP)
   - Enter your password once; VS Code will install its server component on the Pi automatically. ([raspberrypi.com](https://www.raspberrypi.com/news/coding-on-raspberry-pi-remotely-with-visual-studio-code/))

Verification:
```bash
ping -c3 192.168.1.101       # each camera responds
ip addr show                 # Pi shows the static IP you assigned
```

---

## 6. Phase 5 — Install the Hailo AI HAT+ 2 Software Stack

This is the step people get wrong most often — **AI HAT+ 2 uses a different package name than the original AI HAT+/AI Kit.** Do not follow generic "AI HAT+" tutorials that say `sudo apt install hailo-all` — that installs the wrong runtime for your board and the two packages cannot coexist. ([raspberrypi.com](https://www.raspberrypi.com/documentation/computers/ai.html))

```bash
sudo apt update
sudo apt install -y dkms
sudo apt install -y hailo-h10-all      # correct package for AI HAT+ 2 (Hailo-10H)
sudo reboot
```

`hailo-h10-all` pulls in: the kernel driver (via DKMS), HailoRT runtime + Python bindings, TAPPAS post‑processing/GStreamer plugins, and `rpicam-apps` Hailo demo stages. ([raspberrypi.com](https://www.raspberrypi.com/documentation/computers/ai.html), [nexty-ele.com](https://www.nexty-ele.com/en/technical-column/raspberry-pi-5/))

After the reboot, verify the accelerator is visible at three levels:

```bash
lspci | grep -i hailo                 # 1. PCIe bus sees the chip
# expect: 0001:01:00.0 Co-processor: Hailo Technologies Ltd. ...

hailortcli scan                       # 2. HailoRT driver sees the device
# expect: Hailo Devices: [-] Device: 0001:01:00.0

hailortcli fw-control identify        # 3. Firmware responds with board/chip info
```

If all three succeed, your AI accelerator is fully operational and ready for models. If any step fails, go straight to the troubleshooting table (Section 9) — don't reinstall blindly.

---

## 7. Phase 6 — Camera & Vision Tooling (RTSP/ONVIF/OpenCV)

Install the OS-level media libraries first, then the Python side.

```bash
sudo apt install -y ffmpeg libgstreamer1.0-0 gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-libav \
    v4l-utils
```

Quick sanity check that a camera's RTSP stream is reachable at all, before writing any Python:
```bash
ffprobe rtsp://<user>:<password>@192.168.1.101:554/Streaming/Channels/101
```
If this returns stream info (codec, resolution, fps), the network + camera side is confirmed working end‑to‑end — any later Python issues are code‑side, not network‑side.

---

## 8. Phase 7 — Python Development Environment

Raspberry Pi OS (Bookworm+) blocks system‑wide `pip install` by default (PEP 668) — the same `externally-managed-environment` behaviour you've already hit on Ubuntu. Always work inside a virtual environment.

```bash
sudo apt install -y python3-full python3-venv python3-pip git

mkdir -p ~/projects/smart-camera-system && cd ~/projects/smart-camera-system
python3 -m venv --system-site-packages .venv     # --system-site-packages so Hailo's Python bindings (installed system-wide by hailo-h10-all) are visible inside the venv
source .venv/bin/activate
```

`--system-site-packages` matters here specifically because `hailo-h10-all` installs `hailort`'s Python bindings at the system level, not into arbitrary venvs — without this flag `import hailo_platform` will fail inside your venv even though the SDK is installed. ([forums.raspberrypi.com](https://forums.raspberrypi.com/viewtopic.php?t=389516))

Install the core Python packages for the project:
```bash
pip install --upgrade pip
pip install opencv-python-headless numpy onvif-zeep fastapi uvicorn[standard] \
    python-dotenv pydantic APScheduler
```

- `onvif-zeep` — discover/query cameras and fetch RTSP stream URIs over ONVIF ([github.com/FalkTannhaeuser](https://github.com/FalkTannhaeuser/python-onvif-zeep)):
  ```python
  from onvif import ONVIFCamera
  cam = ONVIFCamera('192.168.1.101', 80, 'admin', 'yourpassword')
  media = cam.create_media_service()
  profiles = media.GetProfiles()
  uri = media.GetStreamUri({
      'StreamSetup': {'Stream': 'RTP-Unicast', 'Transport': {'Protocol': 'RTSP'}},
      'ProfileToken': profiles[0].token
  }).Uri
  print(uri)
  ```
- `opencv-python-headless` (not the GUI `opencv-python` build) is correct here since the Pi runs headless — `cv2.VideoCapture(rtsp_uri)` pulls frames for both display-free processing and feeding into the Hailo inference pipeline.
- `fastapi` + `uvicorn` — matches your existing stack for exposing an API/dashboard for events.

Verification:
```bash
python3 -c "import cv2, onvif, hailo_platform; print('all imports OK')"
```

---

## 9. Phase 8 — Project Scaffold (ready to start coding)

Set up Git now so the coding phase starts clean:

```bash
cd ~/projects/smart-camera-system
git init
mkdir -p src/{ingest,inference,events,api} docs models tests
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
*.db
.env
models/*.hef
EOF

cat > README.md << 'EOF'
# Smart Camera System
Edge AI home security system: Raspberry Pi 5 + AI HAT+ 2 (Hailo-10H),
PoE IP cameras (RTSP/ONVIF), event-driven person/object detection.
EOF

git add -A && git commit -m "Initial project scaffold"
```

Suggested module layout:
- `src/ingest/` — RTSP/ONVIF connection handling, substream pulling
- `src/inference/` — Hailo model loading + detection/tracking logic
- `src/events/` — zone rules, event logging (SQLite), alerting
- `src/api/` — FastAPI endpoints for live status/event history
- `models/` — compiled `.hef` Hailo model files (git‑ignored — too large for git)

At this point everything is installed and verified — hardware, OS, network, AI accelerator, camera connectivity, and Python environment. You're ready to start writing the actual ingestion/inference pipeline.

---

## 10. Troubleshooting — Errors and Fixes

| Phase | Symptom | Likely Cause | Fix |
|---|---|---|---|
| Assembly | Pi won't boot after HAT install | Ribbon cable reversed/half-seated | Power off, reseat the PCIe FPC cable so the "1" markings align at both ends, contacts facing the board ([community.hailo.ai](https://community.hailo.ai/t/issue-with-connecting-ai-hat-on-raspberry-pi-5-kernel-and-driver-errors/13212)) |
| Assembly | "Low voltage" lightning bolt icon / random reboots under load | Underpowered or resistive USB‑C cable/PSU | Use the official 27W Pi 5 PSU with a short, thick USB‑C cable; check with `vcgencmd pmic_read_adc EXT5V_V` (should read ~5.1V); avoid mains extension leads/hubs in the power path ([forums.raspberrypi.com](https://forums.raspberrypi.com/viewtopic.php?t=382143)) |
| OS Flash | Imager fails to write / verify | Bad SD card, USB reader issue | Try a different USB port/reader, re-download the OS image cache in Imager, use a different microSD |
| OS Flash | No "OS Customisation" step, or headless settings ignored on first boot | Ubuntu's repo Imager (1.8.5) is too old for cloud-init customisation | Uninstall it and install the official 2.x `.deb` from `downloads.raspberrypi.com`; verify with `rpi-imager --version` ([Raspberry Pi Forums](https://forums.raspberrypi.com/viewtopic.php?t=397018)) |
| OS Flash | AppImage won't launch on Ubuntu 24 | Missing FUSE 2 / needs root | `sudo apt install libfuse2`, then run it with `sudo ./imager_latest_amd64.AppImage` ([raspberrytips](https://raspberrytips.com/install-raspberry-pi-imager-ubuntu/)) |
| First Boot | Can't SSH via `camerapi.local` from Ubuntu | `libnss-mdns` missing or Avahi not running | `sudo apt install libnss-mdns avahi-utils`, confirm `systemctl status avahi-daemon`, test with `avahi-resolve -n camerapi.local`; fall back to `nmap -sn 192.168.1.0/24` or the router's DHCP table ([Avahi on Ubuntu](https://oneuptime.com/blog/post/2026-01-15-configure-mdns-avahi-ubuntu/view)) |
| First Boot | `ssh` rejects the key you set in Imager | Pasted the private key, or wrong key file | Paste the contents of `~/.ssh/id_ed25519.pub` (the `.pub` file); test with `ssh -i ~/.ssh/id_ed25519 -v <user>@camerapi.local` to see which key is offered |
| First Boot | SSH "Permission denied" | Wrong username or SSH not enabled at flash time | Re-check the username set in Imager's customisation step; if SSH wasn't enabled, re-flash or add an empty `ssh` file to the boot partition manually |
| Hailo Install | `lspci \| grep hailo` shows nothing | PCIe not enabled, cable issue, or old firmware | Confirm `dtparam=pciex1_gen=3` is in `/boot/firmware/config.txt`, reseat the ribbon cable, run `sudo rpi-eeprom-update -a` then reboot ([spotpear.com](https://spotpear.com/index.php/wiki/Raspberry-Pi-5-AI-HAT-Plus-Hailo-8L-13-26-Tops-PCIe-M.2.html)) |
| Hailo Install | `lspci` sees the chip but `hailortcli scan` returns nothing | Kernel driver not loaded / DKMS build failed | `sudo dkms status` to check the module; rebuild with `sudo dkms build -m hailo1x_pci -v <version> -k $(uname -r)` then `sudo dkms install -m hailo1x_pci -v <version> -k $(uname -r)`, reboot ([forums.raspberrypi.com](https://forums.raspberrypi.com/viewtopic.php?t=395601)) |
| Hailo Install | `hailortcli fw-control identify` → "FW is not loaded to the device" | Firmware failed to boot on the NPU after a hot-unplug or partial init | Force a PCIe reset: `echo 1 \| sudo tee /sys/bus/pci/devices/0001:01:00.0/reset`, or full remove/rescan: `echo 1 \| sudo tee .../remove` then `echo 1 \| sudo tee /sys/bus/pci/rescan`, then retry ([community.hailo.ai](https://community.hailo.ai/t/ai-hat-on-raspberry-pi-failure-fw-is-not-loaded-to-the-device/18354)) |
| Hailo Install | Installed `hailo-all` by mistake instead of `hailo-h10-all` | Followed a generic AI HAT+ (not +2) guide | `sudo apt remove hailo-all && sudo apt install hailo-h10-all` then reboot — the two runtimes cannot coexist ([raspberrypi.com](https://www.raspberrypi.com/documentation/computers/ai.html)) |
| Python | `error: externally-managed-environment` on `pip install` | Debian/Raspberry Pi OS blocks system-wide pip (PEP 668) | Always work inside a venv: `python3 -m venv .venv && source .venv/bin/activate` before any `pip install` ([forums.raspberrypi.com](https://forums.raspberrypi.com/viewtopic.php?t=389516)) |
| Python | `import hailo_platform` fails inside the venv even though `hailortcli` works | Venv isolated from system-wide Hailo Python bindings | Recreate the venv with `python3 -m venv --system-site-packages .venv` |
| Network | Camera RTSP stream times out from Python but `ffprobe` works from the Pi's shell | Wrong credentials/URL path in code, or camera-side max-connections limit reached | Confirm the exact RTSP path via ONVIF `GetStreamUri()` rather than guessing the URL; close unused `cv2.VideoCapture` handles between test runs |
| Network | Cameras get a different IP after a power cycle | No DHCP reservation set | Set a DHCP reservation by MAC in your router, or a fixed static IP directly on the camera |
| General | `uname -a` shows kernel < 6.6.31 | OS not fully updated before Hailo install | `sudo apt full-upgrade -y && sudo reboot`, then re-check before installing `hailo-h10-all` ([github.com/giladnahor](https://github.com/giladnahor/hailo_rpi5_examples/blob/main/doc/install-raspberry-pi5.md)) |

---

## Readiness Checklist Before Coding

- [ ] `lspci | grep -i hailo` shows the Hailo co-processor
- [ ] `hailortcli scan` and `hailortcli fw-control identify` both succeed
- [ ] Pi has a fixed IP and is reachable over SSH and in VS Code Remote‑SSH
- [ ] Each camera has a fixed IP and responds to `ffprobe rtsp://...`
- [ ] Python venv activates cleanly and `import cv2, onvif, hailo_platform` succeeds
- [ ] Git repo initialized with the module scaffold committed

Once every box is checked, you're ready to move on to writing the actual ingestion → detection → event pipeline.


root@svva:/etc/wireguard# cat server_private.key
OFn57s/CT3uBJWTf2AvvXWIKt+Dfwve1DlRZuYaztWo=

root@svva:/etc/wireguard# cat laptop_public.key
SgNfaHun+UsXiFqddQXi5sDR2Jl+OvYmLavnMHn7sjU=

root@svva:/etc/wireguard# cat phone_public.key
MvmbZ58jpBvv1bFn3j08ypwcuPT7BolmMpRWHpz3qwg=

[Interface]
PrivateKey = <contenido de laptop_private.key>
Address = 10.13.13.2/24

[Peer]
PublicKey = prE8LPUuDmBZ5I6X8LxXyoVBSyv9/rAPVJyfiL2eaQA=
Endpoint = <tu-ip-publica-o-dominio-DDNS>:51820
AllowedIPs = 192.168.1.0/24, 10.13.13.0/24
PersistentKeepalive = 25
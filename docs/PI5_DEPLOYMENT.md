# Marine AI on Raspberry Pi 5 — Deployment & Optimization Guide

Target hardware: **BCM2712 quad-core Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X, 5V/5A USB-C,
active cooling, USB microscope, no internet in the field.**

---

## 0. What actually makes this fast

Roughly in order of impact, measured per 640×640 detection:

| Change | Typical effect |
|---|---|
| PyTorch `.pt` → **NCNN** | 2 000–5 000 ms → 250–450 ms |
| NCNN → **INT8 TFLite** | a further 20–40 %, at some accuracy cost |
| `imgsz` 640 → **416** | ~2.3× fewer pixels through the backbone (~40–55 % faster) |
| **CPU governor = performance** | 20–30 % on bursty single-image requests |
| **Thread caps** (no oversubscription) | 10–25 % under concurrent load, and far less jitter |
| **jemalloc** | 3–8 % latency, and flat RSS over a long session |
| **Threaded camera reader** | removes 60–150 ms of blocking per capture |

The first two require the compile step below. Everything else is configuration.

> Numbers are the expected range for a YOLOv8n/s-class model on a Pi 5 with active
> cooling. Measure your own with `python scripts/pi_healthcheck.py --bench` — the
> exact figures depend on model size and how crowded the frames are.

---

## 1. Compile the model (on a dev machine, **not** the Pi)

NCNN export shells out to a downloaded `pnnx` binary, and the INT8 TFLite path
runs `onnx2tf` on top of full TensorFlow. Both want internet and a lot of RAM, and
TensorFlow's aarch64 converter is slow and prone to OOM on an 8 GB Pi. Compile on
an x86_64 Linux box (or WSL) and copy `models/` across.

```bash
pip install -r requirements.txt -r requirements-export.txt

# Export all three backends at the resolution you intend to run.
python optimize_model.py export --formats ncnn onnx tflite --imgsz 416
```

Produces, and records in `models/exports.json`:

```
models/best_ncnn_model/               # .param + .bin  -> NCNN
models/best.onnx                      # static-shape ONNX
models/best_saved_model/best_int8.tflite
```

**The `--imgsz` you export at is frozen into the artifact.** `app.py` reads it back
from the manifest and overrides `YOLO_IMGSZ` accordingly, printing a note when the
two disagree. To change resolution, re-export.

### Verify there is no accuracy drift

```bash
python optimize_model.py verify --iou-tol 0.85 --conf-tol 0.1 --report drift.json
```

This runs every image in `test_img/` through PyTorch (the reference), then through
each compiled backend, and greedily matches detections:

| Tolerance | Default | Meaning |
|---|---|---|
| `--match-iou` | 0.50 | below this, two boxes are not the same object at all |
| `--iou-tol` | 0.85 | a matched pair below this is reported as **loose_box** |
| `--conf-tol` | 0.10 | a matched pair above this Δconfidence is **conf_drift** |
| `--max-drift-rate` | 0.02 | fraction of reference detections allowed to deviate before the backend FAILs |

Reported categories: `missed` (reference detection absent), `extra` (false positive),
`class_mismatch`, `loose_box`, `conf_drift`. Exit code is non-zero on failure, so
this drops straight into CI.

Expect NCNN to match PyTorch almost exactly (fp32/fp16, same operators). Expect INT8
TFLite to show real confidence drift — that is the cost of 8-bit quantization, and
`--conf-tol` is the dial for how much you will accept.

**Everything runs in a subprocess.** pnnx, the TF converter and onnxruntime all
segfault occasionally on ARM; a crashed worker is reported as one failed step
(`worker crashed with signal 11`) instead of killing the pipeline.

---

## 2. Build the offline bundle (on an online machine)

```bash
# aarch64 Linux host, or x86_64 cross-download — both work
./create_offline_bundle.sh --archive
```

Downloads every wheel for `requirements-pi.txt` into `offline_wheels/` using
`--platform manylinux2014_aarch64 --only-binary=:all:`, writes `SHA256SUMS`, and
optionally tars the whole deployable tree.

`--only-binary=:all:` is not optional. Without it pip will happily fetch an sdist,
and building torch from source on a Pi is a multi-hour job that usually OOMs.

Set `PY_VERSION` to match the Pi's interpreter (Bookworm 64-bit ships **3.11**):

```bash
PY_VERSION=3.11 ./create_offline_bundle.sh
```

Copy across:

```bash
rsync -av --exclude .venv ./ pi@raspberrypi.local:~/Marine_AI/
```

---

## 3. Install on the Pi (no internet needed)

```bash
cd ~/Marine_AI
./install_offline.sh
```

Verifies checksums, creates `.venv`, installs with `--no-index
--find-links=./offline_wheels`, then imports every package and reports which
inference backends are available.

A handful of things are apt packages, not wheels — install them **before** the Pi
goes offline:

```bash
sudo apt update
sudo apt install -y libjemalloc2 libjemalloc-dev v4l-utils zram-tools cpufrequtils
```

---

## 4. OS tuning

```bash
sudo ./scripts/setup_pi_os.sh          # apply
sudo ./scripts/setup_pi_os.sh --status # inspect
./scripts/setup_pi_os.sh --dry-run     # preview, no root needed
sudo reboot
```

### 4.1 ZRAM

Writes `/etc/default/zramswap` with `ALGO=zstd`, `PERCENT=50`, disables the
SD-card swapfile, and sets `vm.swappiness=100`, `vm.page-cluster=0`.

Why on a machine with 8 GB: the app needs ~1.5–2.5 GB, but a long field session
gradually fills RAM with cold anonymous pages while the page cache is what keeps
model loads and image I/O fast. ZRAM compresses those ~2.8× in RAM instead of
evicting cache or writing to the SD card — an SD swap on a Pi is slow enough to
look like a hang and burns write cycles. `page-cluster=0` matters because ZRAM
pages are individually compressed, so reading ahead 8 pages at a time is wasted work.

### 4.2 CPU governor

The Pi 5 boots with `ondemand`, which ramps 1.5 → 2.4 GHz only after observing
sustained load. A single 300 ms inference finishes before the governor reacts, so
every request effectively runs at the low clock. The script sets
`GOVERNOR="performance"` in `/etc/default/cpufrequtils`, applies it live, and
installs `cpu-performance.service` to reapply at boot (cpufrequtils alone sometimes
runs before all cores are online).

This raises idle power and heat — hence the active-cooling requirement.

### 4.3 USB power

UVC microscopes are the classic autosuspend victim: the kernel powers the device
down between captures and the next grab either stalls for hundreds of ms or returns
`-EIO`. The script sets `options usbcore autosuspend=-1`, applies it live to every
device, and adds a udev rule so devices plugged in later (a microscope swapped
mid-cruise) don't silently revert. The same rule creates a stable `/dev/microscope`
symlink.

It also ensures `usb_max_current_enable=1` in `/boot/firmware/config.txt`, which is
what unlocks the Pi 5's full 1.6 A USB budget when a 5 A supply is attached.

---

## 5. Run it

### Foreground

```bash
./run_optimized.sh
./run_optimized.sh --bench    # print the resolved environment and exit
```

Sets thread caps, `LD_PRELOAD`s jemalloc (auto-discovering the `.so`), configures
the camera and web server, then `exec`s `app.py`.

### As a boot service

```bash
sudo cp marine_ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now marine_ai
journalctl -u marine_ai -f
```

Edit `User=`, `WorkingDirectory=` and the `LD_PRELOAD` path if the app isn't at
`/home/pi/Marine_AI`. The service user must be in the `video` group:

```bash
sudo usermod -aG video pi
```

The unit binds `0.0.0.0:7860`, so any laptop or tablet on the Pi's Ethernet link or
Wi-Fi hotspot reaches the UI at `http://<pi-ip>:7860` with no desktop environment
running on the Pi.

> **This is an unauthenticated web app.** On any network you don't fully control,
> uncomment `Environment=GRADIO_AUTH=marine:CHANGE_ME` in the unit (or export
> `GRADIO_AUTH="user:password"`). `app.py` prints a warning at startup when it binds
> to all interfaces with no auth.

`MemoryMax=6G` deliberately sits below the 8 GB total: a runaway allocation gets the
service restarted rather than triggering the kernel OOM killer, which on a Pi
frequently picks `sshd` and locks you out of the box.

---

## 6. Configuration reference

| Variable | Default | Notes |
|---|---|---|
| `CPU_THREADS` | `min(4, ncpu)` | drives every thread pool; try `3` to leave a core for camera I/O |
| `MODEL_BACKEND` | `auto` | `auto` \| `ncnn` \| `tflite` \| `onnx` \| `pytorch` |
| `MODEL_PATH` | — | explicit artifact path; overrides `MODEL_BACKEND` |
| `YOLO_IMGSZ` | `640` (`416` in the service) | ignored for compiled models — the export value wins |
| `MAX_DIMENSION` | `2048` (`1280` in the service) | pre-inference downscale cap |
| `CONFIDENCE_THRESHOLD` | `0.3` | |
| `ENABLE_CAMERA` | `true` | hides the Live Microscope tab when false |
| `CAMERA_DEVICE` | `0` | index or `/dev/video0` / `/dev/microscope` |
| `CAMERA_WIDTH/HEIGHT/FPS` | `1920/1080/30` | what is requested from V4L2 |
| `CAMERA_FOURCC` | `MJPG` | `MJPG` is what makes 1080p30 fit over USB |
| `CAMERA_MAX_WIDTH/HEIGHT` | `1280/720` | capture-thread downscale target |
| `GRADIO_SERVER_NAME` | `127.0.0.1` (`0.0.0.0` in the service) | |
| `GRADIO_SERVER_PORT` | auto (`7860` in the service) | |
| `GRADIO_AUTH` | — | `user:pass[,user2:pass2]` |
| `LD_PRELOAD` | — | set by `run_optimized.sh` to libjemalloc |

---

## 7. The camera pipeline

`camera.py` runs `cv2.VideoCapture` on its own thread with a one-deep double buffer.

Three things this fixes:

1. **Blocking.** A synchronous `cap.read()` on the Gradio request thread blocks for
   exposure + USB transfer + MJPG decode — commonly 60–150 ms at 1080p, much worse
   on a USB 2.0 port. The web worker never waits on the camera now.
2. **Staleness.** V4L2 hands back whatever is next in the driver ring, so a request
   after an idle period gets a frame from seconds ago. `CAP_PROP_BUFFERSIZE=1` plus a
   continuously-draining reader means the buffer always holds the newest frame.
3. **Decode cost.** Frames are downscaled to 1280×720 with `INTER_AREA` **in the
   capture thread**, before anything else touches them. `INTER_AREA` is both the
   cheapest and the highest-quality option when shrinking — it's a box filter over
   source pixels, so it won't alias thin structures (setae, antennae) out of existence
   the way `INTER_LINEAR` does at large downscale factors.

`CAP_V4L2` is requested explicitly: it's what makes the FOURCC/resolution/buffersize
properties stick and what reaches the hardware-assisted MJPG path instead of a
userspace fallback. FOURCC is set *before* the frame size, because switching pixel
format resets the negotiated resolution on most UVC devices.

The reader reconnects with capped exponential backoff, so an unplugged microscope
recovers on its own instead of needing a service restart.

Live preview auto-refresh is **off by default** — each preview frame costs a colour
conversion plus a PNG encode for the browser, which competes directly with inference
on 4 cores. Use "Capture & Analyze" for the actual work.

---

## 8. Troubleshooting

```bash
python scripts/pi_healthcheck.py           # thermals, throttling, threads, camera, models
python scripts/pi_healthcheck.py --bench   # timed sweep over test_img/
```

| Symptom | Check |
|---|---|
| Inference much slower than expected | `--status`: governor should be `performance`; `vcgencmd get_throttled` should be `0x0` |
| `get_throttled` bit 0 set | under-voltage — the 5V/**5A** supply and the cable are both load-bearing |
| RSS creeps up over hours | jemalloc isn't preloaded; healthcheck reports this explicitly. `LD_PRELOAD` only applies at exec — start via `run_optimized.sh` or the unit |
| "Could not open camera" | `ls /dev/video*`, `v4l2-ctl --list-devices`, and confirm the service user is in `video` |
| Camera stalls after idle | USB autosuspend — `./scripts/setup_pi_os.sh --status` |
| `ncnn` import error at startup | `pip install ncnn` from the wheelhouse. With `YOLO_AUTOINSTALL=False`, ultralytics won't (and shouldn't) fetch it at runtime |
| Detections differ from the desktop app | you're on a compiled backend; run `optimize_model.py verify` to quantify it |
| Boxes look right but labels say `class3` | export lost its metadata; `app.py` restores names from `exports.json`, so re-export or check the manifest |

### Log inspection

```bash
journalctl -u marine_ai -n 200 --no-pager
journalctl -u marine_ai -f
systemctl status marine_ai
```

`app.py` prints the resolved backend, imgsz and thread count on startup — that line
is the fastest way to confirm the service is running what you think it is.

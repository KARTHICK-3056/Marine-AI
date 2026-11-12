#!/usr/bin/env bash
#
# Install Marine AI on a Raspberry Pi 5 with NO internet access.
#
# Prerequisite: offline_wheels/ built by create_offline_bundle.sh on an online host.
#
#   ./install_offline.sh                    # venv + runtime deps + verification
#   ./install_offline.sh --system           # install into the system interpreter
#   ./install_offline.sh --with-export-tools
#
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

WHEEL_DIR="${WHEEL_DIR:-$APP_DIR/offline_wheels}"
VENV_DIR="${VENV_DIR:-$APP_DIR/.venv}"
USE_VENV=1
WITH_EXPORT_TOOLS=0

for arg in "$@"; do
  case "$arg" in
    --system) USE_VENV=0 ;;
    --with-export-tools) WITH_EXPORT_TOOLS=1 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "=============================================================="
echo " Marine AI offline install"
echo "   arch    : $(uname -m)"
echo "   python  : $(python3 -V 2>&1)"
echo "   wheels  : $WHEEL_DIR"
echo "=============================================================="

if [[ ! -d "$WHEEL_DIR" ]]; then
  echo "ERROR: $WHEEL_DIR not found. Build it on an online machine first:" >&2
  echo "       ./create_offline_bundle.sh" >&2
  exit 1
fi

if [[ -f "$WHEEL_DIR/SHA256SUMS" ]]; then
  echo "--- verifying wheel checksums ---"
  ( cd "$WHEEL_DIR" && sha256sum --quiet -c SHA256SUMS ) \
    && echo "  all wheels intact" \
    || { echo "ERROR: checksum mismatch - the bundle was corrupted in transit." >&2; exit 1; }
fi

# ----------------------------------------------------------------------
# System packages (apt, from the local cache / preinstalled base image)
# ----------------------------------------------------------------------
# These are libraries, not Python packages, so they cannot live in the wheelhouse.
# Raspberry Pi OS ships most of them; jemalloc and v4l-utils usually need adding
# while the Pi still has a network, before it goes to the field.
MISSING_APT=()
for pkg in libjemalloc2 v4l-utils libgl1 libglib2.0-0; do
  dpkg -s "$pkg" >/dev/null 2>&1 || MISSING_APT+=("$pkg")
done
if [[ ${#MISSING_APT[@]} -gt 0 ]]; then
  echo
  echo "NOTE: missing system packages: ${MISSING_APT[*]}"
  echo "      While the Pi still has internet, run:"
  echo "        sudo apt update && sudo apt install -y ${MISSING_APT[*]}"
  echo "      (jemalloc is a performance optimization; v4l-utils is for microscope"
  echo "       diagnostics; both are optional but recommended.)"
fi

# ----------------------------------------------------------------------
# Python environment
# ----------------------------------------------------------------------
if [[ "$USE_VENV" -eq 1 ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    echo
    echo "--- creating venv at $VENV_DIR ---"
    # --system-site-packages lets the venv see apt-installed python3-libcamera /
    # picamera2 if you later add a CSI camera; harmless otherwise.
    python3 -m venv --system-site-packages "$VENV_DIR"
  fi
  PY="$VENV_DIR/bin/python"
else
  PY="$(command -v python3)"
fi
echo "Using interpreter: $PY"

# Every pip call below is --no-index: pip must never attempt a network round-trip,
# because on a field Pi that is a 15-second DNS timeout per package, not an error.
PIP_ARGS=(--no-index --find-links="$WHEEL_DIR" --no-cache-dir)

echo
echo "--- upgrading pip toolchain from the local wheelhouse ---"
"$PY" -m pip install "${PIP_ARGS[@]}" --upgrade pip setuptools wheel 2>&1 | tail -3 || \
  echo "  (skipping: no pip/setuptools wheels bundled -- not fatal)"

REQ_FILE="$APP_DIR/requirements-pi.txt"
[[ -f "$REQ_FILE" ]] || REQ_FILE="$WHEEL_DIR/requirements-pi.txt"

echo
echo "--- installing runtime dependencies ---"
"$PY" -m pip install "${PIP_ARGS[@]}" -r "$REQ_FILE"

if [[ "$WITH_EXPORT_TOOLS" -eq 1 ]]; then
  echo
  echo "--- installing export toolchain ---"
  "$PY" -m pip install "${PIP_ARGS[@]}" -r "$APP_DIR/requirements-export.txt" || \
    echo "WARNING: export toolchain incomplete; export models on an x86_64 host instead." >&2
fi

# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
echo
echo "--- verifying the install ---"
"$PY" - <<'PYCHECK'
import importlib, platform, sys

print(f"  python      : {sys.version.split()[0]} ({platform.machine()})")
required = ["numpy", "cv2", "torch", "ultralytics", "gradio", "PIL"]
optional = {
    "ncnn": "NCNN backend (fastest on Cortex-A76)",
    "tflite_runtime": "INT8 TFLite backend",
    "onnxruntime": "ONNX Runtime backend",
}

failed = []
for name in required:
    try:
        mod = importlib.import_module(name)
        print(f"  {name:<12}: {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        failed.append(name)
        print(f"  {name:<12}: MISSING ({e})")

print("  --- optional inference backends ---")
for name, why in optional.items():
    try:
        importlib.import_module(name)
        print(f"  {name:<16}: available  - {why}")
    except Exception:
        print(f"  {name:<16}: absent     - {why}")

if failed:
    print(f"\nERROR: missing required packages: {', '.join(failed)}")
    sys.exit(1)

import torch, cv2
torch.set_num_threads(4)
cv2.setNumThreads(4)
print("\n  thread caps applied OK")
PYCHECK

echo
echo "--- checking model artifacts ---"
if [[ -f "$APP_DIR/models/exports.json" ]]; then
  "$PY" - <<'PYMODELS'
import json, pathlib
root = pathlib.Path.cwd()  # install_offline.sh cd's to APP_DIR before this runs
manifest = json.loads((root / "models" / "exports.json").read_text())
for backend, record in manifest.get("exports", {}).items():
    path = root / record["path"]
    mark = "OK " if path.exists() else "MISSING"
    print(f"  {mark} {backend:<8} imgsz={record.get('imgsz')} {record['path']}")
PYMODELS
elif [[ -f "$APP_DIR/models/best.pt" ]]; then
  echo "  only models/best.pt present (PyTorch backend, ~2-5 s/image)."
  echo "  Compile a fast backend on an x86_64 host:  python optimize_model.py export --imgsz 416"
else
  echo "  WARNING: no model found in models/ -- copy best.pt across before running."
fi

chmod +x "$APP_DIR/run_optimized.sh" 2>/dev/null || true

cat <<EOF

==============================================================
 Install complete.

 Start it:
   ./run_optimized.sh

 Install as a boot service:
   sudo cp marine_ai.service /etc/systemd/system/
   sudo systemctl daemon-reload && sudo systemctl enable --now marine_ai

 OS tuning (ZRAM, CPU governor, USB power):
   sudo ./scripts/setup_pi_os.sh
==============================================================
EOF

#!/usr/bin/env bash
#
# Marine AI - optimized launcher for Raspberry Pi 5 (BCM2712 / Cortex-A76 / 8 GB).
#
# Everything here has to happen *before* the Python process starts: LD_PRELOAD is
# only read at exec time, and OpenMP/OpenBLAS/MKL size their thread pools once
# during library load, long before any Python code could change them.
#
#   ./run_optimized.sh                 # start the app
#   ./run_optimized.sh --bench         # print the environment and exit
#   CPU_THREADS=3 ./run_optimized.sh   # leave a core for the OS / camera I/O
#
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

# ----------------------------------------------------------------------
# 1. Thread pool caps
# ----------------------------------------------------------------------
# The Pi 5 has 4 physical cores and no SMT. torch, OpenCV, NCNN, OpenBLAS and
# NumExpr each default to one thread per core, so unconstrained they create
# ~4x more runnable threads than there are cores. On a memory-bandwidth-limited
# SoC that is strictly worse than 1:1: the threads evict each other's L2 lines
# and the scheduler burns time migrating them. Pin every pool to the same budget.
CPU_THREADS="${CPU_THREADS:-$(nproc --all)}"
export CPU_THREADS
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export VECLIB_MAXIMUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

# Keep OpenMP workers on their cores instead of letting the kernel shuffle them
# between clusters, and stop idle workers from spinning (which on a passively
# monitored Pi shows up as 100% CPU while doing nothing and heats the SoC).
export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
export OMP_PLACES="${OMP_PLACES:-cores}"
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-PASSIVE}"
export KMP_BLOCKTIME="${KMP_BLOCKTIME:-0}"

# ----------------------------------------------------------------------
# 2. jemalloc
# ----------------------------------------------------------------------
# glibc's allocator keeps a per-thread arena cache that fragments badly under
# the allocate/free churn of tensor workloads, and its arena locks are a real
# contention point on multi-threaded ARM. Steady-state RSS with glibc typically
# creeps upward over a long field session; jemalloc's size-class binning holds
# it flat and shaves a few percent off inference latency.
#   sudo apt install -y libjemalloc2 libjemalloc-dev
JEMALLOC_LIB="${JEMALLOC_LIB:-}"
if [[ -z "$JEMALLOC_LIB" ]]; then
  for candidate in \
    /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 \
    /usr/lib/aarch64-linux-gnu/libjemalloc.so \
    /usr/lib/arm-linux-gnueabihf/libjemalloc.so.2 \
    /usr/lib/libjemalloc.so.2; do
    if [[ -e "$candidate" ]]; then JEMALLOC_LIB="$candidate"; break; fi
  done
fi

if [[ -n "$JEMALLOC_LIB" && -e "$JEMALLOC_LIB" ]]; then
  export LD_PRELOAD="${JEMALLOC_LIB}${LD_PRELOAD:+:$LD_PRELOAD}"
  # background_thread: return freed pages to the OS from a helper thread rather
  #   than on the allocation path, so a purge never lands inside an inference call.
  # dirty_decay_ms/muzzy_decay_ms: 10s is a compromise between returning RAM and
  #   re-faulting pages every frame; the default (10s/0) is fine, we just make it explicit.
  export MALLOC_CONF="${MALLOC_CONF:-background_thread:true,metadata_thp:auto,dirty_decay_ms:10000,muzzy_decay_ms:10000}"
  echo "[run_optimized] jemalloc: $JEMALLOC_LIB"
else
  echo "[run_optimized] WARNING: jemalloc not found; using glibc malloc." >&2
  echo "[run_optimized]          install it with: sudo apt install -y libjemalloc2" >&2
fi

# ----------------------------------------------------------------------
# 3. Model / inference configuration
# ----------------------------------------------------------------------
# MODEL_BACKEND=auto picks the fastest compiled artifact recorded in
# models/exports.json (ncnn > tflite > onnx > pytorch). Pin it explicitly if you
# want to A/B backends. Build the artifacts with:
#   python optimize_model.py export --imgsz 416
export MODEL_BACKEND="${MODEL_BACKEND:-auto}"

# 416 roughly halves the forward-pass cost versus 640. If a compiled model is in
# use, the resolution frozen at export time wins over this value -- app.py says so
# on startup. Must be a multiple of 32.
export YOLO_IMGSZ="${YOLO_IMGSZ:-416}"
export MAX_DIMENSION="${MAX_DIMENSION:-1280}"
export CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.3}"

# No outbound network calls in the field: no analytics ping, no silent pip installs.
export YOLO_AUTOINSTALL=False
export GRADIO_ANALYTICS_ENABLED=False
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export YOLO_OFFLINE="${YOLO_OFFLINE:-1}"
# Keep ultralytics' settings/cache inside the app dir so a systemd user with no
# writable $HOME does not fail on first run.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-$APP_DIR/.ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

# ----------------------------------------------------------------------
# 4. USB microscope
# ----------------------------------------------------------------------
export ENABLE_CAMERA="${ENABLE_CAMERA:-true}"
export CAMERA_DEVICE="${CAMERA_DEVICE:-0}"
export CAMERA_WIDTH="${CAMERA_WIDTH:-1920}"
export CAMERA_HEIGHT="${CAMERA_HEIGHT:-1080}"
export CAMERA_FPS="${CAMERA_FPS:-30}"
export CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"
export CAMERA_MAX_WIDTH="${CAMERA_MAX_WIDTH:-1280}"
export CAMERA_MAX_HEIGHT="${CAMERA_MAX_HEIGHT:-720}"
# OpenCV's V4L2 backend logs a warning per unsupported property probe; silence it.
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-ERROR}"
export OPENCV_VIDEOIO_PRIORITY_V4L2="${OPENCV_VIDEOIO_PRIORITY_V4L2:-1000}"

# ----------------------------------------------------------------------
# 5. Web server
# ----------------------------------------------------------------------
# 0.0.0.0 so a laptop or tablet on the Pi's Ethernet link / Wi-Fi hotspot can
# reach the UI with no desktop environment running on the Pi itself.
# GRADIO_AUTH is strongly recommended whenever this is not a private link.
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export GRADIO_SHARE="${GRADIO_SHARE:-false}"
export PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------
# 6. Go
# ----------------------------------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$APP_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$APP_DIR/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3)"
  fi
fi

echo "[run_optimized] app     : $APP_DIR"
echo "[run_optimized] python  : $PYTHON_BIN"
echo "[run_optimized] threads : $CPU_THREADS"
echo "[run_optimized] backend : $MODEL_BACKEND (imgsz $YOLO_IMGSZ)"
echo "[run_optimized] listen  : http://$GRADIO_SERVER_NAME:$GRADIO_SERVER_PORT"

if [[ "${1:-}" == "--bench" ]]; then
  echo "[run_optimized] --bench: environment only, not starting the app."
  env | grep -E '^(CPU_THREADS|OMP_|MKL_|OPENBLAS_|VECLIB_|NUMEXPR_|KMP_|LD_PRELOAD|MALLOC_CONF|YOLO_|MODEL_|CAMERA_|GRADIO_|MAX_DIMENSION|CONFIDENCE_)' | sort
  exit 0
fi

# exec so the Python process inherits PID 1 of this unit: systemd's stop/restart
# signals reach the app directly instead of killing a shell that leaves it orphaned.
exec "$PYTHON_BIN" "$APP_DIR/app.py" "$@"

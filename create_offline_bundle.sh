#!/usr/bin/env bash
#
# Build a self-contained wheelhouse for an offline Raspberry Pi 5.
#
# RUN THIS ON AN ONLINE MACHINE. Two supported hosts:
#   * aarch64 Linux (another Pi, an ARM VM, an AWS Graviton box) - native, most reliable
#   * x86_64 Linux/macOS/WSL - cross-downloads via --platform, works because we
#     restrict everything to prebuilt wheels
#
#   ./create_offline_bundle.sh                       # runtime deps only
#   ./create_offline_bundle.sh --with-export-tools   # + the NCNN/TFLite export toolchain
#   ./create_offline_bundle.sh --archive             # also produce marine_ai_offline.tar.gz
#
# Copy the resulting offline_wheels/ (or the tarball) to the Pi and run install_offline.sh.
#
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

WHEEL_DIR="${WHEEL_DIR:-$APP_DIR/offline_wheels}"
# Raspberry Pi OS Bookworm (64-bit) ships Python 3.11. Change this if your Pi
# image differs -- `python3 -V` on the Pi is the authority. Wheels are tagged per
# minor version, so a mismatch here yields a bundle pip will silently reject.
PY_VERSION="${PY_VERSION:-3.11}"
# manylinux2014 (glibc 2.17) is the widest-compatibility aarch64 tag and is what
# torch/opencv/numpy publish. Bookworm's glibc 2.36 accepts it.
PLATFORMS="${PLATFORMS:-manylinux2014_aarch64 manylinux_2_17_aarch64 linux_aarch64}"

WITH_EXPORT_TOOLS=0
MAKE_ARCHIVE=0
for arg in "$@"; do
  case "$arg" in
    --with-export-tools) WITH_EXPORT_TOOLS=1 ;;
    --archive) MAKE_ARCHIVE=1 ;;
    -h|--help) sed -n '2,20p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

echo "=============================================================="
echo " Marine AI offline wheelhouse"
echo "   host      : $(uname -s) $(uname -m)"
echo "   target    : aarch64 / cpython $PY_VERSION"
echo "   output    : $WHEEL_DIR"
echo "=============================================================="

mkdir -p "$WHEEL_DIR"
python3 -m pip install --upgrade pip >/dev/null

download_set() {
  local requirements="$1" label="$2"
  echo
  echo "--- downloading: $label ($requirements) ---"
  local ok=0
  for platform in $PLATFORMS; do
    echo "  trying --platform $platform"
    # --only-binary=:all: is mandatory when cross-downloading: a source dist
    # would otherwise be fetched and then need a compiler on the Pi (torch from
    # source on a Pi 5 is a multi-hour job that usually OOMs).
    if pip download \
        -r "$requirements" \
        -d "$WHEEL_DIR" \
        --platform "$platform" \
        --python-version "$PY_VERSION" \
        --implementation cp \
        --abi "cp${PY_VERSION//./}" \
        --only-binary=:all: 2>&1 | tail -5; then
      ok=1
      break
    fi
    echo "  ...no complete resolution for $platform, trying the next tag"
  done
  if [[ "$ok" -eq 0 ]]; then
    echo "ERROR: could not resolve '$label' for any aarch64 platform tag." >&2
    echo "       Check PY_VERSION ($PY_VERSION) against the Pi's python3 -V, then retry." >&2
    return 1
  fi
}

# pip itself, so the Pi can bootstrap even with an ancient pip in the venv.
echo
echo "--- downloading: bootstrap (pip/setuptools/wheel) ---"
pip download pip setuptools wheel -d "$WHEEL_DIR" --only-binary=:all: 2>&1 | tail -3

download_set "$APP_DIR/requirements-pi.txt" "Pi runtime"

if [[ "$WITH_EXPORT_TOOLS" -eq 1 ]]; then
  # These are only needed if you intend to run `optimize_model.py export` ON the
  # Pi. That is not the recommended flow (TensorFlow/onnx2tf on aarch64 is slow
  # and crash-prone), but the option exists for a fully self-contained field kit.
  download_set "$APP_DIR/requirements-export.txt" "export toolchain" || \
    echo "WARNING: export toolchain incomplete; export on an x86_64 box instead." >&2
fi

# A pip-installable copy of the app's own requirements travels with the bundle so
# install_offline.sh never has to guess what to install.
cp "$APP_DIR/requirements-pi.txt" "$WHEEL_DIR/"
[[ "$WITH_EXPORT_TOOLS" -eq 1 ]] && cp "$APP_DIR/requirements-export.txt" "$WHEEL_DIR/" || true

WHEEL_COUNT=$(find "$WHEEL_DIR" -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')
BUNDLE_SIZE=$(du -sh "$WHEEL_DIR" | cut -f1)

# Checksums: a wheelhouse usually travels on a USB stick to a boat. This is how
# you find out a file got truncated before you are offline with it.
( cd "$WHEEL_DIR" && sha256sum ./*.whl > SHA256SUMS 2>/dev/null || true )

echo
echo "=============================================================="
echo " Done: $WHEEL_COUNT wheels, $BUNDLE_SIZE in $WHEEL_DIR"
echo "=============================================================="

if [[ "$MAKE_ARCHIVE" -eq 1 ]]; then
  ARCHIVE="$APP_DIR/marine_ai_offline_$(date +%Y%m%d).tar.gz"
  echo "Creating $ARCHIVE ..."
  tar -czf "$ARCHIVE" -C "$APP_DIR" \
    offline_wheels requirements-pi.txt install_offline.sh run_optimized.sh \
    app.py camera.py optimize_model.py marine_ai.service scripts docs models 2>/dev/null || \
  tar -czf "$ARCHIVE" -C "$APP_DIR" offline_wheels requirements-pi.txt install_offline.sh
  echo "Archive: $ARCHIVE ($(du -sh "$ARCHIVE" | cut -f1))"
fi

cat <<EOF

Next steps
  1. Copy offline_wheels/ (plus this repo) to the Pi, e.g.
       rsync -av "$WHEEL_DIR" pi@raspberrypi.local:~/Marine_AI/
  2. On the Pi:
       cd ~/Marine_AI && ./install_offline.sh
EOF

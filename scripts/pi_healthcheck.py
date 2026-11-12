#!/usr/bin/env python3
"""Field diagnostics for a Marine AI Raspberry Pi 5 deployment.

Answers the three questions that matter when detection is slower than expected:
is the SoC throttling, are the thread caps and allocator actually in effect, and
is the microscope enumerating?

    python scripts/pi_healthcheck.py
    python scripts/pi_healthcheck.py --bench      # timed inference on test_img/
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OK, WARN, BAD = "  [ok]   ", "  [warn] ", "  [BAD]  "


def _read(path: str) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _cmd(*args: str) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=5, check=False).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def check_platform() -> None:
    section("Platform")
    model = _read("/proc/device-tree/model") or "unknown"
    print(f"{OK}model     : {model.rstrip(chr(0))}")
    print(f"{OK}arch      : {platform.machine()}  kernel {platform.release()}")
    print(f"{OK}python    : {sys.version.split()[0]}")
    if platform.machine() not in ("aarch64", "arm64"):
        print(f"{WARN}not running on aarch64 -- these checks assume a Pi 5")


def check_thermal_and_power() -> None:
    section("Thermal / power")
    temp_raw = _read("/sys/class/thermal/thermal_zone0/temp")
    if temp_raw:
        temp = int(temp_raw) / 1000
        # The Pi 5 soft-limits at 80 C and hard-limits at 85 C. With active
        # cooling a sustained inference load should sit in the 50s-60s.
        tag = OK if temp < 70 else (WARN if temp < 80 else BAD)
        print(f"{tag}SoC temp  : {temp:.1f} C" + ("  (throttling range)" if temp >= 80 else ""))

    throttled = _cmd("vcgencmd", "get_throttled")
    if throttled and "=" in throttled:
        value = int(throttled.split("=")[1], 16)
        flags = {
            0: "under-voltage NOW (check the 5V/5A supply and cable)",
            1: "ARM frequency capped NOW",
            2: "currently throttled",
            3: "soft temperature limit active",
            16: "under-voltage occurred since boot",
            17: "frequency capping occurred since boot",
            18: "throttling occurred since boot",
            19: "soft temp limit occurred since boot",
        }
        active = [msg for bit, msg in flags.items() if value & (1 << bit)]
        if not active:
            print(f"{OK}throttling: none (0x0)")
        else:
            for msg in active:
                print(f"{BAD if 'NOW' in msg or 'currently' in msg else WARN}{msg}")

    gov = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    if gov:
        tag = OK if gov == "performance" else WARN
        note = "" if gov == "performance" else "  (run: sudo ./scripts/setup_pi_os.sh)"
        print(f"{tag}governor  : {gov}{note}")
    freq = _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    if freq:
        print(f"{OK}CPU clock : {int(freq) // 1000} MHz")


def check_memory() -> None:
    section("Memory")
    try:
        import psutil

        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()
        print(f"{OK}RAM       : {vm.used / 1e9:.2f} / {vm.total / 1e9:.2f} GB used ({vm.percent:.0f}%)")
        print(f"{OK}swap      : {sw.used / 1e9:.2f} / {sw.total / 1e9:.2f} GB used")
    except ImportError:
        print(f"{WARN}psutil not installed; skipping memory detail")

    swaps = _read("/proc/swaps") or ""
    if "zram" in swaps:
        print(f"{OK}zram      : active")
    else:
        print(f"{WARN}zram      : not active (run: sudo ./scripts/setup_pi_os.sh)")
    print(f"{OK}swappiness: {_read('/proc/sys/vm/swappiness')}")


def check_allocator_and_threads() -> None:
    section("Allocator / threads")
    preload = os.environ.get("LD_PRELOAD", "")
    if "jemalloc" in preload:
        print(f"{OK}jemalloc  : preloaded ({preload})")
    else:
        found = any(
            Path(p).exists()
            for p in (
                "/usr/lib/aarch64-linux-gnu/libjemalloc.so.2",
                "/usr/lib/aarch64-linux-gnu/libjemalloc.so",
            )
        )
        # A missing LD_PRELOAD here is the single most common reason a service
        # looks "tuned" but still fragments: the variable only takes effect if
        # it was set before exec.
        print(
            f"{WARN}jemalloc  : NOT preloaded"
            f"{' (installed but LD_PRELOAD unset -- start via run_optimized.sh)' if found else ' (not installed: sudo apt install libjemalloc2)'}"
        )
    if "MALLOC_CONF" in os.environ:
        print(f"{OK}MALLOC_CONF: {os.environ['MALLOC_CONF']}")

    expected = os.environ.get("CPU_THREADS", str(min(4, os.cpu_count() or 4)))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        value = os.environ.get(var)
        tag = OK if value == expected else WARN
        print(f"{tag}{var:<24}: {value or 'unset'}")

    try:
        import cv2
        import torch

        print(f"{OK}torch threads : {torch.get_num_threads()}")
        print(f"{OK}cv2 threads   : {cv2.getNumThreads()}")
    except ImportError as e:
        print(f"{BAD}torch/cv2 import failed: {e}")


def check_camera() -> None:
    section("USB microscope")
    devices = sorted(Path("/dev").glob("video*"))
    if not devices:
        print(f"{WARN}no /dev/video* devices found")
    else:
        print(f"{OK}devices   : {', '.join(d.name for d in devices)}")
    listing = _cmd("v4l2-ctl", "--list-devices")
    if listing:
        for line in listing.splitlines():
            print(f"        {line}")
    else:
        print(f"{WARN}v4l2-ctl unavailable (sudo apt install v4l-utils)")

    for f in sorted(Path("/sys/bus/usb/devices").glob("*/power/control")):
        try:
            if f.read_text().strip() == "auto":
                print(
                    f"{WARN}{f.parent.parent.name}: USB autosuspend enabled "
                    "(can stall frame grabs -- run scripts/setup_pi_os.sh)"
                )
        except OSError:
            pass


def check_model() -> None:
    section("Model artifacts")
    import json

    manifest_path = ROOT / "models" / "exports.json"
    if not manifest_path.exists():
        pt = ROOT / "models" / "best.pt"
        print(
            f"{WARN}no exports.json -- running the slow PyTorch path"
            if pt.exists()
            else f"{BAD}no model found in models/"
        )
        print("       compile one with: python optimize_model.py export --imgsz 416")
        return
    manifest = json.loads(manifest_path.read_text()).get("exports", {})
    for backend, record in manifest.items():
        path = ROOT / record["path"]
        tag = OK if path.exists() else BAD
        print(f"{tag}{backend:<8}: imgsz={record.get('imgsz')} size={record.get('size_mb')} MB  {record['path']}")


def bench() -> None:
    section("Inference benchmark")
    import statistics
    import time

    import cv2

    import app  # applies the same thread caps and backend selection as production

    if app.model is None:
        print(f"{BAD}model failed to load; nothing to benchmark")
        return

    images = sorted(p for p in (ROOT / "test_img").iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[:20]
    if not images:
        print(f"{WARN}no images in test_img/")
        return

    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in images]
    for _ in range(2):  # warm-up: first call pays lazy init, not steady-state cost
        app.analyze_rgb(frames[0])

    times = []
    for frame in frames:
        t0 = time.perf_counter()
        app.analyze_rgb(frame)
        times.append((time.perf_counter() - t0) * 1000)

    print(f"{OK}backend   : {app.MODEL_BACKEND} @ imgsz {app.ACTIVE_IMGSZ}, {app.CPU_THREADS} threads")
    print(
        f"{OK}per image : mean {statistics.fmean(times):.0f} ms | "
        f"median {statistics.median(times):.0f} ms | min {min(times):.0f} | max {max(times):.0f}"
    )
    print(f"{OK}throughput: {1000 / statistics.fmean(times):.2f} images/s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench", action="store_true", help="run a timed inference sweep over test_img/")
    args = parser.parse_args()

    print("=" * 62)
    print(" Marine AI - Raspberry Pi 5 health check")
    print("=" * 62)
    check_platform()
    check_thermal_and_power()
    check_memory()
    check_allocator_and_threads()
    check_camera()
    check_model()
    if args.bench:
        bench()
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

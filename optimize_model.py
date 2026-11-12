#!/usr/bin/env python3
"""Model compilation, quantization and accuracy-drift verification for Marine AI.

PyTorch CPU inference on a `.pt` checkpoint is the slowest way to run YOLOv8 on a
Raspberry Pi 5 (~2-5 s/image). This script compiles `models/best.pt` into ARM-friendly
runtimes and then proves the compiled models still agree with the original:

    # 1. Compile (run on an x86_64 Linux dev box -- see docs/PI5_DEPLOYMENT.md)
    python optimize_model.py export --formats ncnn onnx tflite --imgsz 416

    # 2. Verify no accuracy drift (runs fine on the Pi itself)
    python optimize_model.py verify --iou-tol 0.85 --conf-tol 0.1

    # 3. Both
    python optimize_model.py all --imgsz 416

Design notes
------------
* Every export and every inference sweep runs in its own **subprocess**. The native
  toolchains involved (pnnx for NCNN, onnx2tf/TensorFlow for TFLite, onnxruntime)
  are the usual source of hard crashes on aarch64. A SIGSEGV in a worker is reported
  as a failed step instead of taking the whole pipeline down with it.
* Exports are recorded in `models/exports.json`. `app.py` reads that manifest to pick
  a backend and, critically, to learn the `imgsz` each artifact was frozen at -- NCNN
  and INT8 TFLite bake the input resolution in at export time.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Thread caps must be in the environment before torch/OpenCV/OpenMP are imported,
# because each reads them once at load time to size its pool. Same budget as app.py.
_THREADS = str(int(os.environ.get("CPU_THREADS", min(4, os.cpu_count() or 4))))
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, _THREADS)
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"
DEFAULT_TEST_DIR = ROOT / "test_img"
MANIFEST_PATH = MODELS_DIR / "exports.json"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

SUPPORTED_FORMATS = ("ncnn", "onnx", "tflite")


# ======================================================================
# Small helpers
# ======================================================================
def _log(msg: str) -> None:
    print(msg, flush=True)


def _rule(title: str) -> None:
    _log("\n" + "=" * 72)
    _log(title)
    _log("=" * 72)


def list_images(directory: Path, limit: int | None = None) -> list[Path]:
    files = sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        try:
            return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            _log(f"WARNING: could not read {MANIFEST_PATH.name} ({e}); starting a fresh manifest.")
    return {"exports": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _run_worker(args: list[str], timeout: int) -> tuple[bool, str]:
    """Run this script again in worker mode, isolating native crashes.

    Returns (ok, message). A negative return code means the child died on a signal
    (SIGSEGV/SIGABRT/SIGKILL) -- the exact failure mode we are containing here.
    """
    cmd = [sys.executable, str(Path(__file__).resolve()), *args]
    try:
        proc = subprocess.run(cmd, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    if proc.returncode == 0:
        return True, "ok"
    if proc.returncode < 0:
        return False, (
            f"worker crashed with signal {-proc.returncode} (native segfault/abort). "
            "The runtime for this format is likely unusable on this architecture."
        )
    if os.name == "nt" and proc.returncode > 0x80000000:
        return False, f"worker crashed (Windows exception 0x{proc.returncode:08X})"
    return False, f"worker exited with code {proc.returncode}"


# ======================================================================
# Calibration dataset (INT8 TFLite)
# ======================================================================
def build_calibration_dataset(image_dir: Path, class_names: dict[int, str], out_dir: Path) -> Path:
    """Stage `image_dir` as a minimal YOLO detection dataset and return its YAML path.

    Ultralytics' INT8 path drives a real `YOLODataset` over `data['val']`, so it needs
    an images/labels tree and a YAML with absolute paths. Labels are written empty:
    calibration only samples the *input* distribution, never the targets.
    """
    images_dir = out_dir / "images" / "val"
    labels_dir = out_dir / "labels" / "val"
    for d in (images_dir, labels_dir):
        d.mkdir(parents=True, exist_ok=True)

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No calibration images found in {image_dir}")

    for src in images:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        (labels_dir / f"{src.stem}.txt").touch()

    names_block = "\n".join(f"  {i}: {name}" for i, name in sorted(class_names.items()))
    yaml_path = out_dir / "calibration.yaml"
    yaml_path.write_text(
        f"path: {out_dir.as_posix()}\ntrain: images/val\nval: images/val\nnames:\n{names_block}\n",
        encoding="utf-8",
    )
    _log(f"  Calibration set: {len(images)} images -> {yaml_path}")
    return yaml_path


# ======================================================================
# Export worker (runs in a child process)
# ======================================================================
def _worker_export(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    weights = Path(args.weights)
    model = YOLO(str(weights))
    class_names = dict(model.names)

    kwargs: dict = {"format": args.format, "imgsz": args.imgsz, "batch": 1}
    calib_dir = None

    if args.format == "onnx":
        kwargs.update(opset=args.opset, simplify=True, dynamic=False, half=False)
    elif args.format == "ncnn":
        # NCNN runs fp32/fp16 on NEON; there is no calibration step here.
        kwargs.update(half=args.ncnn_half)
    elif args.format == "tflite":
        kwargs.update(int8=True, half=False)
        calib_dir = MODELS_DIR / ".calibration"
        kwargs["data"] = str(build_calibration_dataset(Path(args.test_dir), class_names, calib_dir))
        kwargs["fraction"] = args.fraction

    _log(f"  ultralytics export kwargs: {kwargs}")
    started = time.perf_counter()
    produced = model.export(**kwargs)
    elapsed = time.perf_counter() - started

    produced_path = Path(str(produced)).resolve()
    if not produced_path.exists():
        _log(f"ERROR: export reported {produced_path} but it does not exist.")
        return 1

    record = {
        "format": args.format,
        "path": os.path.relpath(produced_path, ROOT).replace(os.sep, "/"),
        "imgsz": args.imgsz,
        "int8": args.format == "tflite",
        "source_weights": os.path.relpath(weights.resolve(), ROOT).replace(os.sep, "/"),
        "names": {str(k): v for k, v in class_names.items()},
        "export_seconds": round(elapsed, 2),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "ultralytics": _pkg_version("ultralytics"),
        "size_mb": round(_tree_size(produced_path) / 1e6, 2),
    }
    manifest = load_manifest()
    manifest.setdefault("exports", {})[args.format] = record
    save_manifest(manifest)

    _log(f"  -> {record['path']} ({record['size_mb']} MB) in {elapsed:.1f}s")

    if calib_dir and not args.keep_calibration and calib_dir.exists():
        shutil.rmtree(calib_dir, ignore_errors=True)
    return 0


def _pkg_version(name: str) -> str:
    try:
        from importlib import metadata

        return metadata.version(name)
    except (metadata.PackageNotFoundError, ValueError):  # pragma: no cover - informational only
        return "unknown"


def _tree_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


# ======================================================================
# Prediction worker (runs in a child process, one per backend)
# ======================================================================
def _worker_predict(args: argparse.Namespace) -> int:
    import numpy as np
    from ultralytics import YOLO

    try:
        import torch

        torch.set_num_threads(int(_THREADS))
    except ImportError:  # pragma: no cover - torch is an ultralytics hard dep
        pass

    import cv2

    cv2.setNumThreads(int(_THREADS))

    model = YOLO(args.model, task="detect")
    images = list_images(Path(args.test_dir), args.max_images)
    if not images:
        _log(f"ERROR: no images in {args.test_dir}")
        return 1

    # Warm-up: first call pays for lazy graph/weight loading and allocator growth,
    # which would otherwise be charged to image #1 and skew the latency numbers.
    warm = cv2.imread(str(images[0]))
    for _ in range(max(0, args.warmup)):
        model.predict(source=warm, imgsz=args.imgsz, conf=args.conf, verbose=False, save=False)

    per_image: dict[str, list[dict]] = {}
    latencies: list[float] = []

    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            _log(f"  skipping unreadable image: {path.name}")
            continue
        t0 = time.perf_counter()
        results = model.predict(source=frame, imgsz=args.imgsz, conf=args.conf, verbose=False, save=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        dets = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cls_id = int(box.cls.item())
            dets.append(
                {
                    "cls": cls_id,
                    "name": str(model.names.get(cls_id, cls_id)),
                    "conf": round(float(box.conf.item()), 6),
                    "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                }
            )
        dets.sort(key=lambda d: d["conf"], reverse=True)
        per_image[path.name] = dets

    payload = {
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "threads": int(_THREADS),
        "images": per_image,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "median": round(statistics.median(latencies), 1) if latencies else None,
            "min": round(min(latencies), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
            "count": len(latencies),
        },
        "numpy": np.__version__,
    }
    Path(args.out).write_text(json.dumps(payload), encoding="utf-8")
    return 0


# ======================================================================
# Drift comparison
# ======================================================================
def iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compare_runs(reference: dict, candidate: dict, *, match_iou: float, iou_tol: float, conf_tol: float) -> dict:
    """Greedy-match detections per image and classify every disagreement.

    `match_iou` decides which boxes are "the same object" at all; `iou_tol` and
    `conf_tol` then decide whether that pair is close enough to pass.
    """
    ref_images = reference["images"]
    cand_images = candidate["images"]

    issues: list[dict] = []
    ious: list[float] = []
    conf_deltas: list[float] = []
    counters = {"matched": 0, "loose_box": 0, "conf_drift": 0, "class_mismatch": 0, "missed": 0, "extra": 0}
    ref_total = 0

    for name, ref_dets in ref_images.items():
        cand_dets = cand_images.get(name, [])
        ref_total += len(ref_dets)
        used: set[int] = set()

        for ref_det in ref_dets:
            best_idx, best_iou = -1, 0.0
            for i, cand_det in enumerate(cand_dets):
                if i in used:
                    continue
                overlap = iou(ref_det["xyxy"], cand_det["xyxy"])
                if overlap > best_iou:
                    best_idx, best_iou = i, overlap

            if best_idx < 0 or best_iou < match_iou:
                counters["missed"] += 1
                issues.append({"image": name, "kind": "missed", "ref": ref_det, "best_iou": round(best_iou, 3)})
                continue

            cand_det = cand_dets[best_idx]
            used.add(best_idx)
            ious.append(best_iou)
            dconf = abs(ref_det["conf"] - cand_det["conf"])
            conf_deltas.append(dconf)

            if cand_det["cls"] != ref_det["cls"]:
                counters["class_mismatch"] += 1
                issues.append({"image": name, "kind": "class_mismatch", "ref": ref_det, "cand": cand_det})
                continue

            failed = False
            if best_iou < iou_tol:
                counters["loose_box"] += 1
                issues.append(
                    {"image": name, "kind": "loose_box", "ref": ref_det, "cand": cand_det, "iou": round(best_iou, 3)}
                )
                failed = True
            if dconf > conf_tol:
                counters["conf_drift"] += 1
                issues.append(
                    {"image": name, "kind": "conf_drift", "ref": ref_det, "cand": cand_det, "delta": round(dconf, 3)}
                )
                failed = True
            if not failed:
                counters["matched"] += 1

        for i, cand_det in enumerate(cand_dets):
            if i not in used:
                counters["extra"] += 1
                issues.append({"image": name, "kind": "extra", "cand": cand_det})

    cand_total = sum(len(v) for v in cand_images.values())
    return {
        "counters": counters,
        "ref_detections": ref_total,
        "cand_detections": cand_total,
        "mean_iou": round(statistics.fmean(ious), 4) if ious else None,
        "mean_conf_delta": round(statistics.fmean(conf_deltas), 4) if conf_deltas else None,
        "max_conf_delta": round(max(conf_deltas), 4) if conf_deltas else None,
        "clean_match_rate": round(counters["matched"] / ref_total, 4) if ref_total else 1.0,
        "issues": issues,
    }


def _print_report(backend: str, run: dict, cmp_result: dict, ref_latency: dict, args) -> bool:
    c = cmp_result["counters"]
    lat = run["latency_ms"]
    speedup = ""
    if ref_latency.get("mean") and lat.get("mean"):
        speedup = f"  ({ref_latency['mean'] / lat['mean']:.2f}x vs PyTorch)"

    _log(f"\n--- {backend.upper()} ---")
    _log(f"  latency      : mean {lat['mean']} ms | median {lat['median']} ms | max {lat['max']} ms{speedup}")
    _log(f"  detections   : {cmp_result['cand_detections']} (reference: {cmp_result['ref_detections']})")
    _log(
        f"  mean IoU     : {cmp_result['mean_iou']}   mean |dconf|: {cmp_result['mean_conf_delta']}   "
        f"max |dconf|: {cmp_result['max_conf_delta']}"
    )
    _log(f"  clean matches: {c['matched']} ({cmp_result['clean_match_rate'] * 100:.1f}% of reference)")
    _log(f"  missed       : {c['missed']}")
    _log(f"  extra (FP)   : {c['extra']}")
    _log(f"  class swaps  : {c['class_mismatch']}")
    _log(f"  loose boxes  : {c['loose_box']}  (IoU < {args.iou_tol})")
    _log(f"  conf drift   : {c['conf_drift']}  (|dconf| > {args.conf_tol})")

    failures = c["missed"] + c["extra"] + c["class_mismatch"] + c["loose_box"] + c["conf_drift"]
    budget = max(0, round(cmp_result["ref_detections"] * args.max_drift_rate))
    passed = failures <= budget

    if failures:
        shown = cmp_result["issues"][: args.max_issues]
        _log(f"  first {len(shown)} of {len(cmp_result['issues'])} issue(s):")
        for issue in shown:
            ref_det, cand_det = issue.get("ref"), issue.get("cand")
            detail = f"    [{issue['kind']:<14}] {issue['image']}"
            if ref_det:
                detail += f" ref={ref_det['name']}@{ref_det['conf']:.2f}{ref_det['xyxy']}"
            if cand_det:
                detail += f" cand={cand_det['name']}@{cand_det['conf']:.2f}{cand_det['xyxy']}"
            _log(detail)

    verdict = "PASS" if passed else "FAIL"
    _log(
        f"  VERDICT      : {verdict}  ({failures} deviation(s), budget {budget} "
        f"= {args.max_drift_rate * 100:.0f}% of {cmp_result['ref_detections']})"
    )
    return passed


# ======================================================================
# Commands
# ======================================================================
def cmd_export(args: argparse.Namespace) -> int:
    weights = Path(args.weights)
    if not weights.exists():
        _log(f"ERROR: weights not found: {weights}")
        return 2
    test_dir = Path(args.test_dir)
    if "tflite" in args.formats and not test_dir.is_dir():
        _log(f"ERROR: INT8 TFLite calibration needs images in {test_dir}")
        return 2

    _rule(f"EXPORT  weights={weights.name}  imgsz={args.imgsz}  formats={', '.join(args.formats)}")
    if _is_arm():
        _log(
            "NOTE: running on aarch64. NCNN/ONNX export usually works here, but the INT8\n"
            "      TFLite path (onnx2tf + TensorFlow) is fragile on ARM. Prefer exporting on\n"
            "      an x86_64 Linux machine and copying models/ across -- see docs/PI5_DEPLOYMENT.md."
        )

    results: dict[str, str] = {}
    for fmt in args.formats:
        _log(f"\n[{fmt}] exporting in an isolated subprocess...")
        worker_args = [
            "_worker-export",
            "--format",
            fmt,
            "--weights",
            str(weights),
            "--imgsz",
            str(args.imgsz),
            "--opset",
            str(args.opset),
            "--test-dir",
            str(test_dir),
            "--fraction",
            str(args.fraction),
        ]
        if args.ncnn_half:
            worker_args.append("--ncnn-half")
        if args.keep_calibration:
            worker_args.append("--keep-calibration")

        ok, msg = _run_worker(worker_args, timeout=args.export_timeout)
        results[fmt] = "OK" if ok else f"FAILED ({msg})"
        if not ok:
            _log(f"[{fmt}] {results[fmt]}")

    _rule("EXPORT SUMMARY")
    for fmt, status in results.items():
        _log(f"  {fmt:<8} {status}")
    _log(f"\nManifest: {MANIFEST_PATH}")
    return 0 if all(v == "OK" for v in results.values()) else 1


def cmd_verify(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    exports = manifest.get("exports", {})

    targets: list[tuple[str, str, int]] = []  # (label, model path, imgsz)
    for fmt in args.formats:
        record = exports.get(fmt)
        if not record:
            _log(f"NOTE: no '{fmt}' entry in {MANIFEST_PATH.name}; skipping. Run `export` first.")
            continue
        path = (ROOT / record["path"]).resolve()
        if not path.exists():
            _log(f"NOTE: '{fmt}' artifact missing at {path}; skipping.")
            continue
        # NCNN and INT8 TFLite have their input resolution frozen at export time,
        # so each backend is replayed at the imgsz it was actually built for.
        targets.append((fmt, str(path), int(record.get("imgsz", args.imgsz))))

    if not targets:
        _log("Nothing to verify: no exported artifacts found.")
        return 2

    test_dir = Path(args.test_dir)
    images = list_images(test_dir, args.max_images)
    _rule(f"VERIFY  {len(images)} image(s) from {test_dir.name}  conf={args.conf}")
    _log(
        f"Tolerances: match IoU >= {args.match_iou} | pass IoU >= {args.iou_tol} | "
        f"|dconf| <= {args.conf_tol} | drift budget {args.max_drift_rate * 100:.0f}%"
    )

    work_dir = Path(args.work_dir) if args.work_dir else MODELS_DIR / ".verify"
    work_dir.mkdir(parents=True, exist_ok=True)

    def run_backend(label: str, model_path: str, imgsz: int, tag: str = "") -> dict | None:
        out = work_dir / f"{label}{tag}.json"
        _log(f"\n[{label}] inference sweep (imgsz={imgsz}) in an isolated subprocess...")
        ok, msg = _run_worker(
            [
                "_worker-predict",
                "--model",
                model_path,
                "--test-dir",
                str(test_dir),
                "--imgsz",
                str(imgsz),
                "--conf",
                str(args.conf),
                "--warmup",
                str(args.warmup),
                "--out",
                str(out),
            ]
            + (["--max-images", str(args.max_images)] if args.max_images else []),
            timeout=args.predict_timeout,
        )
        if not ok:
            _log(f"[{label}] FAILED: {msg}")
            return None
        return json.loads(out.read_text(encoding="utf-8"))

    # The PyTorch reference is replayed at each candidate's own imgsz. Comparing a
    # 416 NCNN model against a 640 PyTorch run would blame the compiler for drift
    # that is really just the resolution change -- two separate questions. Use
    # --reference-imgsz to pin the reference instead (e.g. to measure the combined
    # effect of switching from an existing 640 PyTorch deployment).
    reference_cache: dict[int, dict | None] = {}

    def reference_for(imgsz: int) -> dict | None:
        wanted = args.reference_imgsz or imgsz
        if wanted not in reference_cache:
            reference_cache[wanted] = run_backend("pytorch", str(Path(args.weights)), wanted, tag=f"_{wanted}")
            run = reference_cache[wanted]
            if run is not None:
                _log(
                    f"[pytorch] reference @ imgsz {wanted}: "
                    f"{sum(len(v) for v in run['images'].values())} detections, "
                    f"mean {run['latency_ms']['mean']} ms/image"
                )
        return reference_cache[wanted]

    _rule("ACCURACY DRIFT REPORT")
    # Three distinct outcomes per backend, tracked separately: a backend that
    # never ran (crashed / timed out / missing runtime) proves nothing about its
    # accuracy either way, and must not be reported the same as one that ran and
    # showed unacceptable drift. Conflating "couldn't test" with "failed" hides
    # exactly the information you'd want from a report like this.
    passed_backends: list[str] = []
    failed_backends: list[str] = []  # ran, drift exceeded tolerance
    untestable_backends: dict[str, str] = {}  # never ran here at all, with why
    report: dict[str, dict] = {}

    for label, model_path, imgsz in targets:
        reference = reference_for(imgsz)
        if reference is None:
            _log(f"[{label}] skipped: the PyTorch reference sweep failed.")
            untestable_backends[label] = "PyTorch reference sweep failed"
            continue
        run = run_backend(label, model_path, imgsz)
        if run is None:
            untestable_backends[label] = "could not run in this environment (see FAILED line above)"
            continue
        cmp_result = compare_runs(
            reference, run, match_iou=args.match_iou, iou_tol=args.iou_tol, conf_tol=args.conf_tol
        )
        passed = _print_report(label, run, cmp_result, reference["latency_ms"], args)
        (passed_backends if passed else failed_backends).append(label)
        report[label] = {
            "passed": passed,
            "latency_ms": run["latency_ms"],
            "imgsz": imgsz,
            "reference_imgsz": reference["imgsz"],
            "reference_latency_ms": reference["latency_ms"],
            **{k: v for k, v in cmp_result.items() if k != "issues"},
            "issues": cmp_result["issues"][: args.max_issues],
        }

    if not reference_cache:
        _log("\nNothing ran.")
        return 2

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "reference": {
                        "weights": str(args.weights),
                        "imgsz_runs": {str(k): v["latency_ms"] for k, v in reference_cache.items() if v is not None},
                    },
                    "tolerances": {
                        "match_iou": args.match_iou,
                        "iou_tol": args.iou_tol,
                        "conf_tol": args.conf_tol,
                        "max_drift_rate": args.max_drift_rate,
                    },
                    "backends": report,
                    "untestable": untestable_backends,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _log(f"\nJSON report written to {args.report}")

    if not args.keep_work:
        shutil.rmtree(work_dir, ignore_errors=True)

    _rule("SUMMARY")
    if passed_backends:
        _log(f"  PASSED      : {', '.join(passed_backends)}")
    if failed_backends:
        _log(f"  FAILED      : {', '.join(failed_backends)}  (drift exceeded tolerance -- do not deploy as-is)")
    if untestable_backends:
        _log(f"  UNTESTABLE  : {', '.join(untestable_backends)}  (could not run in this environment)")
        for label, why in untestable_backends.items():
            _log(f"    - {label}: {why}")
        _log("    A backend that can't run here (e.g. NCNN on non-ARM) says nothing about whether")
        _log("    it works on the actual Pi -- re-run `verify` there before trusting it.")

    # A drift failure always fails the run -- that's a real accuracy problem.
    # A backend that simply couldn't run here (crashed/missing runtime) does not,
    # as long as something else was actually verified clean; it's surfaced above
    # as UNTESTABLE rather than silently passing or wrongly failing the build.
    ok = bool(passed_backends) and not failed_backends
    _rule(
        "RESULT: "
        + ("PASS - verified backend(s) agree with PyTorch" if ok else "FAIL - review the drift above before deploying")
    )
    return 0 if ok else 1


def cmd_all(args: argparse.Namespace) -> int:
    export_rc = cmd_export(args)
    if export_rc != 0:
        _log("\nExport reported failures; verifying whatever did succeed.")
    verify_rc = cmd_verify(args)
    return max(export_rc, verify_rc)


def _is_arm() -> bool:
    import platform

    return platform.machine().lower() in ("aarch64", "arm64", "armv7l", "armv8l")


# ======================================================================
# CLI
# ======================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="source .pt checkpoint")
        p.add_argument("--test-dir", default=str(DEFAULT_TEST_DIR), help="images for calibration/verification")
        p.add_argument(
            "--imgsz",
            type=int,
            default=int(os.environ.get("YOLO_IMGSZ", "640")),
            help="inference resolution, multiple of 32 (416 is a good Pi 5 setting)",
        )
        p.add_argument("--formats", nargs="+", choices=SUPPORTED_FORMATS, default=list(SUPPORTED_FORMATS))

    def add_export_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--opset", type=int, default=12, help="ONNX opset")
        p.add_argument("--ncnn-half", action="store_true", help="fp16 NCNN weights (smaller, ~same NEON speed)")
        p.add_argument("--fraction", type=float, default=1.0, help="fraction of test-dir used for INT8 calibration")
        p.add_argument("--keep-calibration", action="store_true", help="keep the staged calibration dataset")
        p.add_argument("--export-timeout", type=int, default=3600, help="per-format export timeout (s)")

    def add_verify_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--conf", type=float, default=float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3")))
        p.add_argument(
            "--reference-imgsz",
            type=int,
            default=0,
            help="pin the PyTorch reference to this imgsz (0 = match each candidate, "
            "which isolates compilation drift from resolution drift)",
        )
        p.add_argument("--match-iou", type=float, default=0.5, help="IoU above which two boxes are the same object")
        p.add_argument("--iou-tol", type=float, default=0.85, help="matched boxes must reach this IoU to pass")
        p.add_argument("--conf-tol", type=float, default=0.10, help="max allowed |confidence delta| on a match")
        p.add_argument(
            "--max-drift-rate",
            type=float,
            default=0.02,
            help="fraction of reference detections allowed to deviate before FAIL",
        )
        p.add_argument("--max-images", type=int, default=0, help="cap images used (0 = all)")
        p.add_argument("--warmup", type=int, default=2, help="untimed warm-up inferences per backend")
        p.add_argument("--max-issues", type=int, default=15, help="issues printed per backend")
        p.add_argument("--report", default="", help="write a JSON drift report here")
        p.add_argument("--work-dir", default="", help="scratch dir for per-backend raw results")
        p.add_argument("--keep-work", action="store_true", help="keep raw per-backend JSON output")
        p.add_argument("--predict-timeout", type=int, default=3600, help="per-backend sweep timeout (s)")

    p_export = sub.add_parser("export", help="compile best.pt to NCNN / ONNX / INT8 TFLite")
    add_common(p_export)
    add_export_opts(p_export)
    p_export.set_defaults(func=cmd_export)

    p_verify = sub.add_parser("verify", help="regression-test compiled models against PyTorch")
    add_common(p_verify)
    add_verify_opts(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_all = sub.add_parser("all", help="export then verify")
    add_common(p_all)
    add_export_opts(p_all)
    add_verify_opts(p_all)
    p_all.set_defaults(func=cmd_all)

    # Internal worker entrypoints (subprocess isolation), hidden from --help.
    p_we = sub.add_parser("_worker-export")
    p_we.add_argument("--format", required=True)
    p_we.add_argument("--weights", required=True)
    p_we.add_argument("--imgsz", type=int, required=True)
    p_we.add_argument("--opset", type=int, default=12)
    p_we.add_argument("--test-dir", required=True)
    p_we.add_argument("--fraction", type=float, default=1.0)
    p_we.add_argument("--ncnn-half", action="store_true")
    p_we.add_argument("--keep-calibration", action="store_true")
    p_we.set_defaults(func=_worker_export)

    p_wp = sub.add_parser("_worker-predict")
    p_wp.add_argument("--model", required=True)
    p_wp.add_argument("--test-dir", required=True)
    p_wp.add_argument("--imgsz", type=int, required=True)
    p_wp.add_argument("--conf", type=float, required=True)
    p_wp.add_argument("--warmup", type=int, default=2)
    p_wp.add_argument("--max-images", type=int, default=0)
    p_wp.add_argument("--out", required=True)
    p_wp.set_defaults(func=_worker_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "max_images", 0) == 0:
        args.max_images = None
    try:
        return args.func(args)
    except KeyboardInterrupt:
        _log("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())

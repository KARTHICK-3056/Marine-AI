# Marine Zooplankton Detector

import os

# Must be set before importing ultralytics/gradio/torch, since all of them read
# these as module-level constants at import time.
# - YOLO_AUTOINSTALL: ultralytics will otherwise silently `pip install` a
#   missing/mismatched backend package (e.g. onnxruntime) at runtime. On a
#   Pi with no/limited internet this can hang or fail; even with internet,
#   it can pull in an unwanted multi-hundred-MB GPU package. Disabled by
#   default; this repo pins what it needs in requirements.txt instead.
# - GRADIO_ANALYTICS_ENABLED: skips a network call on startup, which is
#   pure latency/hang risk on an offline field device with no benefit here.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

# Raspberry Pi 5 has 4 physical CPU cores and no CUDA GPU, so inference is
# CPU-bound. Every numeric library underneath us (OpenMP for torch/ncnn,
# OpenBLAS, MKL, NumExpr, Accelerate) defaults to one thread per core *each*,
# so with four of them live at once the Pi ends up with ~16 runnable threads
# fighting over 4 cores -- more context-switching, worse cache locality, and
# latency spikes. Cap them all to the same budget. run_optimized.sh exports
# these too (they are read by native libs during `import`, before Python runs);
# setting them here keeps a bare `python app.py` equally well behaved.
CPU_THREADS = int(os.environ.get("CPU_THREADS", min(4, os.cpu_count() or 4)))
for _thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_var, str(CPU_THREADS))

import html
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import ImageOps
from ultralytics import YOLO

import camera

torch.set_num_threads(CPU_THREADS)
cv2.setNumThreads(CPU_THREADS)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EXPORT_MANIFEST = MODELS_DIR / "exports.json"

# Cap the working image size so a single very large upload (e.g. 8K) can't
# blow up inference latency/memory on the synchronous request path. Lower
# this (e.g. 1280) on Raspberry Pi to reduce per-request preprocessing cost.
MAX_DIMENSION = int(os.environ.get("MAX_DIMENSION", "2048"))

# YOLO always internally resizes to this before the forward pass regardless
# of MAX_DIMENSION above. This is the main speed/accuracy dial on CPU-only
# hardware: dropping it (e.g. to 480 or 416) meaningfully cuts inference
# latency on a Pi at some accuracy cost. Must be a multiple of 32.
# NOTE: for compiled backends (NCNN / INT8 TFLite / static ONNX) the input
# resolution is frozen at export time, so the manifest value wins -- see
# resolve_model().
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))

# Minimum confidence a detection needs to be kept. Lower to catch more
# (noisier) detections, raise to cut false positives, without editing code.
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))

# ==============================
# DEFINE CLASS COLORS
# ==============================
CLASS_COLORS = {
    "chaetognath": (255, 0, 0),  # Red
    "larval fish": (0, 255, 0),  # Green
    "Hydromedusa": (0, 0, 255),  # Blue
    "lobate ctenophore": (255, 255, 0),  # Yellow
    "Pleurobrachia": (255, 0, 255),  # Magenta
    "shrimp": (0, 255, 255),  # Cyan
    "Siphonophore": (128, 0, 128),  # Purple
    "stomatopod larva": (0, 128, 128),  # Teal
    "Unknown": (128, 128, 128),  # Gray
    "Thaliac": (255, 165, 0),  # Orange
    "polychaete worm": (75, 0, 130),  # Indigo
    "Cumacean": (238, 130, 238),  # Violet
    "ctenophore": (0, 100, 0),  # Dark Green
}

# ==============================
# BACKEND SELECTION
# ==============================
BACKEND_PRIORITY = ("ncnn", "tflite", "onnx", "pytorch")
STATIC_IMGSZ_BACKENDS = ("ncnn", "tflite", "onnx")


def _detect_backend(path: Path) -> str:
    name = path.name.lower()
    if name.endswith("ncnn_model"):
        return "ncnn"
    if name.endswith(".tflite"):
        return "tflite"
    if name.endswith(".onnx"):
        return "onnx"
    if name.endswith((".pt", ".pth")):
        return "pytorch"
    return "unknown"


def _load_export_manifest() -> dict:
    """Read models/exports.json, written by optimize_model.py."""
    if not EXPORT_MANIFEST.exists():
        return {}
    try:
        return json.loads(EXPORT_MANIFEST.read_text(encoding="utf-8")).get("exports", {})
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: ignoring unreadable {EXPORT_MANIFEST.name}: {e}")
        return {}


def _describe(path: Path, record: dict) -> tuple[Path, str, int, dict | None]:
    """Turn a (path, manifest record) pair into a full candidate tuple."""
    backend = _detect_backend(path)
    imgsz = YOLO_IMGSZ
    if backend in STATIC_IMGSZ_BACKENDS and record.get("imgsz"):
        imgsz = int(record["imgsz"])
    names = {int(k): v for k, v in record.get("names", {}).items()} if record.get("names") else None
    return path, backend, imgsz, names


def candidate_models() -> list[tuple[Path, str, int, dict | None]]:
    manifest = _load_export_manifest()
    requested = os.environ.get("MODEL_BACKEND", "auto").strip().lower()
    fallback = MODELS_DIR / "best.pt"

    def entry_for(path: Path) -> dict:
        for record in manifest.values():
            if (BASE_DIR / record.get("path", "")).resolve() == path:
                return record
        return {}

    explicit = os.environ.get("MODEL_PATH")
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        return [_describe(path, entry_for(path))]

    if requested != "auto":
        record = manifest.get(requested)
        if not record:
            print(
                f"WARNING: MODEL_BACKEND={requested} requested but not in {EXPORT_MANIFEST.name}; "
                "falling back to best.pt. Run: python optimize_model.py export"
            )
            return [_describe(fallback, {})]
        return [_describe((BASE_DIR / record["path"]).resolve(), record)]

    candidates = []
    for backend in BACKEND_PRIORITY:
        record = manifest.get(backend)
        if not record:
            continue
        path = (BASE_DIR / record["path"]).resolve()
        if path.exists():
            candidates.append(_describe(path, record))
    candidates.append(_describe(fallback, {}))
    return candidates


def resolve_model():
    return candidate_models()[0]


BACKEND_PROBE_TIMEOUT = int(os.environ.get("BACKEND_PROBE_TIMEOUT", "120"))

_PROBE_SOURCE = """
import sys
import numpy as np
from ultralytics import YOLO

path, imgsz = sys.argv[1], int(sys.argv[2])
model = YOLO(path, task="detect")
model.predict(source=np.zeros((imgsz, imgsz, 3), dtype=np.uint8), imgsz=imgsz, verbose=False, save=False)
"""


def probe_backend(path: Path, imgsz: int) -> tuple[bool, str]:
    import subprocess
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE, str(path), str(imgsz)],
            capture_output=True,
            timeout=BACKEND_PROBE_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {BACKEND_PROBE_TIMEOUT}s"
    except OSError as e:
        return True, f"probe could not run ({e}); assuming the backend is usable"

    if proc.returncode == 0:
        return True, "ok"
    if proc.returncode < 0:
        return False, f"native crash (signal {-proc.returncode})"
    if os.name == "nt" and proc.returncode > 0x80000000:
        return False, f"native crash (Windows exception 0x{proc.returncode:08X})"
    tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
    return False, tail[-1] if tail else f"exit code {proc.returncode}"


def _probe_enabled(candidate_count: int) -> bool:
    mode = os.environ.get("BACKEND_PROBE", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "on"):
        return True
    return candidate_count > 1


def _names_look_generic(names) -> bool:
    try:
        values = list(names.values()) if isinstance(names, dict) else list(names)
    except (AttributeError, TypeError):
        return False
    return bool(values) and all(str(v).startswith("class") for v in values)


# ==============================
# LOAD YOUR TRAINED MODEL
# ==============================
def load_model():
    candidates = candidate_models()
    probe = _probe_enabled(len(candidates))
    last = (None, str(candidates[-1][0]), candidates[-1][1], candidates[-1][2])

    for index, (model_path, backend, imgsz, manifest_names) in enumerate(candidates):
        is_last = index == len(candidates) - 1
        print(f"Loading model from: {model_path}  (backend={backend}, imgsz={imgsz})")

        if not model_path.exists():
            print(f"  not found at {model_path}")
            last = (None, str(model_path), backend, imgsz)
            continue

        if probe and not is_last:
            ok, detail = probe_backend(model_path, imgsz)
            if not ok:
                print(f"  backend probe FAILED for {backend}: {detail}")
                print(f"  falling back to {candidates[index + 1][1]}")
                last = (None, str(model_path), backend, imgsz)
                continue

        try:
            model = YOLO(str(model_path), task="detect")
            if manifest_names and _names_look_generic(model.names):
                print("NOTE: exported model had no class metadata; restoring names from exports.json")
                model.names = manifest_names
            print(f"Model loaded successfully! ({backend}, {len(model.names)} classes)")
            if imgsz != YOLO_IMGSZ:
                print(
                    f"NOTE: this {backend} model was compiled at imgsz={imgsz}, so YOLO_IMGSZ="
                    f"{YOLO_IMGSZ} is ignored. Re-export with --imgsz {YOLO_IMGSZ} to change it."
                )
            return model, str(model_path), backend, imgsz
        except (RuntimeError, ValueError, OSError, ImportError, AttributeError, KeyError) as e:
            print(f"  failed to load: {e}")
            if backend != "pytorch":
                print(
                    "  hint: install this backend's runtime on the Pi "
                    "(ncnn / tflite-runtime / onnxruntime), or set MODEL_BACKEND=pytorch."
                )
            last = (None, str(model_path), backend, imgsz)

    print("ERROR: no usable model backend found.")
    return last


model, MODEL_PATH, MODEL_BACKEND, ACTIVE_IMGSZ = load_model()


# ==============================
# HTML SNIPPETS
# ==============================
def _error_card(title: str, message: str) -> str:
    return f"""
    <div style='padding: 16px 20px; background: #161b22; border-left: 4px solid #f85149; border-radius: 8px; color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <h4 style='margin: 0 0 6px 0; font-size: 16px; font-weight: 600; color: #f85149;'>{html.escape(title)}</h4>
        <p style='margin: 0; opacity: 0.85; font-size: 14px;'>{html.escape(message)}</p>
    </div>
    """


MODEL_MISSING_HTML = _error_card(
    "Model Not Found",
    "Please ensure 'best.pt' is in your directory and restart the application.",
)
ANALYSIS_FAILED_HTML = _error_card(
    "Analysis Failed",
    "Something went wrong while analyzing this image. Please try a different image.",
)


# ==============================
# DETECTION CORE
# ==============================
def _fit_max_dimension(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, MAX_DIMENSION / max(h, w))
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def analyze_rgb(img: np.ndarray):
    img = _fit_max_dimension(img)
    original_img = img.copy()
    img_rgb = img.copy()

    started = time.perf_counter()
    results = model.predict(
        source=img, conf=CONFIDENCE_THRESHOLD, imgsz=ACTIVE_IMGSZ, save=False, show=False, verbose=False
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    detections_by_class = defaultdict(int)
    total_confidence = defaultdict(list)

    ref_dim = max(img_rgb.shape[:2])
    box_thickness = max(1, round(ref_dim / 700))
    font_scale = max(0.4, ref_dim / 2800)
    text_thickness = max(1, round(ref_dim / 1000))

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = model.names[cls_id]

        detections_by_class[class_name] += 1
        total_confidence[class_name].append(confidence)

        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, box_thickness)

        label = f"{class_name}"
        thickness = text_thickness
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_rgb, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img_rgb, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    report = build_report(detections_by_class, total_confidence, elapsed_ms)
    return original_img, img_rgb, report


def build_report(detections_by_class, total_confidence, elapsed_ms: float) -> str:
    if not detections_by_class:
        return f"""
        <div style='padding: 24px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; text-align: center; color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h3 style='margin: 0 0 8px 0; font-size: 18px; font-weight: 600; color: #8b949e;'>No Marine Life Detected</h3>
            <p style='margin: 0; font-size: 14px; opacity: 0.7;'>Try uploading a different sample or adjusting the confidence threshold.</p>
            <p style='margin: 12px 0 0 0; font-size: 12px; color: #6e7681;'>Backend: {MODEL_BACKEND} &bull; Resolution: {ACTIVE_IMGSZ} &bull; Latency: {elapsed_ms:.1f} ms</p>
        </div>
        """

    total_detections = sum(detections_by_class.values())
    sorted_detections = sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_detections[0][1] if sorted_detections else 1
    diversity = len(detections_by_class)

    rows_html = ""
    for class_name, count in sorted_detections:
        rgb_color = CLASS_COLORS.get(class_name, (128, 128, 128))
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)
        avg_conf = np.mean(total_confidence[class_name]) * 100 if total_confidence[class_name] else 0
        bar_width = min(100, max(8, int((count / max_count) * 100)))
        safe_name = html.escape(class_name)

        rows_html += f"""
        <tr style='border-bottom: 1px solid #21262d;'>
            <td style='padding: 10px 12px; vertical-align: middle;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='display: inline-block; width: 12px; height: 12px; border-radius: 3px; background-color: {hex_color}; flex-shrink: 0;'></span>
                    <span style='font-weight: 500; color: #f0f6fc;'>{safe_name}</span>
                </div>
            </td>
            <td style='padding: 10px 12px; text-align: right; font-weight: 600; color: #f0f6fc;'>{count}</td>
            <td style='padding: 10px 12px; text-align: right; color: #8b949e; font-size: 13px;'>{avg_conf:.1f}%</td>
            <td style='padding: 10px 12px; width: 35%; vertical-align: middle;'>
                <div style='width: 100%; background: #21262d; height: 6px; border-radius: 3px; overflow: hidden;'>
                    <div style='width: {bar_width}%; background: {hex_color}; height: 100%; border-radius: 3px;'></div>
                </div>
            </td>
        </tr>
        """

    return f"""
    <div style='background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #21262d;'>
            <div>
                <h3 style='margin: 0; font-size: 18px; font-weight: 600;'>Taxonomic Classification Report</h3>
                <p style='margin: 4px 0 0 0; font-size: 13px; color: #8b949e;'>Inference Engine: {MODEL_BACKEND} ({ACTIVE_IMGSZ}px) &bull; {elapsed_ms:.1f} ms</p>
            </div>
            <span style='padding: 4px 10px; background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); color: #38bdf8; font-size: 12px; font-weight: 600; border-radius: 4px;'>ANALYSIS COMPLETE</span>
        </div>

        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;'>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #38bdf8;'>{total_detections}</div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Total Count</div>
            </div>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #34d399;'>{diversity}</div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Taxa Diversity</div>
            </div>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #fbbf24;'>{elapsed_ms:.0f}<span style='font-size: 14px; font-weight: 400;'> ms</span></div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Inference Time</div>
            </div>
        </div>

        <div style='background: #0d1117; border: 1px solid #21262d; border-radius: 6px; overflow: hidden;'>
            <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                <thead>
                    <tr style='background: #161b22; border-bottom: 1px solid #21262d; text-align: left;'>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600;'>Taxa Classification</th>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600; text-align: right;'>Count</th>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600; text-align: right;'>Avg Conf</th>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600;'>Abundance</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """


# ==============================
# DETECTION FUNCTION (UPLOAD PATH)
# ==============================
def detect_zooplankton(input_image):
    if input_image is None:
        return (
            None,
            None,
            "<div style='text-align: center; color: #ff6b6b; font-size: 18px;'>Please upload an image first.</div>",
        )

    if model is None:
        return input_image, input_image, MODEL_MISSING_HTML

    try:
        input_image = ImageOps.exif_transpose(input_image)
        img = np.array(input_image.convert("RGB"))
        return analyze_rgb(img)
    except (RuntimeError, ValueError, TypeError, OSError, cv2.error, KeyError, AttributeError) as e:
        print(f"Detection failed: {e}")
        return input_image, input_image, ANALYSIS_FAILED_HTML


# ==============================
# LIVE MICROSCOPE PATH
# ==============================
CAMERA_ENABLED = os.environ.get("ENABLE_CAMERA", "true").strip().lower() in ("1", "true", "yes")
_HAS_TIMER = hasattr(gr, "Timer")


def _camera_status_html(message: str, ok: bool = True) -> str:
    color = "#10b981" if ok else "#ff6b6b"
    return f"<div style='padding:10px;border-radius:8px;color:{color};font-size:14px;'>{html.escape(message)}</div>"


def camera_start():
    try:
        stream = camera.get_stream()
        frame, _, _ = stream.wait_for_frame(timeout=5.0)
        if frame is None:
            return None, _camera_status_html(
                f"Camera opened but delivered no frames within 5 s. Last error: {stream.last_error or 'none'}", ok=False
            )
        status = stream.status()
        return (
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            _camera_status_html(f"Live • {status['resolution']} • device {status['device']}"),
        )
    except camera.CameraError as e:
        return None, _camera_status_html(str(e), ok=False)
    except (RuntimeError, ValueError, OSError, cv2.error) as e:
        print(f"Camera start failed: {e}")
        return None, _camera_status_html(f"Camera start failed: {e}", ok=False)


def camera_stop():
    camera.release_stream()
    return None, _camera_status_html("Camera stopped.")


def camera_preview():
    stream = camera.current_stream()
    if stream is None or not stream.running:
        return gr.update()
    frame, _, _ = stream.read()
    if frame is None:
        return gr.update()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def camera_capture_and_analyze():
    if model is None:
        return None, None, MODEL_MISSING_HTML

    stream = camera.current_stream()
    if stream is None or not stream.running:
        return None, None, _error_card("Camera Not Running", "Press 'Start Camera' first.")

    frame, _, _ = stream.wait_for_frame(timeout=5.0)
    if frame is None:
        return (
            None,
            None,
            _error_card("No Frame Available", stream.last_error or "The microscope did not deliver a frame in time."),
        )

    try:
        return analyze_rgb(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except (RuntimeError, ValueError, TypeError, OSError, cv2.error, KeyError, AttributeError) as e:
        print(f"Live detection failed: {e}")
        return None, None, ANALYSIS_FAILED_HTML


def camera_diagnostics():
    stream = camera.current_stream()
    if stream is None:
        return _camera_status_html("Camera not started.", ok=False)
    status = stream.status()
    rows = "".join(
        f"<tr><td style='padding:4px 12px 4px 0;opacity:0.7;'>{html.escape(str(k))}</td>"
        f"<td style='padding:4px 0;'>{html.escape(str(v))}</td></tr>"
        for k, v in status.items()
    )
    return f"<table style='font-size:14px;'>{rows}</table>"


# ==============================
# VIDEO SAMPLE ANALYSIS PATH
# ==============================
VIDEO_DIR = BASE_DIR / "video_sample"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
OUTPUT_DIR = BASE_DIR / "outputs"

# Analyzing every frame of a video would take (clip duration) x (per-frame
# YOLO latency) on CPU-only hardware -- for a 30fps clip that is ~30x longer
# than the clip itself. Sample every Nth frame instead to keep total request
# time bounded; raise/lower via env without touching code.
VIDEO_FRAME_STRIDE = max(1, int(os.environ.get("VIDEO_FRAME_STRIDE", "5")))
# Hard cap on frames processed per video so an unexpectedly long upload can't
# hang the request indefinitely.
VIDEO_MAX_FRAMES = int(os.environ.get("VIDEO_MAX_FRAMES", "300"))


def _load_sample_videos() -> list[str]:
    if not VIDEO_DIR.exists():
        return []
    return sorted(str(p) for p in VIDEO_DIR.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)


def _annotate_frame(frame_rgb: np.ndarray, results, detections_by_class, total_confidence) -> np.ndarray:
    annotated = frame_rgb.copy()
    ref_dim = max(annotated.shape[:2])
    box_thickness = max(1, round(ref_dim / 700))
    font_scale = max(0.4, ref_dim / 2800)
    text_thickness = max(1, round(ref_dim / 1000))

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls.item())
        confidence = float(box.conf.item())
        class_name = model.names[cls_id]

        detections_by_class[class_name] += 1
        total_confidence[class_name].append(confidence)

        color = CLASS_COLORS.get(class_name, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)

        label = class_name
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        cv2.rectangle(annotated, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), text_thickness
        )

    return annotated


def build_video_report(
    detections_by_class, total_confidence, elapsed_ms: float, frames_processed: int, frames_total: int
) -> str:
    if not detections_by_class:
        return f"""
        <div style='padding: 24px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; text-align: center; color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
            <h3 style='margin: 0 0 8px 0; font-size: 18px; font-weight: 600; color: #8b949e;'>No Marine Life Detected</h3>
            <p style='margin: 0; font-size: 14px; opacity: 0.7;'>Try a different video sample or adjusting the confidence threshold.</p>
            <p style='margin: 12px 0 0 0; font-size: 12px; color: #6e7681;'>Backend: {MODEL_BACKEND} &bull; Frames analyzed: {frames_processed}/{frames_total} &bull; Latency: {elapsed_ms:.0f} ms</p>
        </div>
        """

    total_detections = sum(detections_by_class.values())
    sorted_detections = sorted(detections_by_class.items(), key=lambda x: x[1], reverse=True)
    max_count = sorted_detections[0][1] if sorted_detections else 1
    diversity = len(detections_by_class)

    rows_html = ""
    for class_name, count in sorted_detections:
        rgb_color = CLASS_COLORS.get(class_name, (128, 128, 128))
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb_color)
        avg_conf = np.mean(total_confidence[class_name]) * 100 if total_confidence[class_name] else 0
        bar_width = min(100, max(8, int((count / max_count) * 100)))
        safe_name = html.escape(class_name)

        rows_html += f"""
        <tr style='border-bottom: 1px solid #21262d;'>
            <td style='padding: 10px 12px; vertical-align: middle;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='display: inline-block; width: 12px; height: 12px; border-radius: 3px; background-color: {hex_color}; flex-shrink: 0;'></span>
                    <span style='font-weight: 500; color: #f0f6fc;'>{safe_name}</span>
                </div>
            </td>
            <td style='padding: 10px 12px; text-align: right; font-weight: 600; color: #f0f6fc;'>{count}</td>
            <td style='padding: 10px 12px; text-align: right; color: #8b949e; font-size: 13px;'>{avg_conf:.1f}%</td>
            <td style='padding: 10px 12px; width: 35%; vertical-align: middle;'>
                <div style='width: 100%; background: #21262d; height: 6px; border-radius: 3px; overflow: hidden;'>
                    <div style='width: {bar_width}%; background: {hex_color}; height: 100%; border-radius: 3px;'></div>
                </div>
            </td>
        </tr>
        """

    return f"""
    <div style='background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid #21262d;'>
            <div>
                <h3 style='margin: 0; font-size: 18px; font-weight: 600;'>Video Taxonomic Classification Report</h3>
                <p style='margin: 4px 0 0 0; font-size: 13px; color: #8b949e;'>Inference Engine: {MODEL_BACKEND} ({ACTIVE_IMGSZ}px) &bull; {frames_processed}/{frames_total} frames analyzed &bull; {elapsed_ms:.0f} ms total</p>
            </div>
            <span style='padding: 4px 10px; background: rgba(14, 165, 233, 0.15); border: 1px solid rgba(14, 165, 233, 0.4); color: #38bdf8; font-size: 12px; font-weight: 600; border-radius: 4px;'>ANALYSIS COMPLETE</span>
        </div>

        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;'>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #38bdf8;'>{total_detections}</div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Total Detections</div>
            </div>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #34d399;'>{diversity}</div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Taxa Diversity</div>
            </div>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #fbbf24;'>{frames_processed}</div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Frames Analyzed</div>
            </div>
            <div style='padding: 14px; background: #0d1117; border: 1px solid #21262d; border-radius: 6px; text-align: center;'>
                <div style='font-size: 24px; font-weight: 700; color: #f0f6fc;'>{elapsed_ms / 1000:.1f}<span style='font-size: 14px; font-weight: 400;'> s</span></div>
                <div style='font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px;'>Processing Time</div>
            </div>
        </div>

        <div style='background: #0d1117; border: 1px solid #21262d; border-radius: 6px; overflow: hidden;'>
            <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                <thead>
                    <tr style='background: #161b22; border-bottom: 1px solid #21262d; text-align: left;'>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600;'>Taxa Classification</th>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600; text-align: right;'>Count</th>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600; text-align: right;'>Avg Conf</th>
                        <th style='padding: 10px 12px; font-size: 12px; text-transform: uppercase; color: #8b949e; font-weight: 600;'>Abundance</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </div>
    """


_DEFAULT_PROGRESS = gr.Progress()


def analyze_video(video_path, progress=_DEFAULT_PROGRESS):
    """Generator: yields (live_frame, report_html, download_path) per analyzed
    frame so the UI can show boxes appearing in real time, instead of only a
    finished video file -- browsers can't inline-play the MPEG-4 (mp4v) codec
    OpenCV falls back to here (no working H.264 encoder on this system), so a
    streamed image is the reliable way to actually *see* live annotations.
    The compiled mp4 is still written to disk and handed back at the end as a
    download for offline playback (e.g. VLC), not for in-browser preview.
    """
    if not video_path:
        yield (
            None,
            "<div style='text-align: center; color: #ff6b6b; font-size: 18px;'>Please upload or select a video sample first.</div>",
            None,
        )
        return

    if model is None:
        yield None, MODEL_MISSING_HTML, None
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield None, _error_card("Video Failed To Open", f"Could not open '{Path(video_path).name}'."), None
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{Path(video_path).stem}_annotated.mp4"

    writer = None
    detections_by_class = defaultdict(int)
    total_confidence = defaultdict(list)
    frames_processed = 0
    frame_index = 0
    last_annotated = None
    started = time.perf_counter()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_index % VIDEO_FRAME_STRIDE == 0:
                if frames_processed >= VIDEO_MAX_FRAMES:
                    break

                frame_rgb = _fit_max_dimension(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                results = model.predict(
                    source=frame_rgb,
                    conf=CONFIDENCE_THRESHOLD,
                    imgsz=ACTIVE_IMGSZ,
                    save=False,
                    show=False,
                    verbose=False,
                )
                annotated = _annotate_frame(frame_rgb, results, detections_by_class, total_confidence)
                last_annotated = annotated

                if writer is None:
                    h, w = annotated.shape[:2]
                    out_fps = max(1.0, src_fps / VIDEO_FRAME_STRIDE)
                    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))
                writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
                frames_processed += 1

                if total_frames:
                    progress(min(1.0, frame_index / total_frames), desc=f"Analyzing frame {frame_index}/{total_frames}")

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                live_report = build_video_report(
                    detections_by_class, total_confidence, elapsed_ms, frames_processed, frame_index + 1
                )
                yield annotated, live_report, None

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    if frames_processed == 0 or writer is None:
        yield None, _error_card("No Frames Processed", "The video contained no readable frames."), None
        return

    report = build_video_report(detections_by_class, total_confidence, elapsed_ms, frames_processed, frame_index)
    yield last_annotated, report, str(out_path)


# ==============================
# UI HELPERS & GALLERY LOADER
# ==============================
def _load_sample_gallery(limit=8):
    test_dir = BASE_DIR / "test_img"
    if not test_dir.exists():
        return []
    images = sorted([str(p) for p in test_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not images:
        return []
    step = max(1, len(images) // limit)
    return images[::step][:limit]


def _build_system_info_html():
    legend_rows = ""
    for class_name, rgb in sorted(CLASS_COLORS.items()):
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        legend_rows += f"""
        <tr style='border-bottom: 1px solid #21262d;'>
            <td style='padding: 8px 12px; font-weight: 500; color: #f0f6fc;'>
                <span style='display: inline-block; width: 10px; height: 10px; border-radius: 2px; background-color: {hex_color}; margin-right: 8px;'></span>
                {html.escape(class_name)}
            </td>
            <td style='padding: 8px 12px; color: #8b949e; font-size: 13px; font-family: monospace;'>{hex_color}</td>
        </tr>
        """

    return f"""
    <div style='background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; color: #f0f6fc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <h3 style='margin: 0 0 16px 0; font-size: 18px; font-weight: 600;'>System Configuration & Parameters</h3>
        <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 24px;'>
            <div style='background: #0d1117; padding: 12px 16px; border: 1px solid #21262d; border-radius: 6px;'>
                <div style='font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;'>Model Artifact</div>
                <div style='font-size: 14px; font-weight: 600; color: #f0f6fc; margin-top: 4px;'>{Path(MODEL_PATH).name}</div>
            </div>
            <div style='background: #0d1117; padding: 12px 16px; border: 1px solid #21262d; border-radius: 6px;'>
                <div style='font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;'>Inference Backend</div>
                <div style='font-size: 14px; font-weight: 600; color: #38bdf8; margin-top: 4px;'>{MODEL_BACKEND.upper()}</div>
            </div>
            <div style='background: #0d1117; padding: 12px 16px; border: 1px solid #21262d; border-radius: 6px;'>
                <div style='font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;'>Input Resolution</div>
                <div style='font-size: 14px; font-weight: 600; color: #f0f6fc; margin-top: 4px;'>{ACTIVE_IMGSZ} x {ACTIVE_IMGSZ}</div>
            </div>
            <div style='background: #0d1117; padding: 12px 16px; border: 1px solid #21262d; border-radius: 6px;'>
                <div style='font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;'>Confidence Cutoff</div>
                <div style='font-size: 14px; font-weight: 600; color: #f0f6fc; margin-top: 4px;'>{CONFIDENCE_THRESHOLD * 100:.0f}%</div>
            </div>
            <div style='background: #0d1117; padding: 12px 16px; border: 1px solid #21262d; border-radius: 6px;'>
                <div style='font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;'>Thread Budget</div>
                <div style='font-size: 14px; font-weight: 600; color: #f0f6fc; margin-top: 4px;'>{CPU_THREADS} CPU Cores</div>
            </div>
        </div>

        <h4 style='margin: 0 0 12px 0; font-size: 15px; font-weight: 600;'>Taxonomic Class Palette (13 Target Classes)</h4>
        <div style='background: #0d1117; border: 1px solid #21262d; border-radius: 6px; overflow: hidden;'>
            <table style='width: 100%; border-collapse: collapse; font-size: 14px;'>
                <thead>
                    <tr style='background: #161b22; border-bottom: 1px solid #21262d; text-align: left;'>
                        <th style='padding: 8px 12px; font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: 600;'>Taxon</th>
                        <th style='padding: 8px 12px; font-size: 12px; color: #8b949e; text-transform: uppercase; font-weight: 600;'>HEX Color Code</th>
                    </tr>
                </thead>
                <tbody>
                    {legend_rows}
                </tbody>
            </table>
        </div>
    </div>
    """


# ==============================
# GRADIO INTERFACE
# ==============================
sample_images = _load_sample_gallery(limit=8)
sample_videos = _load_sample_videos()

with gr.Blocks(
    title="Marine AI - Embedded Intelligent Microscopy System",
    theme=gr.themes.Base(
        primary_hue="sky",
        neutral_hue="slate",
    ),
    css="""
    body, .gradio-container { background-color: #0d1117 !important; color: #f0f6fc !important; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important; }
    .gradio-container { max-width: 1240px !important; margin: 0 auto !important; padding: 16px !important; }
    footer { display: none !important; }
    .primary-btn { background-color: #0284c7 !important; border: none !important; color: #ffffff !important; font-weight: 600 !important; border-radius: 6px !important; }
    .primary-btn:hover { background-color: #0369a1 !important; }
    .panel-box { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px !important; }
    """,
) as demo:
    gr.HTML("""
    <div style='display: flex; justify-content: space-between; align-items: center; padding: 16px 20px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 16px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;'>
        <div>
            <h1 style='margin: 0; font-size: 22px; font-weight: 700; color: #f0f6fc; letter-spacing: -0.3px;'>Marine AI</h1>
            <p style='margin: 4px 0 0 0; font-size: 14px; color: #8b949e;'>Embedded Intelligent Microscopy System for Zooplankton Identification</p>
        </div>
        <div style='text-align: right;'>
            <span style='display: inline-block; padding: 4px 10px; background: #0d1117; border: 1px solid #30363d; border-radius: 4px; font-size: 12px; color: #38bdf8; font-weight: 600;'>
                OFFLINE OPERATIONAL MODE
            </span>
        </div>
    </div>
    """)

    with gr.Tabs():
        with gr.Tab("Sample Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Input Sample Image")
                    # Explicitly restricting sources to upload and clipboard hides the webcam button
                    input_image = gr.Image(
                        label="Upload or Select Sample Image",
                        type="pil",
                        height=320,
                        sources=["upload", "clipboard"],
                    )
                    detect_btn = gr.Button(
                        "Execute Classification & Counting", variant="primary", size="lg", elem_classes=["primary-btn"]
                    )

                    if sample_images:
                        gr.Markdown("#### Preset Test Samples")
                        gr.Examples(
                            examples=sample_images,
                            inputs=input_image,
                            label="Click a sample to load",
                            examples_per_page=8,
                        )
                with gr.Column(scale=1):
                    gr.Markdown("#### Processing Output")
                    with gr.Row():
                        before_image = gr.Image(label="Original Frame", interactive=False, height=240)
                        after_image = gr.Image(label="Annotated Detections", interactive=False, height=240)
                    output_report = gr.HTML(
                        value="<div style='padding: 24px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; text-align: center; color: #8b949e;'>Select or upload an image and click Execute Classification to run inference.</div>"
                    )

            detect_btn.click(
                fn=detect_zooplankton,
                inputs=input_image,
                outputs=[before_image, after_image, output_report],
                concurrency_limit=1,
            )

        if CAMERA_ENABLED:
            with gr.Tab("Live Microscope"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("#### USB Digital Microscope Controls")
                        with gr.Row():
                            start_btn = gr.Button("Start Camera Stream", variant="primary", size="sm")
                            stop_btn = gr.Button("Stop Stream", size="sm")

                        live_toggle = gr.Checkbox(
                            label="Auto-refresh preview (~2 FPS)" if _HAS_TIMER else "Auto-refresh requires Gradio 5+",
                            value=False,
                            interactive=_HAS_TIMER,
                        )
                        refresh_btn = gr.Button("Refresh Frame", size="sm")
                        capture_btn = gr.Button(
                            "Capture & Analyze Frame", variant="primary", size="lg", elem_classes=["primary-btn"]
                        )
                        camera_status = gr.HTML(value=_camera_status_html("Microscope stream offline."))

                        with gr.Accordion("Camera Hardware Diagnostics", open=False):
                            diag_btn = gr.Button("Poll Diagnostics", size="sm")
                            diag_html = gr.HTML(value="")

                    with gr.Column(scale=1):
                        gr.Markdown("#### Live View & Results")
                        preview_image = gr.Image(label="Microscope Live Feed", interactive=False, height=280)
                        with gr.Row():
                            cam_before = gr.Image(label="Captured Frame", interactive=False, height=200)
                            cam_after = gr.Image(label="Annotated Results", interactive=False, height=200)
                        cam_report = gr.HTML(
                            value="<div style='padding: 20px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; text-align: center; color: #8b949e;'>Start microscope stream and click Capture & Analyze.</div>"
                        )

                start_btn.click(fn=camera_start, outputs=[preview_image, camera_status])
                stop_btn.click(fn=camera_stop, outputs=[preview_image, camera_status])
                refresh_btn.click(fn=camera_preview, outputs=preview_image)
                capture_btn.click(
                    fn=camera_capture_and_analyze,
                    outputs=[cam_before, cam_after, cam_report],
                    concurrency_limit=1,
                )
                diag_btn.click(fn=camera_diagnostics, outputs=diag_html)

                if _HAS_TIMER:
                    preview_timer = gr.Timer(0.5, active=False)
                    preview_timer.tick(fn=camera_preview, outputs=preview_image, show_progress="hidden")
                    live_toggle.change(
                        fn=lambda on: gr.Timer(active=bool(on)),
                        inputs=live_toggle,
                        outputs=preview_timer,
                    )

        with gr.Tab("Video Analysis"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Input Sample Video")
                    input_video = gr.Video(
                        label="Upload or Select Sample Video",
                        sources=["upload"],
                        height=320,
                    )
                    video_detect_btn = gr.Button(
                        "Execute Video Classification & Counting",
                        variant="primary",
                        size="lg",
                        elem_classes=["primary-btn"],
                    )
                    gr.Markdown(
                        f"<span style='font-size:12px;color:#8b949e;'>Frames are sampled every {VIDEO_FRAME_STRIDE} frame(s) "
                        f"(up to {VIDEO_MAX_FRAMES}) to keep processing time bounded on CPU-only hardware.</span>"
                    )

                    if sample_videos:
                        # gr.Examples eagerly copies every example file into
                        # Gradio's cache dir at startup -- fine for small
                        # images, but these microscope captures run into the
                        # gigabytes, so a dropdown that hands back the
                        # existing path (no copy) is used instead.
                        video_names = {Path(p).name: p for p in sample_videos}
                        gr.Markdown("#### Preset Video Samples")
                        sample_video_picker = gr.Dropdown(
                            choices=list(video_names.keys()),
                            label="Select a preset sample",
                            value=None,
                        )
                        sample_video_picker.change(
                            fn=video_names.get,
                            inputs=sample_video_picker,
                            outputs=input_video,
                        )
                with gr.Column(scale=1):
                    gr.Markdown("#### Live Annotated Frames")
                    live_video_frame = gr.Image(
                        label="Detections update as each sampled frame is analyzed", interactive=False, height=320
                    )
                    video_download = gr.File(
                        label="Download Annotated Video (MP4 -- for offline playback, e.g. VLC; most browsers can't preview this codec inline)"
                    )
                    video_report = gr.HTML(
                        value="<div style='padding: 24px; background: #161b22; border: 1px solid #30363d; border-radius: 8px; text-align: center; color: #8b949e;'>Select or upload a video sample and click Execute Video Classification to run inference.</div>"
                    )

            video_detect_btn.click(
                fn=analyze_video,
                inputs=input_video,
                outputs=[live_video_frame, video_report, video_download],
                concurrency_limit=1,
            )

        with gr.Tab("System Overview"):
            gr.HTML(value=_build_system_info_html())

    gr.HTML("""
    <div style='margin-top: 24px; padding-top: 16px; border-top: 1px solid #21262d; text-align: center; color: #6e7681; font-size: 13px; font-family: -apple-system, sans-serif;'>
        Marine AI System &bull; Embedded Zooplankton Classification &bull; Built by Team CodeFather (SIH 2025)
    </div>
    """)


# ==============================
# LAUNCH APP
# ==============================
def _parse_bool_env(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def _parse_auth_env():
    raw = os.environ.get("GRADIO_AUTH", "").strip()
    if not raw:
        return None
    pairs = [tuple(p.split(":", 1)) for p in raw.split(",") if ":" in p]
    return pairs or None


if __name__ == "__main__":
    share = _parse_bool_env(os.environ.get("GRADIO_SHARE", "false"))
    server_name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    server_port_raw = os.environ.get("GRADIO_SERVER_PORT", "").strip()
    server_port = int(server_port_raw) if server_port_raw.isdigit() else None
    auth = _parse_auth_env()

    if server_name == "0.0.0.0" and not auth and not share:
        print(
            "WARNING: listening on all interfaces with no authentication. "
            "Set GRADIO_AUTH='user:password' to require a login."
        )

    print(f"Starting Marine AI Gradio app on {server_name}:{server_port or 'auto'} ...")
    try:
        demo.launch(server_name=server_name, server_port=server_port, share=share, show_api=False, auth=auth)
    except (OSError, RuntimeError, ValueError) as e:
        print(f"Error launching app: {e}")
        print("Retrying with default settings...")
        demo.close()
        demo.launch(share=share, auth=auth)
    finally:
        camera.release_stream()

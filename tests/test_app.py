import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app


class TestModelLoading(unittest.TestCase):
    """Guards against ISSUE-001 (hardcoded D:\\ path) regressing."""

    def test_model_path_is_relative_to_script(self):
        base_dir = os.path.dirname(os.path.abspath(app.__file__))
        models_dir = os.path.join(base_dir, "models")
        self.assertTrue(
            os.path.normcase(app.MODEL_PATH).startswith(os.path.normcase(models_dir)),
            msg=f"{app.MODEL_PATH} is not under {models_dir}",
        )

    def test_model_loaded_successfully(self):
        self.assertIsNotNone(app.model)


class TestBackendResolution(unittest.TestCase):
    """The compiled-backend selector must not silently pick the wrong artifact."""

    def test_backend_detection_by_path(self):
        from pathlib import Path

        cases = {
            "models/best.pt": "pytorch",
            "models/best_ncnn_model": "ncnn",
            "models/best_saved_model/best_int8.tflite": "tflite",
            "models/best.onnx": "onnx",
            "models/best.bin": "unknown",
        }
        for path, expected in cases.items():
            self.assertEqual(app._detect_backend(Path(path)), expected, msg=path)

    def test_active_backend_is_known(self):
        self.assertIn(app.MODEL_BACKEND, ("pytorch", "ncnn", "tflite", "onnx"))

    def test_imgsz_is_a_multiple_of_32(self):
        """YOLO strides are 8/16/32; a non-multiple silently gets rounded up."""
        self.assertEqual(app.ACTIVE_IMGSZ % 32, 0)

    def test_falls_back_to_best_pt_without_a_manifest(self):
        """No exports.json (the pre-optimization state) must still boot."""
        from unittest.mock import patch

        with patch.object(app, "_load_export_manifest", return_value={}):
            path, backend, imgsz, names = app.resolve_model()
        self.assertEqual(os.path.basename(str(path)), "best.pt")
        self.assertEqual(backend, "pytorch")
        self.assertEqual(imgsz, app.YOLO_IMGSZ)
        self.assertIsNone(names)

    def test_manifest_imgsz_overrides_env_for_compiled_backends(self):
        """A compiled model's frozen input size must win over YOLO_IMGSZ."""
        from unittest.mock import patch

        manifest = {"ncnn": {"path": "models/best_ncnn_model", "imgsz": 416, "names": {"0": "shrimp"}}}
        with (
            patch.object(app, "_load_export_manifest", return_value=manifest),
            patch.object(app.Path, "exists", lambda self: True),
        ):
            _, backend, imgsz, names = app.resolve_model()
        self.assertEqual(backend, "ncnn")
        self.assertEqual(imgsz, 416)
        self.assertEqual(names, {0: "shrimp"})

    def test_candidate_list_ends_with_the_pytorch_fallback(self):
        from unittest.mock import patch

        manifest = {"ncnn": {"path": "models/best_ncnn_model", "imgsz": 416}}
        with (
            patch.object(app, "_load_export_manifest", return_value=manifest),
            patch.object(app.Path, "exists", lambda self: True),
        ):
            candidates = app.candidate_models()
        self.assertEqual([c[1] for c in candidates], ["ncnn", "pytorch"])

    def test_explicit_model_path_yields_exactly_one_candidate(self):
        """An operator who names a file means it -- no silent substitution."""
        from unittest.mock import patch

        with (
            patch.dict(os.environ, {"MODEL_PATH": "models/best.onnx"}),
            patch.object(app, "_load_export_manifest", return_value={}),
        ):
            candidates = app.candidate_models()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][1], "onnx")

    def test_generic_class_names_are_detected(self):
        self.assertTrue(app._names_look_generic({0: "class0", 1: "class1"}))
        self.assertFalse(app._names_look_generic({0: "shrimp", 1: "class1"}))
        self.assertFalse(app._names_look_generic({}))


class TestBackendProbe(unittest.TestCase):
    """A native backend that segfaults must degrade to the next one, not crash
    the service -- no try/except can catch SIGSEGV in-process."""

    def test_probe_reports_native_crash_from_a_signal(self):
        from unittest.mock import MagicMock, patch

        with patch("subprocess.run", return_value=MagicMock(returncode=-11, stderr=b"")):
            ok, detail = app.probe_backend(app.Path("models/best_ncnn_model"), 416)
        self.assertFalse(ok)
        self.assertIn("signal 11", detail)

    def test_probe_reports_success(self):
        from unittest.mock import MagicMock, patch

        with patch("subprocess.run", return_value=MagicMock(returncode=0, stderr=b"")):
            ok, _ = app.probe_backend(app.Path("models/best.onnx"), 416)
        self.assertTrue(ok)

    def test_probe_surfaces_the_last_stderr_line_on_a_clean_exit(self):
        from unittest.mock import MagicMock, patch

        stderr = b"Traceback...\nModuleNotFoundError: No module named 'ncnn'\n"
        with patch("subprocess.run", return_value=MagicMock(returncode=1, stderr=stderr)):
            ok, detail = app.probe_backend(app.Path("models/best_ncnn_model"), 416)
        self.assertFalse(ok)
        self.assertIn("No module named 'ncnn'", detail)

    def test_probe_timeout_is_a_failure_not_an_exception(self):
        import subprocess
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 1)):
            ok, detail = app.probe_backend(app.Path("models/best.onnx"), 416)
        self.assertFalse(ok)
        self.assertIn("timed out", detail)

    def test_probe_that_cannot_spawn_does_not_block_startup(self):
        """If the subprocess itself can't run, assume the backend is fine rather
        than refusing to start."""
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=OSError("no fork")):
            ok, _ = app.probe_backend(app.Path("models/best.onnx"), 416)
        self.assertTrue(ok)

    def test_probe_mode_defaults_to_auto(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BACKEND_PROBE", None)
            self.assertTrue(app._probe_enabled(3))  # something to fall back to
            self.assertFalse(app._probe_enabled(1))  # nowhere to fall back to

    def test_probe_mode_can_be_forced_or_disabled(self):
        from unittest.mock import patch

        with patch.dict(os.environ, {"BACKEND_PROBE": "on"}):
            self.assertTrue(app._probe_enabled(1))
        with patch.dict(os.environ, {"BACKEND_PROBE": "off"}):
            self.assertFalse(app._probe_enabled(5))


class TestThreadEnvironment(unittest.TestCase):
    """Every native thread pool has to be capped, not just torch's and OpenCV's."""

    def test_all_thread_env_vars_are_set(self):
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            self.assertEqual(os.environ.get(var), str(app.CPU_THREADS), msg=var)


class TestPiCpuTuning(unittest.TestCase):
    """CPU-bound-hardware tuning: thread caps must actually be applied, and
    ultralytics' auto-pip-install must stay disabled (it silently broke this
    dev environment once by installing an unrelated GPU package)."""

    def test_torch_and_cv2_thread_pools_are_capped(self):
        import cv2
        import torch

        torch.set_num_threads(app.CPU_THREADS)
        cv2.setNumThreads(app.CPU_THREADS)
        self.assertEqual(torch.get_num_threads(), app.CPU_THREADS)
        self.assertEqual(cv2.getNumThreads(), app.CPU_THREADS)
        self.assertLessEqual(app.CPU_THREADS, 4)

    def test_yolo_autoinstall_disabled(self):
        from ultralytics.utils import AUTOINSTALL

        self.assertFalse(AUTOINSTALL)

    def test_gradio_analytics_disabled(self):
        self.assertEqual(os.environ.get("GRADIO_ANALYTICS_ENABLED"), "False")


class TestDetectZooplankton(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_img_dir = os.path.join(os.path.dirname(os.path.abspath(app.__file__)), "test_img")
        sample_files = sorted(f for f in os.listdir(test_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        cls.sample_path = os.path.join(test_img_dir, sample_files[0])

    def test_none_input_returns_prompt_message(self):
        original, annotated, report = app.detect_zooplankton(None)
        self.assertIsNone(original)
        self.assertIsNone(annotated)
        self.assertIn("upload an image", report.lower())

    def test_runs_on_sample_image_and_returns_matching_shapes(self):
        from PIL import Image

        img = Image.open(self.sample_path)
        original, annotated, report = app.detect_zooplankton(img)
        self.assertIsNotNone(original)
        self.assertIsNotNone(annotated)
        self.assertEqual(original.shape, annotated.shape)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 0)

    def test_large_image_is_downscaled_below_cap(self):
        """ISSUE-005: a very large upload must not be processed at full resolution."""
        from PIL import Image

        oversized = Image.new("RGB", (6000, 4000), color=(10, 20, 30))
        original, annotated, _ = app.detect_zooplankton(oversized)
        self.assertLessEqual(max(original.shape[0], original.shape[1]), app.MAX_DIMENSION)
        self.assertEqual(original.shape, annotated.shape)

    def test_blank_image_returns_no_detections_report(self):
        """AUDIT ISSUE-006: the zero-detections HTML branch had no coverage."""
        from PIL import Image

        blank = Image.new("RGB", (64, 64), color=(10, 20, 30))
        _, _, report = app.detect_zooplankton(blank)
        self.assertIn("No Marine Life Detected", report)

    def test_model_none_returns_friendly_error_without_crashing(self):
        """AUDIT ISSUE-006: the model-not-loaded degraded-mode branch had no coverage."""
        from PIL import Image

        original_model = app.model
        app.model = None
        try:
            img = Image.new("RGB", (64, 64), color=(10, 20, 30))
            original, annotated, report = app.detect_zooplankton(img)
        finally:
            app.model = original_model

        self.assertIs(original, img)
        self.assertIs(annotated, img)
        self.assertIn("Model Not Found", report)

    def test_inference_exception_returns_friendly_error_without_crashing(self):
        """AUDIT ISSUE-005: exceptions during predict()/annotation must not propagate raw."""
        from unittest.mock import patch

        from PIL import Image

        img = Image.new("RGB", (64, 64), color=(10, 20, 30))
        with patch.object(app.model, "predict", side_effect=RuntimeError("boom")):
            original, annotated, report = app.detect_zooplankton(img)

        # exif_transpose() (run before the mocked predict() call) always
        # returns a copy, so identity isn't preserved here the way it is
        # in the model-None path above -- compare size instead.
        self.assertEqual(original.size, img.size)
        self.assertEqual(annotated.size, img.size)
        self.assertIn("Analysis Failed", report)


class TestHtmlEscaping(unittest.TestCase):
    """ISSUE-015: class names must be escaped before landing in the HTML report."""

    def test_html_escape_neutralizes_markup(self):
        unsafe = "<img src=x onerror=alert(1)>"
        self.assertNotIn("<img", app.html.escape(unsafe))


class TestShareFlagParsing(unittest.TestCase):
    """ISSUE-003: `share` must default to False and only opt in on explicit truthy values."""

    def test_truthy_values(self):
        for value in ("1", "true", "TRUE", "True", "yes", "Yes"):
            self.assertTrue(app._parse_bool_env(value), msg=value)

    def test_falsy_values(self):
        for value in ("0", "false", "False", "no", "", "anything-else"):
            self.assertFalse(app._parse_bool_env(value), msg=value)

    def test_default_is_false_when_env_var_unset(self):
        os.environ.pop("GRADIO_SHARE", None)
        self.assertFalse(app._parse_bool_env(os.environ.get("GRADIO_SHARE", "false")))


class TestCameraStream(unittest.TestCase):
    """The capture thread is exercised against a fake VideoCapture: no hardware
    needed, and CI would otherwise never cover the reconnect/downscale paths."""

    @staticmethod
    def _fake_capture(frames, fail_after=None):
        """Build a stand-in for cv2.VideoCapture that yields `frames` forever."""
        import itertools
        import threading

        class FakeCapture:
            def __init__(self, *args, **kwargs):
                self.released = False
                self.props = {}
                self._frames = itertools.cycle(frames)
                self._count = 0
                self._lock = threading.Lock()

            def isOpened(self):
                return not self.released

            def set(self, prop, value):
                self.props[prop] = value
                return True

            def read(self):
                with self._lock:
                    self._count += 1
                    if fail_after is not None and self._count > fail_after:
                        return False, None
                return True, next(self._frames).copy()

            def release(self):
                self.released = True

        return FakeCapture

    def test_downscale_fits_box_and_never_upscales(self):
        import numpy as np

        import camera

        big = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = camera.downscale(big, 1280, 720)
        self.assertEqual(out.shape[:2], (720, 1280))

        small = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIs(camera.downscale(small, 1280, 720), small)

    def test_parse_device_accepts_index_or_path(self):
        import camera

        self.assertEqual(camera.parse_device("0"), 0)
        self.assertEqual(camera.parse_device(2), 2)
        self.assertEqual(camera.parse_device("/dev/video0"), "/dev/video0")

    def test_reader_thread_publishes_downscaled_frames(self):
        from unittest.mock import patch

        import numpy as np

        import camera

        frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
        with patch.object(camera.cv2, "VideoCapture", self._fake_capture([frame])):
            stream = camera.CameraStream(max_width=640, max_height=360).start()
            try:
                got, seq, ts = stream.wait_for_frame(timeout=5.0)
                self.assertIsNotNone(got)
                self.assertEqual(got.shape[:2], (360, 640))
                self.assertGreater(seq, 0)
                self.assertGreater(ts, 0)
            finally:
                stream.stop()
        self.assertFalse(stream.running)

    def test_low_latency_capture_properties_are_requested(self):
        """MJPG + a 1-frame driver queue are what keep capture latency bounded."""
        from unittest.mock import patch

        import cv2
        import numpy as np

        import camera

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        fake_cls = self._fake_capture([frame])
        with patch.object(camera.cv2, "VideoCapture", fake_cls):
            stream = camera.CameraStream(width=1920, height=1080, fourcc="MJPG").start()
            try:
                stream.wait_for_frame(timeout=5.0)
                props = stream._cap.props
            finally:
                stream.stop()
        self.assertEqual(props[cv2.CAP_PROP_BUFFERSIZE], 1)
        self.assertEqual(props[cv2.CAP_PROP_FRAME_WIDTH], 1920)
        self.assertEqual(props[cv2.CAP_PROP_FOURCC], cv2.VideoWriter_fourcc(*"MJPG"))

    def test_read_is_safe_before_any_frame_arrives(self):
        import camera

        stream = camera.CameraStream()
        frame, seq, ts = stream.read()
        self.assertIsNone(frame)
        self.assertEqual((seq, ts), (0, 0.0))

    def test_open_failure_raises_camera_error(self):
        from unittest.mock import patch

        import camera

        class DeadCapture:
            def __init__(self, *args, **kwargs):
                pass

            def isOpened(self):
                return False

            def release(self):
                pass

        with patch.object(camera.cv2, "VideoCapture", DeadCapture), self.assertRaises(camera.CameraError):
            camera.CameraStream().start()

    def test_reader_reconnects_after_a_read_failure(self):
        """An unplugged microscope must recover without a service restart."""
        from unittest.mock import patch

        import numpy as np

        import camera

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(camera.cv2, "VideoCapture", self._fake_capture([frame], fail_after=3)):
            stream = camera.CameraStream(reconnect=True).start()
            try:
                self.assertIsNotNone(stream.wait_for_frame(timeout=5.0)[0])
                deadline = time.monotonic() + 5.0
                while stream.frames_dropped == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertGreater(stream.frames_dropped, 0)
                # A fresh capture object is opened after the failure, so frames
                # keep flowing rather than the thread exiting.
                self.assertIsNotNone(stream.wait_for_frame(timeout=5.0)[0])
            finally:
                stream.stop()


class TestCameraHandlers(unittest.TestCase):
    """Gradio callbacks must degrade to an error card, never raise into the UI."""

    def tearDown(self):
        import camera

        camera.release_stream()

    def test_capture_without_a_running_camera_returns_error_card(self):
        before, after, report = app.camera_capture_and_analyze()
        self.assertIsNone(before)
        self.assertIsNone(after)
        self.assertIn("Camera Not Running", report)

    def test_start_failure_is_reported_not_raised(self):
        from unittest.mock import patch

        import camera

        with patch.object(camera, "get_stream", side_effect=camera.CameraError("no device")):
            preview, status = app.camera_start()
        self.assertIsNone(preview)
        self.assertIn("no device", status)

    def test_preview_without_a_stream_is_a_noop_update(self):
        self.assertIsNotNone(app.camera_preview())

    def test_analyze_rgb_shared_by_upload_and_camera_paths(self):
        from unittest.mock import MagicMock, patch

        import numpy as np

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        # Mock predict to return a result object with no boxes
        mock_result = MagicMock()
        mock_result.boxes = []
        with patch.object(app.model, "predict", return_value=[mock_result]):
            original, annotated, report = app.analyze_rgb(frame)
        self.assertEqual(original.shape, annotated.shape)
        self.assertIn("No Marine Life Detected", report)


if __name__ == "__main__":
    unittest.main()

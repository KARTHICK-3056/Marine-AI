# USB microscope capture for Marine AI.
#
# A synchronous `cap.read()` on the Gradio request thread blocks for however long
# the microscope's exposure + USB transfer + MJPG decode takes (commonly 60-150 ms
# at 1080p on a Pi 5, and much worse if the sensor is on a USB 2.0 port). Worse,
# V4L2 hands back whatever is next in the driver's ring buffer, so a request that
# arrives after an idle period gets a stale frame from seconds ago.
#
# This module runs the capture loop on its own thread and keeps exactly one frame
# (double-buffered under a lock). Readers always get the newest complete frame and
# never block on the camera; the driver queue never backs up.

from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
from typing_extensions import Self

# Defaults are env-overridable so the systemd unit can retune a deployed Pi
# without a code change.
DEFAULT_DEVICE = os.environ.get("CAMERA_DEVICE", "0")
DEFAULT_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1920"))
DEFAULT_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "1080"))
DEFAULT_FPS = int(os.environ.get("CAMERA_FPS", "30"))
DEFAULT_FOURCC = os.environ.get("CAMERA_FOURCC", "MJPG")

# Microscopes routinely stream 1080p or 4K. Full-resolution frames cost far more
# in colour conversion, copying and letterboxing than they buy in detection
# quality once YOLO downsamples to imgsz anyway, so the capture thread shrinks
# every frame to fit this box before it is ever published to a reader.
DEFAULT_MAX_WIDTH = int(os.environ.get("CAMERA_MAX_WIDTH", "1280"))
DEFAULT_MAX_HEIGHT = int(os.environ.get("CAMERA_MAX_HEIGHT", "720"))


def parse_device(device: str | int) -> str | int:
    """Accept either a V4L2 index ("0") or a device path ("/dev/video0")."""
    if isinstance(device, int):
        return device
    text = str(device).strip()
    return int(text) if text.isdigit() else text


def preferred_backend() -> int:
    """V4L2 on Linux (the deployment target); let OpenCV choose elsewhere.

    CAP_V4L2 is what makes the FOURCC/resolution/buffersize properties below
    actually stick, and it is the path that reaches the Pi's hardware-assisted
    MJPG decode instead of a userspace fallback.
    """
    if hasattr(cv2, "CAP_V4L2") and os.name == "posix":
        return cv2.CAP_V4L2
    return getattr(cv2, "CAP_ANY", 0)


def downscale(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Shrink to fit the box, preserving aspect ratio. Never upscales.

    INTER_AREA is both the cheapest and the highest-quality choice when
    shrinking -- it is a box filter over the source pixels, so it does not
    alias thin structures (setae, antennae) into nothing the way INTER_LINEAR
    does at large downscale factors.
    """
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return frame
    scale = min(1.0, max_width / w, max_height / h)
    if scale >= 1.0:
        return frame
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


class CameraError(RuntimeError):
    """Raised when the capture device cannot be opened."""


class CameraStream:
    """Background-thread reader for a USB microscope, with a 1-deep frame buffer.

    Usage:
        cam = CameraStream().start()
        frame, seq, ts = cam.read()      # newest frame, or (None, 0, 0.0)
        cam.stop()
    """

    def __init__(
        self,
        device: str | int = DEFAULT_DEVICE,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        fps: int = DEFAULT_FPS,
        fourcc: str = DEFAULT_FOURCC,
        max_width: int = DEFAULT_MAX_WIDTH,
        max_height: int = DEFAULT_MAX_HEIGHT,
        downscale_in_thread: bool = True,
        reconnect: bool = True,
    ):
        self.device = parse_device(device)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.max_width = max_width
        self.max_height = max_height
        self.downscale_in_thread = downscale_in_thread
        self.reconnect = reconnect

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._new_frame = threading.Event()
        self._stop = threading.Event()

        # The double buffer: the capture thread only ever writes `_back`, then
        # swaps it into `_front` under the lock. Readers copy nothing and hold
        # the lock for the duration of a pointer swap, so a slow reader can
        # never stall the capture loop.
        self._front: np.ndarray | None = None
        self._seq = 0
        self._timestamp = 0.0

        self.frames_captured = 0
        self.frames_dropped = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _open(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.device, preferred_backend())
        if not cap.isOpened():
            cap.release()
            raise CameraError(
                f"Could not open camera {self.device!r}. Check that the microscope is plugged in "
                f"(`v4l2-ctl --list-devices`) and that the service user is in the 'video' group."
            )

        # FOURCC must be set before the frame size: switching the pixel format
        # resets the driver's negotiated resolution on most UVC devices.
        # MJPG is what lets a UVC microscope stream 1080p30 over USB 2.0 at all --
        # raw YUYV at that size needs more bandwidth than the bus provides.
        if self.fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.fps:
            cap.set(cv2.CAP_PROP_FPS, self.fps)
        # A 1-frame driver queue keeps latency low; without it V4L2 buffers
        # several frames and `read()` returns increasingly stale images.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self) -> CameraStream:
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._cap = self._open()  # opened here so start() raises on a bad device
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        cap, self._cap = self._cap, None
        if cap is not None:
            cap.release()
        with self._lock:
            self._front = None
        self._new_frame.set()  # release anyone parked in wait_for_frame()

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Capture loop
    # ------------------------------------------------------------------
    def _loop(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            cap = self._cap
            if cap is None:
                if not self.reconnect:
                    return
                try:
                    self._cap = self._open()
                    self.last_error = None
                    backoff = 0.5
                except CameraError as e:
                    self.last_error = str(e)
                    # Cap the backoff so an unplugged microscope retries about
                    # twice a second at worst instead of spinning the CPU.
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 5.0)
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self.frames_dropped += 1
                self.last_error = "camera read failed (device unplugged or stream stalled)"
                cap.release()
                self._cap = None
                if not self.reconnect:
                    return
                continue

            if self.downscale_in_thread:
                frame = downscale(frame, self.max_width, self.max_height)

            with self._lock:
                self._front = frame
                self._seq += 1
                self._timestamp = time.time()
            self.frames_captured += 1
            self._new_frame.set()

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------
    def read(self) -> tuple[np.ndarray | None, int, float]:
        """Return (frame_bgr, sequence, capture_time). Never blocks on the camera."""
        with self._lock:
            frame, seq, ts = self._front, self._seq, self._timestamp
        if frame is None:
            return None, 0, 0.0
        if not self.downscale_in_thread:
            frame = downscale(frame, self.max_width, self.max_height)
        return frame, seq, ts

    def read_rgb(self) -> tuple[np.ndarray | None, int, float]:
        """Same as read(), converted to RGB for Gradio/PIL/YOLO consumers."""
        frame, seq, ts = self.read()
        if frame is None:
            return None, seq, ts
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), seq, ts

    def wait_for_frame(self, timeout: float = 5.0) -> tuple[np.ndarray | None, int, float]:
        """Block until at least one frame has been captured (or `timeout` elapses)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            frame, seq, ts = self.read()
            if frame is not None:
                return frame, seq, ts
            self._new_frame.wait(timeout=min(0.25, max(0.0, deadline - time.monotonic())))
            self._new_frame.clear()
        return None, 0, 0.0

    def status(self) -> dict:
        frame, seq, ts = self.read()
        return {
            "running": self.running,
            "device": self.device,
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}" if frame is not None else "-",
            "frames_captured": self.frames_captured,
            "frames_dropped": self.frames_dropped,
            "sequence": seq,
            "age_s": round(time.time() - ts, 2) if ts else None,
            "last_error": self.last_error,
        }


_stream: CameraStream | None = None
_stream_lock = threading.Lock()


def get_stream(**kwargs) -> CameraStream:
    """Process-wide singleton. A USB camera cannot be opened twice, and Gradio
    serves requests from a thread pool, so the stream has to be shared."""
    global _stream
    with _stream_lock:
        if _stream is None or not _stream.running:
            if _stream is not None:
                _stream.stop()
            _stream = CameraStream(**kwargs).start()
        return _stream


def release_stream() -> None:
    global _stream
    with _stream_lock:
        if _stream is not None:
            _stream.stop()
            _stream = None


def is_active() -> bool:
    with _stream_lock:
        return _stream is not None and _stream.running


def current_stream() -> CameraStream | None:
    with _stream_lock:
        return _stream

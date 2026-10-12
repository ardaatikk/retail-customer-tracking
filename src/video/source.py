from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Thread
from typing import Union
import math
import os
import time

import cv2


VideoInput = Union[str, Path, int]


@dataclass(frozen=True)
class VideoSourceInfo:
    fps: float
    width: int
    height: int
    is_live: bool


class VideoSource:
    """Sequential file reader or latest-frame live reader.

    Live read timeout returns (False, None); callers may retry. Only RTSP
    sources reconnect. Capture access/release belongs to the worker thread.
    """

    def __init__(
        self,
        source: VideoInput,
        fallback_fps: float = 25.0,
        *,
        read_timeout: float = 2.0,
        reconnect_interval: float = 1.0,
        open_timeout: float = 5.0,
        capture_read_timeout: float = 2.0,
    ) -> None:
        for name, value in (
            ("read_timeout", read_timeout),
            ("reconnect_interval", reconnect_interval),
            ("open_timeout", open_timeout),
            ("capture_read_timeout", capture_read_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.source = source
        self.is_live = self._is_live_source(source)
        self._is_rtsp = str(source).lower().startswith(("rtsp://", "rtsps://"))
        self._fallback_fps = fallback_fps
        self.read_timeout = read_timeout
        self.reconnect_interval = reconnect_interval
        self.open_timeout = open_timeout
        self.capture_read_timeout = capture_read_timeout
        self._condition = Condition()
        self._stop_event = Event()
        self._thread = None
        self._latest_frame = None
        self._frame_id = 0
        self._last_read_id = 0
        self._reader_failed = False
        self._connected = False
        self.cap = None
        self.info = VideoSourceInfo(fallback_fps, 0, 0, self.is_live)

        if self._is_rtsp:
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = self._open_capture()
        if self.cap is not None:
            self._update_info()
            self._connected = True
        elif not self._is_rtsp:
            raise RuntimeError(f"Could not open video source: {source}")

        if self.is_live:
            self._thread = Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    @staticmethod
    def _is_live_source(source: VideoInput) -> bool:
        return isinstance(source, int) or str(source).lower().startswith(
            ("rtsp://", "rtsps://", "http://", "https://")
        )

    @property
    def connected(self) -> bool:
        with self._condition:
            return self._connected and not self._stop_event.is_set()

    def _open_capture(self):
        source = str(self.source) if isinstance(self.source, Path) else self.source
        cap = None
        try:
            if self._is_rtsp:
                # These properties are open-only; set() after open is ineffective.
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, max(1, int(self.open_timeout * 1000)),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    max(1, int(self.capture_read_timeout * 1000)),
                ])
            else:
                cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap
        except cv2.error:
            if not self._is_rtsp:
                raise
        if cap is not None:
            cap.release()
        return None

    def _update_info(self) -> None:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.info = VideoSourceInfo(
            fps=fps if math.isfinite(fps) and fps > 0 else self._fallback_fps,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            is_live=self.is_live,
        )

    def _invalidate_frame(self) -> None:
        # Called under the condition lock. IDs remain monotonic across reconnects.
        self._connected = False
        self._latest_frame = None
        self._last_read_id = self._frame_id
        self._condition.notify_all()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.cap is None:
                    if self._stop_event.wait(self.reconnect_interval):
                        break
                    self.cap = self._open_capture()
                    if self.cap is None:
                        continue
                    self._update_info()
                    # Connection becomes usable only after the first decoded frame.
                try:
                    ret, frame = self.cap.read()
                except cv2.error:
                    ret, frame = False, None
                with self._condition:
                    if self._stop_event.is_set():
                        break
                    if ret and frame is not None:
                        self._latest_frame = frame
                        self._frame_id += 1
                        self._connected = True
                        self._condition.notify_all()
                        continue
                    self._invalidate_frame()
                    if not self._is_rtsp:
                        self._reader_failed = True
                self.cap.release()
                self.cap = None
                if not self._is_rtsp:
                    break
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            with self._condition:
                self._reader_failed = True
                self._invalidate_frame()

    def read(self, timeout: float = None):
        if not self.is_live:
            return self.cap.read()
        timeout = self.read_timeout if timeout is None else timeout
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be finite and non-negative")
        deadline = time.monotonic() + timeout
        with self._condition:
            while not self._stop_event.is_set():
                if self._latest_frame is not None and self._frame_id > self._last_read_id:
                    self._last_read_id = self._frame_id
                    return True, self._latest_frame
                if self._reader_failed:
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._condition.wait(remaining)
        return False, None

    def release(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._invalidate_frame()
        if self._thread is not None:
            # Do not release capture from another thread during native read/open.
            self._thread.join(timeout=max(self.open_timeout, self.capture_read_timeout) + 1)
            if self._thread.is_alive():
                raise RuntimeError("Capture worker did not stop within its backend timeout")
        elif self.cap is not None:
            self.cap.release()

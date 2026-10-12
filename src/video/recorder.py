"""Independent RTSP stream-copy recorder; no decoding or inference in this process."""
from pathlib import Path
import shutil
import signal
import subprocess


class StreamRecorder:
    def __init__(self, source: str, output_path: Path) -> None:
        if not str(source).lower().startswith(('rtsp://', 'rtsps://')):
            raise ValueError('Stream recording requires an RTSP source')
        executable = shutil.which('ffmpeg')
        if executable is None:
            raise RuntimeError('FFmpeg is required for stream recording')
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # MKV tolerates interrupted recording better than a conventional MP4.
        # Keep stderr private: backend errors can contain camera credentials.
        self.process = subprocess.Popen([
            executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
            '-rtsp_transport', 'tcp', '-timeout', '5000000',
            '-i', str(source), '-map', '0:v:0', '-an', '-c:v', 'copy',
            '-f', 'matroska', str(self.output_path),
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL)

    @property
    def running(self) -> bool:
        return self.process.poll() is None

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=7)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)

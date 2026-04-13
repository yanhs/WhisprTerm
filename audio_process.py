"""Audio recording using sd.rec() — purely C-level, no GIL needed."""
import numpy as np
import sounddevice as sd
import time as _time
import config

MAX_SECONDS = 120


class AudioProcess:
    def __init__(self):
        self._buffer = None
        self._start_time = 0
        self._recording = False

    def init_stream(self):
        pass

    def start(self):
        self._buffer = sd.rec(
            frames=config.SAMPLE_RATE * MAX_SECONDS,
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocking=False,
        )
        self._start_time = _time.monotonic()
        self._recording = True

    def get_audio_so_far(self):
        if not self._recording or self._buffer is None:
            return np.array([], dtype="float32")
        elapsed = _time.monotonic() - self._start_time
        pos = min(int(elapsed * config.SAMPLE_RATE), len(self._buffer))
        if pos <= 0:
            return np.array([], dtype="float32")
        return self._buffer[:pos, 0].copy()

    def stop(self):
        self._recording = False
        sd.stop()
        if self._buffer is None:
            return np.array([], dtype="float32")
        elapsed = _time.monotonic() - self._start_time
        pos = min(int(elapsed * config.SAMPLE_RATE), len(self._buffer))
        result = self._buffer[:pos, 0].copy() if pos > 0 else np.array([], dtype="float32")
        self._buffer = None
        return result

    def close(self):
        self._recording = False
        sd.stop()

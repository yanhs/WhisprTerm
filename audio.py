import numpy as np
import sounddevice as sd
import threading
import config


class AudioRecorder:
    def __init__(self):
        self._chunks = []
        self._stream = None
        self._lock = threading.Lock()

    def _callback(self, indata, frames, time_info, status):
        with self._lock:
            self._chunks.append(indata.copy())

    def start(self):
        with self._lock:
            self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.array([], dtype="float32")
            audio = np.concatenate(self._chunks, axis=0).flatten()
            self._chunks.clear()
            return audio

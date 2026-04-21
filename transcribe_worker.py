"""Transcription worker in a separate process.
Only transcribes — recording is done in the main process for instant start."""
import multiprocessing
import numpy as np


def _worker_loop(model_name, device, audio_conn, result_conn, ready_event, stop_event):
    import os, sys, glob
    for pattern in [
        os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin"),
        os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python*",
                     "Lib", "site-packages", "nvidia", "*", "bin"),
    ]:
        for p in glob.glob(pattern):
            if os.path.isdir(p):
                os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
                try:
                    os.add_dll_directory(p)
                except (OSError, AttributeError):
                    pass

    from faster_whisper import WhisperModel

    compute = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_name, device=device, compute_type=compute)
    ready_event.set()

    while not stop_event.is_set():
        try:
            if not audio_conn.poll(timeout=0.1):
                continue
            audio_bytes = audio_conn.recv()
        except (EOFError, OSError):
            break

        if audio_bytes is None:
            break

        audio = np.frombuffer(audio_bytes, dtype="float32")
        # Normalize quiet audio
        peak = np.max(np.abs(audio))
        if peak > 0.001:
            audio = audio * (0.9 / peak)

        try:
            segments, info = model.transcribe(
                audio, language=None, beam_size=5,
                vad_filter=True, vad_parameters={"threshold": 0.15},
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
        except Exception:
            text = ""

        try:
            if text:
                result_conn.send(("final", "", text))
            else:
                result_conn.send(("done", "", ""))
        except (EOFError, OSError):
            pass


class TranscribeWorker:
    def __init__(self, model_name, device):
        self._model_name = model_name
        self._device = device
        self._process = None
        self._audio_send = None
        self._result_recv = None

    def start(self):
        audio_recv, self._audio_send = multiprocessing.Pipe(duplex=False)
        self._result_recv, result_send = multiprocessing.Pipe(duplex=False)
        ready = multiprocessing.Event()
        stop = multiprocessing.Event()
        self._stop = stop
        self._process = multiprocessing.Process(
            target=_worker_loop,
            args=(self._model_name, self._device, audio_recv, result_send, ready, stop),
            daemon=True,
        )
        self._process.start()
        ready.wait(timeout=120)

    def transcribe_audio(self, audio_data):
        """Send audio numpy array to worker for transcription."""
        self._audio_send.send(audio_data.tobytes())

    def get_result(self, timeout=30):
        if self._result_recv.poll(timeout=timeout):
            return self._result_recv.recv()
        return None

    def stop(self):
        try:
            self._audio_send.send(None)
        except Exception:
            pass
        if self._stop:
            self._stop.set()
        if self._process:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()

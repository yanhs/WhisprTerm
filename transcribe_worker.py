"""Transcription worker in a separate process.
Always-on recording — no audio is lost when hotkey is pressed."""
import multiprocessing
import numpy as np
import time


def _worker_loop(model_name, device, cmd_conn, result_conn, ready_event, stop_event):
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

    import sounddevice as sd
    from faster_whisper import WhisperModel

    compute = "int8" if device == "cpu" else "float16"
    model = WhisperModel(model_name, device=device, compute_type=compute)

    SAMPLE_RATE = 16000
    BUF_SECONDS = 300  # 5 min ring buffer
    PRE_ROLL = 0.5  # seconds of audio BEFORE hotkey press to capture

    # Start always-on recording immediately
    buf = sd.rec(frames=SAMPLE_RATE * BUF_SECONDS, samplerate=SAMPLE_RATE,
                 channels=1, dtype="float32", blocking=False)
    buf_start = time.monotonic()

    ready_event.set()

    while not stop_event.is_set():
        try:
            if not cmd_conn.poll(timeout=0.1):
                continue
            cmd = cmd_conn.recv()
        except (EOFError, OSError):
            break

        if cmd == "start":
            # Mark start position with pre-roll (capture audio from 0.5s ago)
            now = time.monotonic()
            rec_start = now - PRE_ROLL
            start_sample = max(0, int((rec_start - buf_start) * SAMPLE_RATE))

            # Wait for stop
            while True:
                if cmd_conn.poll(timeout=0.1):
                    try:
                        cmd2 = cmd_conn.recv()
                    except (EOFError, OSError):
                        break
                    if cmd2 == "stop":
                        break

            # Get audio from start_sample to now
            end_time = time.monotonic()
            end_sample = min(int((end_time - buf_start) * SAMPLE_RATE), len(buf))

            if end_sample - start_sample >= SAMPLE_RATE * 0.3:
                audio = buf[start_sample:end_sample, 0].copy()
                try:
                    segments, info = model.transcribe(audio, language=None, beam_size=5, vad_filter=True)
                    text = " ".join(seg.text.strip() for seg in segments).strip()
                except Exception:
                    text = ""
                if text:
                    try:
                        result_conn.send(("final", "", text))
                    except (EOFError, OSError):
                        pass
                else:
                    try:
                        result_conn.send(("done", "", ""))
                    except (EOFError, OSError):
                        pass
            else:
                try:
                    result_conn.send(("done", "", ""))
                except (EOFError, OSError):
                    pass

            # If buffer is getting full (>4 min), restart it
            elapsed_total = time.monotonic() - buf_start
            if elapsed_total > BUF_SECONDS - 30:
                sd.stop()
                buf = sd.rec(frames=SAMPLE_RATE * BUF_SECONDS, samplerate=SAMPLE_RATE,
                             channels=1, dtype="float32", blocking=False)
                buf_start = time.monotonic()

        elif cmd == "quit":
            break

    sd.stop()


class TranscribeWorker:
    def __init__(self, model_name, device):
        self._model_name = model_name
        self._device = device
        self._process = None
        self._cmd_send = None
        self._result_recv = None

    def start(self):
        cmd_recv, self._cmd_send = multiprocessing.Pipe(duplex=False)
        self._result_recv, result_send = multiprocessing.Pipe(duplex=False)
        ready = multiprocessing.Event()
        stop = multiprocessing.Event()
        self._stop = stop
        self._process = multiprocessing.Process(
            target=_worker_loop,
            args=(self._model_name, self._device, cmd_recv, result_send, ready, stop),
            daemon=True,
        )
        self._process.start()
        ready.wait(timeout=120)

    def start_recording(self):
        self._cmd_send.send("start")

    def stop_recording(self):
        self._cmd_send.send("stop")

    def get_result(self, timeout=30):
        if self._result_recv.poll(timeout=timeout):
            return self._result_recv.recv()
        return None

    def stop(self):
        try:
            self._cmd_send.send("quit")
        except Exception:
            pass
        if self._stop:
            self._stop.set()
        if self._process:
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()

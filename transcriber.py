import os
import sys
import glob

# Add CUDA DLLs to PATH before importing faster_whisper
_cuda_paths = []
for pattern in [
    os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin"),
    os.path.join(os.path.expanduser("~"), "AppData", "Local", "Programs", "Python", "Python*", "Lib", "site-packages", "nvidia", "*", "bin"),
]:
    _cuda_paths.extend(glob.glob(pattern))
for p in _cuda_paths:
    if os.path.isdir(p):
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(p)
        except (OSError, AttributeError):
            pass

from faster_whisper import WhisperModel
import config


class Transcriber:
    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = device or config.WHISPER_DEVICE
        self.compute = "int8" if self.device == "cpu" else "float16"
        self.model = None
        self.load()

    def load(self):
        print(f"Loading Whisper model '{self.model_name}' on {self.device} ({self.compute})...")
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute,
        )
        print("Model loaded.")

    def reload(self, model_name=None, device=None):
        if model_name:
            self.model_name = model_name
        if device:
            self.device = device
            self.compute = "int8" if device == "cpu" else "float16"
        self.load()

    def transcribe(self, audio_data):
        if len(audio_data) < config.SAMPLE_RATE * 0.3:
            return ""
        segments, info = self.model.transcribe(
            audio_data,
            language=config.LANGUAGE,
            beam_size=5,
            vad_filter=True,
        )
        text = " ".join(seg.text.strip() for seg in segments)
        return text.strip()

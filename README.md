# Whispr Term

Voice-to-text for browser terminals (ttyd / xterm.js in Chrome).
Push-to-talk: **hold `Ctrl+Win`, speak, release** — the transcription is pasted into the focused terminal.

## Specs
- Engine: `faster-whisper large-v3` on CUDA (float16), Russian + English auto-detect
- Hotkey: `Ctrl+Win` held = record, release = transcribe + paste (`Ctrl+Shift+V`)
- Survives sleep/wake (LL-hook watchdog) and worker crashes (auto-restart)
- System tray icon (green = ready, red = recording, yellow = processing)

## Run from source
```
pip install -r requirements.txt  # faster-whisper, sounddevice, pyperclip, pystray, Pillow, numpy
pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
python main.py
```

## Build EXE (PyInstaller)
```
pyinstaller --noconfirm WhisprTerm.spec
```
Output: `dist/WhisprTerm/WhisprTerm.exe`

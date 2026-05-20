import sys
import os
import json
import time
import threading
import numpy as np
import pystray
from PIL import Image, ImageDraw
import config

SETTINGS_FILE = os.path.join(os.path.expanduser("~"), ".whispr_term.json")

# --- Win32 ---
import ctypes
from ctypes import wintypes, CFUNCTYPE, POINTER

_user32 = ctypes.windll.user32
_INPUT_KEYBOARD = 1
_KEYEVENTF_KEYUP = 0x0002
_KEYEVENTF_UNICODE = 0x0004
_VK_RETURN = 0x0D
_VK_CONTROL = 0x11
_VK_SHIFT = 0x10
_VK_MENU = 0x12
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C


class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wintypes.DWORD), ("wParamL", wintypes.WORD), ("wParamH", wintypes.WORD)]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


def _key_event(vk=0, scan=0, flags=0):
    inp = _INPUT()
    inp.type = _INPUT_KEYBOARD
    inp.union.ki.wVk = vk
    inp.union.ki.wScan = scan
    inp.union.ki.dwFlags = flags
    _user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(_INPUT))


def _release_all_modifiers():
    for vk in (_VK_CONTROL, _VK_SHIFT, _VK_MENU, _VK_LWIN, _VK_RWIN):
        _key_event(vk=vk, flags=_KEYEVENTF_KEYUP)


def _send_text_to_window(text):
    """Paste via clipboard + Ctrl+Shift+V (works in xterm.js/Chrome terminals)."""
    import pyperclip
    old_clip = ""
    try:
        old_clip = pyperclip.paste()
    except Exception:
        pass
    pyperclip.copy(text)
    time.sleep(0.05)
    _key_event(vk=_VK_CONTROL)
    _key_event(vk=_VK_SHIFT)
    _key_event(vk=0x56)  # V
    _key_event(vk=0x56, flags=_KEYEVENTF_KEYUP)
    _key_event(vk=_VK_SHIFT, flags=_KEYEVENTF_KEYUP)
    _key_event(vk=_VK_CONTROL, flags=_KEYEVENTF_KEYUP)
    time.sleep(0.1)
    try:
        pyperclip.copy(old_clip)
    except Exception:
        pass


def _send_enter_to_window():
    _key_event(vk=_VK_RETURN)
    _key_event(vk=_VK_RETURN, flags=_KEYEVENTF_KEYUP)


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_settings(settings):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)


def make_icon(color):
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=color)
    draw.rounded_rectangle([24, 16, 40, 38], radius=6, fill="white")
    draw.rectangle([30, 38, 34, 46], fill="white")
    draw.arc([20, 30, 44, 50], start=0, end=180, fill="white", width=2)
    return img


ICON_READY = make_icon("#4CAF50")
ICON_RECORDING = make_icon("#F44336")
ICON_PROCESSING = make_icon("#FFC107")
ICON_LOADING = make_icon("#9E9E9E")

# LL keyboard hook types
WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
WM_SYSKEYUP = 0x0105
WM_REINSTALL_HOOK = 0x0400 + 100  # custom WM_USER message for watchdog
VK_LCONTROL = 0xA2
VK_RCONTROL = 0xA3


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD), ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


HOOKPROC = CFUNCTYPE(ctypes.c_long, ctypes.c_int, wintypes.WPARAM, POINTER(KBDLLHOOKSTRUCT))


class WhisprTerm:
    def __init__(self):
        self.settings = load_settings()
        self.model_name = self.settings.get("model", config.WHISPER_MODEL)
        self.device = self.settings.get("device", config.WHISPER_DEVICE)
        self.worker = None
        self.recording = False
        self.model_ready = False
        self.tray = None
        self._rec_start_time = 0
        self._rec_buf = None
        self._keys_down = set()
        self._hook_proc_ref = None  # prevent GC
        self._block_next_win_up = False  # block Win release after recording
        self._hook_handle = 0  # current installed hook
        self._pending_start = threading.Event()
        self._pending_stop = threading.Event()

    def _load_worker_bg(self):
        from transcribe_worker import TranscribeWorker
        self.worker = TranscribeWorker(self.model_name, self.device)
        print("[LOAD] Starting worker (loading model on GPU)...")
        self.worker.start()
        self.model_ready = True
        if self.tray:
            self.tray.icon = ICON_READY
            self.tray.title = "Whispr Term — ready"
        print("[OK] Ready! Hold Ctrl+Win and speak.")

    def _start_recording(self):
        if not self.model_ready or self.recording:
            return
        self.recording = True
        self._rec_start_time = time.monotonic()
        # Start recording IMMEDIATELY in main process (C-thread, no GIL delay)
        import sounddevice as sd
        self._rec_buf = sd.rec(
            frames=config.SAMPLE_RATE * 120,
            samplerate=config.SAMPLE_RATE,
            channels=1, dtype="float32", blocking=False,
        )
        if self.tray:
            self.tray.icon = ICON_RECORDING
            self.tray.title = "Whispr Term — RECORDING"
        print("[REC] Recording...")

    def _stop_recording(self):
        if not self.recording:
            return
        elapsed = time.monotonic() - self._rec_start_time
        if elapsed < 0.2:
            return
        self.recording = False
        import sounddevice as sd
        sd.stop()
        # Get recorded audio
        pos = min(int(elapsed * config.SAMPLE_RATE), len(self._rec_buf))
        audio = self._rec_buf[:pos, 0].copy() if pos > 0 else np.array([], dtype="float32")
        self._rec_buf = None
        if self.tray:
            self.tray.icon = ICON_PROCESSING
            self.tray.title = "Whispr Term — processing..."
        print(f"[STOP] {elapsed:.1f}s ({pos} samples)")

        def _transcribe():
            if len(audio) < config.SAMPLE_RATE * 0.3:
                print("[SKIP] Too short")
                if self.tray:
                    self.tray.icon = ICON_READY
                    self.tray.title = "Whispr Term — ready"
                return
            # Send audio to worker for transcription
            try:
                self.worker.transcribe_audio(audio)
                result = self.worker.get_result(timeout=20)
            except Exception as e:
                print(f"[ERR] worker comm failed: {e}")
                result = None

            if result is None:
                # Worker hung/died (e.g. CUDA context lost after sleep) — restart it
                print("[WARN] No result — restarting worker...")
                if self.tray:
                    self.tray.title = "Whispr Term — restarting engine..."
                try:
                    self.worker.stop()
                except Exception:
                    pass
                try:
                    from transcribe_worker import TranscribeWorker
                    self.worker = TranscribeWorker(self.model_name, self.device)
                    self.worker.start()
                    print("[OK] Worker restarted — retrying...")
                    self.worker.transcribe_audio(audio)
                    result = self.worker.get_result(timeout=30)
                except Exception as e:
                    print(f"[ERR] worker restart failed: {e}")
                    result = None

            if result:
                rtype, old_text, new_text = result
                if rtype == "final" and new_text:
                    time.sleep(0.15)
                    _release_all_modifiers()
                    time.sleep(0.1)
                    _send_text_to_window(new_text)
                    if config.SEND_ENTER:
                        _send_enter_to_window()
                    print(f"[SENT] {new_text}")
                else:
                    print("[SKIP] No speech")
            else:
                print("[ERR] No result after restart")
            if self.tray:
                self.tray.icon = ICON_READY
                self.tray.title = "Whispr Term — ready"

        threading.Thread(target=_transcribe, daemon=True).start()

    def _recording_worker(self):
        """Dedicated thread: does heavy sd.rec()/sd.stop() work off the hook thread.
        The hook callback must stay fast or Windows kills the hook (LowLevelHooksTimeout)."""
        while True:
            if self._pending_start.wait(timeout=0.5):
                self._pending_start.clear()
                self._start_recording()
            if self._pending_stop.is_set():
                self._pending_stop.clear()
                self._stop_recording()

    def setup_hotkey(self):
        """LL keyboard hook: Ctrl+Win = record. Hook callback only sets flags
        (stays fast). A watchdog re-installs the hook if Windows drops it (sleep/wake)."""
        app = self

        # Heavy-work thread — keeps sd.rec() off the hook callback
        threading.Thread(target=self._recording_worker, daemon=True).start()

        @HOOKPROC
        def hook_proc(nCode, wParam, lParam):
            if nCode >= 0:
                vk = lParam.contents.vkCode
                is_down = wParam in (WM_KEYDOWN, WM_SYSKEYDOWN)
                is_up = wParam in (WM_KEYUP, WM_SYSKEYUP)

                if is_down:
                    app._keys_down.add(vk)
                elif is_up:
                    app._keys_down.discard(vk)

                ctrl = VK_LCONTROL in app._keys_down or VK_RCONTROL in app._keys_down
                win = _VK_LWIN in app._keys_down or _VK_RWIN in app._keys_down

                # Win key events — ALWAYS block when Ctrl involved
                if vk in (_VK_LWIN, _VK_RWIN):
                    if is_down and ctrl:
                        app._pending_start.set()
                        app._block_next_win_up = True
                        return 1
                    elif is_up:
                        if app.recording or app._pending_start.is_set():
                            app._pending_stop.set()
                            app._block_next_win_up = False
                            return 1
                        elif app._block_next_win_up:
                            app._block_next_win_up = False
                            return 1
                        elif ctrl:
                            return 1
                    elif ctrl:
                        return 1

                # Ctrl key events
                if vk in (VK_LCONTROL, VK_RCONTROL):
                    if is_down and win:
                        app._pending_start.set()
                        app._block_next_win_up = True
                        return 0
                    elif is_up and (app.recording or app._pending_start.is_set()):
                        app._pending_stop.set()
                        return 0

            return _user32.CallNextHookEx(0, nCode, wParam, lParam)

        self._hook_proc_ref = hook_proc

        def _hook_thread():
            # LL hook requires a BLOCKING GetMessage pump on the installing thread
            # — the callback only fires while the thread is waiting for messages.
            kernel32 = ctypes.windll.kernel32
            thread_id = kernel32.GetCurrentThreadId()
            self._hook_handle = _user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_proc, 0, 0)
            if not self._hook_handle:
                print(f"  [ERR] Hook failed: {ctypes.GetLastError()}")
                self._fallback_polling()
                return
            print(f"  Hook OK: {self._hook_handle}")

            # Watchdog: every 4s post a message that triggers a hook re-install.
            # Posting wakes the blocking GetMessage loop without breaking it.
            def _watchdog():
                while True:
                    time.sleep(4)
                    _user32.PostThreadMessageW(thread_id, WM_REINSTALL_HOOK, 0, 0)
            threading.Thread(target=_watchdog, daemon=True).start()

            msg = wintypes.MSG()
            while _user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
                if msg.message == WM_REINSTALL_HOOK:
                    # Re-install hook — restores it if Windows dropped it (sleep/wake)
                    new = _user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_proc, 0, 0)
                    if new:
                        old = self._hook_handle
                        self._hook_handle = new
                        if old:
                            _user32.UnhookWindowsHookEx(old)
                    continue
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))

        threading.Thread(target=_hook_thread, daemon=True).start()
        print("  Hotkey: Ctrl+Win (LL hook + watchdog, survives sleep/wake)")

    def _fallback_polling(self):
        """Fallback if LL hook fails — polling based."""
        print("  [FALLBACK] Using polling")
        def _poll():
            was = False
            while True:
                ctrl = bool(_user32.GetAsyncKeyState(VK_LCONTROL) & 0x8000) or \
                       bool(_user32.GetAsyncKeyState(VK_RCONTROL) & 0x8000)
                win = bool(_user32.GetAsyncKeyState(_VK_LWIN) & 0x8000) or \
                      bool(_user32.GetAsyncKeyState(_VK_RWIN) & 0x8000)
                both = ctrl and win
                if both and not was:
                    self._start_recording()
                elif not both and was and self.recording:
                    self._stop_recording()
                was = both
                time.sleep(0.01)
        threading.Thread(target=_poll, daemon=True).start()

    def _build_menu(self):
        status = "ready" if self.model_ready else "loading..."
        return pystray.Menu(
            pystray.MenuItem(f"{status} | {self.model_name} ({self.device})", None, enabled=False),
            pystray.MenuItem("Hotkey: Ctrl+Win (hold)", None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.quit_app),
        )

    def quit_app(self, icon=None, item=None):
        if self.worker:
            self.worker.stop()
        if self.tray:
            self.tray.stop()

    def run(self):
        print("Whispr Term starting...")
        print(f"  Model:  {self.model_name} ({self.device})")
        self.setup_hotkey()
        threading.Thread(target=self._load_worker_bg, daemon=True).start()
        self.tray = pystray.Icon("whispr_term", ICON_LOADING, "Whispr Term — loading...", self._build_menu())

        def on_setup(icon):
            icon.visible = True

        self.tray.run(on_setup)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    mutex = ctypes.windll.kernel32.CreateMutexW(None, True, "WhisprTermMutex")
    if ctypes.windll.kernel32.GetLastError() == 183:
        sys.exit(0)
    app = WhisprTerm()
    app.run()

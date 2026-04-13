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
        self._keys_down = set()
        self._hook_proc_ref = None  # prevent GC

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
        self._rec_start_time = time.time()
        if self.tray:
            self.tray.icon = ICON_RECORDING
            self.tray.title = "Whispr Term — RECORDING"
        self.worker.start_recording()
        print("[REC] Recording...")

    def _stop_recording(self):
        if not self.recording:
            return
        elapsed = time.time() - self._rec_start_time
        if elapsed < 0.2:
            return
        self.recording = False
        self.worker.stop_recording()
        if self.tray:
            self.tray.icon = ICON_PROCESSING
            self.tray.title = "Whispr Term — processing..."
        print(f"[STOP] {elapsed:.1f}s")

        def _wait_result():
            result = self.worker.get_result(timeout=30)
            if result:
                rtype, old_text, new_text = result
                if rtype == "final" and new_text:
                    time.sleep(0.2)
                    _release_all_modifiers()
                    time.sleep(0.1)
                    _send_text_to_window(new_text)
                    if config.SEND_ENTER:
                        _send_enter_to_window()
                    print(f"[SENT] {new_text}")
                else:
                    print("[SKIP] No speech")
            else:
                print("[ERR] No result")
            if self.tray:
                self.tray.icon = ICON_READY
                self.tray.title = "Whispr Term — ready"

        threading.Thread(target=_wait_result, daemon=True).start()

    def setup_hotkey(self):
        """LL keyboard hook: Ctrl+Win = record, blocks Win to prevent Start menu."""
        app = self

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

                # Win key events
                if vk in (_VK_LWIN, _VK_RWIN):
                    if is_down and ctrl:
                        # Ctrl already held + Win pressed = START RECORDING
                        app._start_recording()
                        return 1  # block Win
                    elif is_up and app.recording:
                        # Win released while recording = STOP
                        app._stop_recording()
                        return 1  # block Win release (prevent Start menu)
                    elif ctrl:
                        return 1  # block Win while Ctrl held

                # Ctrl key events
                if vk in (VK_LCONTROL, VK_RCONTROL):
                    if is_down and win:
                        # Win already held + Ctrl pressed = START RECORDING
                        app._start_recording()
                        return 0  # don't block Ctrl
                    elif is_up and app.recording:
                        # Ctrl released while recording = STOP
                        app._stop_recording()
                        return 0

            return _user32.CallNextHookEx(0, nCode, wParam, lParam)

        self._hook_proc_ref = hook_proc

        def _hook_thread():
            hook = _user32.SetWindowsHookExW(WH_KEYBOARD_LL, hook_proc, 0, 0)
            if not hook:
                print(f"  [ERR] Hook failed: {ctypes.GetLastError()}")
                # Fallback to polling
                self._fallback_polling()
                return
            print(f"  Hook OK: {hook}")
            msg = wintypes.MSG()
            while _user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
                _user32.TranslateMessage(ctypes.byref(msg))
                _user32.DispatchMessageW(ctypes.byref(msg))
            _user32.UnhookWindowsHookEx(hook)

        threading.Thread(target=_hook_thread, daemon=True).start()
        print("  Hotkey: Ctrl+Win (LL hook, instant, no Start menu)")

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

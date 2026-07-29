"""STT con Whisper + shortcut global (multiplataforma).

F5 empieza a grabar; F5 de nuevo para, transcribe y pega el texto donde tengas
el foco (vía portapapeles + Cmd/Ctrl+V). Si no se puede pegar, el texto queda
igual en el portapapeles listo para pegar a mano.

Motor de transcripción según la plataforma:
  - macOS (Apple Silicon): mlx-whisper sobre la GPU del Mac, usando los modelos
    MLX locales de ~/AI. No descarga nada.
  - Windows / Linux: faster-whisper (CPU o GPU NVIDIA). Descarga el modelo la
    primera vez y lo cachea.

Instalar:  pip install -r requirements.txt
macOS: dar permisos de Accesibilidad y Micrófono a la terminal en
       Ajustes > Privacidad y seguridad.
"""

import os
import queue
import sys
import threading

# Detectar plataforma: en Mac usamos MLX; en el resto, faster-whisper.
IS_MAC = sys.platform == "darwin"

# Apuntar el cache de HuggingFace a ~/AI para que mlx-whisper encuentre el
# modelo local por nombre sin descargar. Debe ir ANTES de importar mlx_whisper.
if IS_MAC:
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/AI"))

import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller, Key

# .env leído a mano (pocos valores) para no sumar python-dotenv.
_env = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env):
    for line in open(_env):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

SAMPLE_RATE = 16000  # whisper trabaja a 16kHz
HOTKEY = keyboard.Key.f5

# El modelo por defecto depende de la plataforma: en Mac, el repo MLX local;
# en Windows/Linux, un tamaño de faster-whisper que descarga solo.
_DEFAULT_MODEL = (
    "mlx-community/whisper-large-v3-turbo" if IS_MAC else "large-v3"
)
MODEL = os.environ.get("WHISPER_MODEL", _DEFAULT_MODEL)
LANG = os.environ.get("WHISPER_LANG") or None

_kbd = Controller()

# --- Motor de transcripción (se elige una vez, según la plataforma) -----------

if IS_MAC:
    import mlx_whisper

    def transcribe(audio):
        result = mlx_whisper.transcribe(
            audio, path_or_hf_repo=MODEL, language=LANG
        )
        return result["text"]
else:
    from faster_whisper import WhisperModel

    # compute_type="int8" corre bien en CPU; en GPU NVIDIA se puede usar "float16".
    _model = WhisperModel(MODEL, compute_type="int8")

    def transcribe(audio):
        segments, _ = _model.transcribe(audio, language=LANG)
        return "".join(seg.text for seg in segments)


print(f"Plataforma: {'macOS (MLX)' if IS_MAC else sys.platform + ' (faster-whisper)'}")
print(f"Modelo: {MODEL}")
print(f"Listo. Pulsá {HOTKEY} para grabar/parar. Ctrl+C para salir.")

_frames = queue.Queue()
recording = False
# Serializa el toggle: impide arrancar una grabación nueva mientras el hilo
# anterior todavía está transcribiendo (los dos tocan _frames y `recording`).
_lock = threading.Lock()


def _callback(indata, frames, time, status):
    if recording:
        _frames.put(indata.copy())


def paste(text):
    """Deja el texto en el portapapeles y simula Cmd/Ctrl+V. Devuelve True si pegó."""
    try:
        pyperclip.copy(text)
        modifier = Key.cmd if IS_MAC else Key.ctrl
        with _kbd.pressed(modifier):
            _kbd.press("v")
            _kbd.release("v")
        return True
    except Exception:
        return False


def toggle():
    global recording
    if not recording:
        while not _frames.empty():  # limpiar audio viejo
            _frames.get()
        recording = True
        print("● Grabando...")
        return

    recording = False
    print("■ Transcribiendo...")
    chunks = []
    while not _frames.empty():
        chunks.append(_frames.get())
    if not chunks:
        print("(sin audio)")
        return

    audio = np.concatenate(chunks).flatten().astype(np.float32)
    text = transcribe(audio).strip()
    print(f"→ {text!r}")
    if text:
        if paste(text):
            print("✔ Pegado en la app activa.")
        else:
            print("✘ No se pudo pegar (¿falta permiso de Accesibilidad?). "
                  "El texto quedó en el portapapeles para pegar a mano.")


def _toggle_locked():
    # El lock asegura que un toggle termine (incluida la transcripción) antes
    # de que otro F5 arranque una grabación nueva. Evita que dos hilos se pisen
    # sobre _frames y `recording`.
    if not _lock.acquire(blocking=False):
        print("… ocupado, esperá a que termine.")
        return
    try:
        toggle()
    finally:
        _lock.release()


def on_press(key):
    if key == HOTKEY:
        # transcribir en otro hilo para no bloquear el listener de teclado
        threading.Thread(target=_toggle_locked, daemon=True).start()


# stream de micrófono siempre abierto; el flag `recording` decide si guardamos
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=_callback):
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

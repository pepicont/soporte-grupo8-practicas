import io
import threading

import pygame
from gtts import gTTS

pygame.mixer.init()
_mixer_lock = threading.Lock()


def _reproducir(texto, lang):
    tts = gTTS(text=texto, lang=lang)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)

    with _mixer_lock:
        pygame.mixer.music.load(fp)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)


def speak(text, lang='en', on_error=None, on_start=None, on_end=None):
    def _run():
        if on_start:
            on_start()
        try:
            _reproducir(text, lang)
        except Exception as exc:
            if on_error:
                on_error(str(exc))
        finally:
            if on_end:
                on_end()

    threading.Thread(target=_run, daemon=True).start()

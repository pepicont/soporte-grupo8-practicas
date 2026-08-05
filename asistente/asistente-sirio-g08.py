"""
Asistente de voz "Sir-IA" (Gemini)
-----------------------------------
Idea general:
1. El micrófono queda escuchando frases cortas hasta reconocer la palabra
   de activación ("Oye Siria" / "Oie Siria").
2. Una vez activado, se abre una charla que se mantiene abierta: no hace
   falta repetir la palabra clave para seguir preguntando.
3. Si pasan 25 segundos sin recibir audio, la charla se cierra y el
   asistente vuelve a esperar la palabra de activación (se pierde el
   contexto de esa charla puntual).
4. Cada charla usa un objeto `chat` propio de google-genai, que va
   acumulando el historial de mensajes automáticamente: por eso Gemini
   "recuerda" lo que se dijo antes dentro de la misma charla, sin que
   nosotros tengamos que reenviar los mensajes previos a mano.
5. Las respuestas de texto se pasan por gTTS para generar un mp3 y se
   reproducen con pygame.

Dependencias (instalar con pip):
    pip install -r requirements.txt


Configuración de la API key:
    1. Crea un .env en la misma carpeta que este script (asistente-sirio-g08.py).
    2. Pegá tu clave de Gemini en la línea GEMINI_API_KEY=... del .env.
"""

import os
import re
import tempfile
import threading

from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import pygame
from google import genai
from google.genai import errors as genai_errors


# ------------------------------------------------------------------
# Configuración general
# ------------------------------------------------------------------
# load_dotenv() busca un archivo ".env" en la carpeta del proyecto y
# carga sus variables como si fueran variables de entorno del sistema.
load_dotenv()

CLAVE_API_GEMINI = os.environ.get("GEMINI_API_KEY")
MODELO_GEMINI = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
MODELOS_DE_RESPALDO = list(dict.fromkeys([
    MODELO_GEMINI,
    "gemini-2.5-flash-lite",
    "gemini-1.5-flash",
]))

IDIOMA_RECONOCIMIENTO = "es-AR"                  # idioma para pasar voz a texto
IDIOMA_VOZ = "es"                                # idioma para pasar texto a voz
PALABRAS_DE_ACTIVACION = ("oye siria", "oie siria")  # frase que despierta al asistente
SEGUNDOS_SIN_RESPUESTA = 25                      # sin audio en este lapso -> vuelve a standby
PALABRAS_PARA_CERRAR = ("salir", "terminar", "chau", "basta")
FRASES_DE_CORTE = ("cortala siria", "corta siria", "basta")  # interrumpen la lectura en curso

cliente_gemini = genai.Client(api_key=CLAVE_API_GEMINI)
pygame.mixer.init()


# ------------------------------------------------------------------
# 1) Detectar la palabra de activación escuchando en frases cortas
# ------------------------------------------------------------------
def detectar_palabra_clave(reconocedor, microfono):
    """
    Escucha en pequeños tramos (para no bloquear el programa de forma
    indefinida) hasta reconocer alguna de las frases de activación.
    Los silencios y los audios que no se entienden simplemente se
    descartan y se sigue escuchando.
    """
    while True:
        try:
            # Cortamos cada intento de escucha a los 3 segundos para
            # poder revisar el resultado y volver a intentar enseguida.
            audio_capturado = reconocedor.listen(
                microfono, timeout=None, phrase_time_limit=3
            )
        except sr.WaitTimeoutError:
            continue

        try:
            texto_oido = reconocedor.recognize_google(
                audio_capturado, language=IDIOMA_RECONOCIMIENTO
            )
        except (sr.UnknownValueError, sr.RequestError):
            continue

        texto_oido = texto_oido.lower().strip()
        if any(frase in texto_oido for frase in PALABRAS_DE_ACTIVACION):
            return True


# ------------------------------------------------------------------
# 2) Abrir una charla nueva probando los modelos de respaldo en orden
# ------------------------------------------------------------------
def abrir_charla_con_memoria():
    """
    Prueba, en orden, cada modelo de MODELOS_DE_RESPALDO hasta lograr
    crear una sesión de chat. Ese objeto de chat guarda su propio
    historial, así que reutilizarlo en los próximos mensajes es lo que
    le da "memoria" a la charla. Devuelve (chat, modelo) o (None, None)
    si no se pudo iniciar con ninguno.
    """
    for modelo in MODELOS_DE_RESPALDO:
        try:
            charla = cliente_gemini.chats.create(model=modelo)
            print(f"Charla iniciada usando el modelo: {modelo}")
            return charla, modelo
        except genai_errors.ClientError as error:
            print(f"No se pudo abrir charla con {modelo}: {error}")

    print("Ningún modelo de la lista de respaldo respondió.")
    return None, None


# ------------------------------------------------------------------
# 3) Mandar el mensaje del usuario a la charla activa
# ------------------------------------------------------------------
def consultar_gemini(charla, modelo_en_uso, mensaje_usuario):
    if charla is None:
        return (
            "No pude abrir la charla con Gemini. Revisá la cuota de la "
            "API, la facturación o el valor de GEMINI_MODEL."
        )

    try:
        respuesta = charla.send_message(mensaje_usuario)
        print(f"Sir-IA ({modelo_en_uso}): {respuesta.text}")
        return respuesta.text
    except genai_errors.ClientError as error:
        print(f"Error al consultar Gemini ({modelo_en_uso}): {error}")
        return (
            "No pude consultar a Gemini en este momento. Revisá la cuota "
            "de la API, la facturación o el valor de GEMINI_MODEL."
        )


# ------------------------------------------------------------------
# 4) Sacarle el formato tipo Markdown antes de convertir a voz
# ------------------------------------------------------------------
def quitar_formato_markdown(texto):
    """
    Gemini suele devolver texto con marcado estilo Markdown (negrita,
    cursiva, títulos, viñetas, links, bloques de código). Si eso se lee
    tal cual, el TTS termina pronunciando símbolos sueltos como
    "asterisco" o "numeral". Acá los sacamos y dejamos solo el texto
    plano que tiene sentido leer en voz alta.
    """
    # Bloques y fragmentos de código entre ``` o `
    texto = re.sub(r"```.*?```", "", texto, flags=re.DOTALL)
    texto = re.sub(r"`([^`]*)`", r"\1", texto)

    # Links en formato [texto](url) -> se queda solo con el texto visible
    texto = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", texto)

    # Encabezados (#, ##, ...) al inicio de línea
    texto = re.sub(r"^\s{0,3}#{1,6}\s*", "", texto, flags=re.MULTILINE)

    # Negrita, cursiva y tachado: **texto**, *texto*, __texto__, ~~texto~~
    texto = re.sub(r"(\*\*\*|\*\*|\*|__|_|~~)", "", texto)

    # Viñetas de listas ("- ", "* ", "+ ") al inicio de línea
    texto = re.sub(r"^\s*[-*+]\s+", "", texto, flags=re.MULTILINE)

    # Espacios repetidos que puedan haber quedado tras las limpiezas
    texto = re.sub(r"[ \t]{2,}", " ", texto)

    return texto.strip()


# ------------------------------------------------------------------
# 5) Escuchar en paralelo si piden cortar la lectura en curso
# ------------------------------------------------------------------
def vigilar_orden_de_corte(reconocedor, microfono, hay_que_parar, se_pidio_corte):
    """
    Corre en un hilo aparte mientras suena el audio de la respuesta.
    Escucha en tramos cortos (1s) y, si en algún momento reconoce una de
    las FRASES_DE_CORTE, frena la reproducción y avisa mediante el
    evento se_pidio_corte. hay_que_parar es la señal para terminar este
    hilo cuando el audio ya terminó por su cuenta.

    Ojo: como el micrófono sigue abierto mientras suena el parlante,
    puede llegar a "escucharse a sí mismo" y generar falsos positivos;
    para un uso más prolijo convendría auriculares o cancelación de eco.
    """
    while not hay_que_parar.is_set():
        try:
            audio_capturado = reconocedor.listen(
                microfono, timeout=1, phrase_time_limit=3
            )
        except sr.WaitTimeoutError:
            continue

        try:
            texto_oido = reconocedor.recognize_google(
                audio_capturado, language=IDIOMA_RECONOCIMIENTO
            ).lower().strip()
        except (sr.UnknownValueError, sr.RequestError):
            continue

        if any(frase in texto_oido for frase in FRASES_DE_CORTE):
            pygame.mixer.music.stop()
            se_pidio_corte.set()
            return


# ------------------------------------------------------------------
# 6) Pasar la respuesta a voz y reproducirla (cancelable por voz)
# ------------------------------------------------------------------
def reproducir_respuesta(texto, reconocedor, microfono):
    texto_limpio = quitar_formato_markdown(texto)

    # El mp3 se genera como archivo temporal del sistema operativo
    # (en Windows, dentro de algo como C:\Users\<usuario>\AppData\Local\Temp\,
    # que es lo que devuelve tempfile.gettempdir()). Se borra apenas
    # termina de reproducirse, así no se acumulan archivos de audio.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as archivo_mp3:
        ruta_mp3 = archivo_mp3.name

    voz = gTTS(text=texto_limpio, lang=IDIOMA_VOZ)
    voz.save(ruta_mp3)

    hay_que_parar = threading.Event()
    se_pidio_corte = threading.Event()
    hilo_vigia = threading.Thread(
        target=vigilar_orden_de_corte,
        args=(reconocedor, microfono, hay_que_parar, se_pidio_corte),
        daemon=True,
    )

    pygame.mixer.music.load(ruta_mp3)
    pygame.mixer.music.play()
    hilo_vigia.start()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    # Si la música ya terminó sola, le avisamos al hilo vigía que pare
    # de escuchar (si fue el propio hilo el que la cortó, esto es un
    # no-op porque ya volvió del bucle).
    hay_que_parar.set()
    hilo_vigia.join(timeout=2)

    pygame.mixer.music.unload()
    os.remove(ruta_mp3)

    if se_pidio_corte.is_set():
        print("Listo, corté la lectura. Te escucho de nuevo.")


# ------------------------------------------------------------------
# Bucle principal
# ------------------------------------------------------------------
def ejecutar_asistente():
    print(
        'Bienvenido al asistente Sirio "Sir-IA" '
        f"(usa el modelo: {MODELO_GEMINI})"
    )
    print("Palabra de activación: 'Oye Siria' / 'Oie Siria'.\n")
    print(
        "Una vez activado, la charla queda abierta: podés seguir "
        "preguntando sin repetir la palabra clave."
    )
    print(
        f"Si pasan {SEGUNDOS_SIN_RESPUESTA} segundos sin que digas nada, "
        "vuelve a standby y esa charla puntual se cierra."
    )
    print(
        "Si una respuesta es muy larga, decí 'cortala Siria' o 'basta' "
        "mientras la está leyendo para que pare y puedas preguntar otra "
        "cosa sin salir de la charla."
    )
    print("Decí 'salir', 'terminar', 'basta' o 'chau' para cerrar el programa.\n")

    reconocedor = sr.Recognizer()

    with sr.Microphone() as microfono:
        print(f"Listo. Esperando que digas 'Oye Siria'...\n")

        while True:
            # Standby: se queda acá hasta escuchar la palabra clave
            detectar_palabra_clave(reconocedor, microfono)
            print("\n¡Te escucho!")

            # Cada activación abre una charla nueva con su propia memoria
            charla_actual, modelo_activo = abrir_charla_con_memoria()

            charla_abierta = True
            while charla_abierta:
                print(
                    f"Escuchando... (si no decís nada en "
                    f"{SEGUNDOS_SIN_RESPUESTA}s vuelvo a standby)"
                )
                try:
                    audio_capturado = reconocedor.listen(
                        microfono, timeout=SEGUNDOS_SIN_RESPUESTA, phrase_time_limit=15
                    )
                    mensaje_usuario = reconocedor.recognize_google(
                        audio_capturado, language=IDIOMA_RECONOCIMIENTO
                    )
                    print(f"Vos dijiste: {mensaje_usuario}")
                except sr.WaitTimeoutError:
                    print(
                        f"\nPasaron {SEGUNDOS_SIN_RESPUESTA}s sin actividad. "
                        "Vuelvo a standby...\n"
                    )
                    charla_abierta = False
                    continue
                except sr.UnknownValueError:
                    print("No entendí lo que dijiste, seguimos en la charla.")
                    continue
                except sr.RequestError as error:
                    print(f"Error con el servicio de reconocimiento: {error}")
                    continue

                if mensaje_usuario.lower().strip() in PALABRAS_PARA_CERRAR:
                    print("Cerrando Sir-IA...")
                    return

                texto_respuesta = consultar_gemini(
                    charla_actual, modelo_activo, mensaje_usuario
                )
                reproducir_respuesta(texto_respuesta, reconocedor, microfono)

            print("Esperando que digas 'Oye Siria'...")


if __name__ == "__main__":
    ejecutar_asistente()

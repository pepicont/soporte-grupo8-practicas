from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import site
import sqlite3
import sys
import time

import multiprocessing

import cv2
import face_recognition
import numpy as np


def _agregar_site_packages_locales() -> None:
    ruta_site_packages = Path(__file__).resolve().parents[1] / ".venv" / "Lib" / "site-packages"
    if ruta_site_packages.exists() and str(ruta_site_packages) not in sys.path:
        site.addsitedir(str(ruta_site_packages))


_agregar_site_packages_locales()

from flask import Flask, abort, flash, redirect, render_template, request, send_file, url_for
from PIL import Image, ImageDraw

APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "banco.db"
DATASET_DIR = APP_DIR / "dataset_caras"

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "pagoconreconocimiento")

ENCODINGS_CONOCIDOS: list[np.ndarray] = []
NOMBRES_CONOCIDOS: list[str] = []


# --- Base de datos -------------------------------------------------------
def inicializar_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre TEXT UNIQUE NOT NULL,
            saldo REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.commit()
    conn.close()


def obtener_usuarios_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, saldo FROM usuarios ORDER BY id ASC")
    usuarios = cursor.fetchall()
    conn.close()
    return usuarios


def obtener_usuario_por_id(usuario_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, nombre, saldo FROM usuarios WHERE id = ?", (usuario_id,))
    usuario = cursor.fetchone()
    conn.close()
    return usuario


def actualizar_saldo(nombre, monto_a_cobrar):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT saldo FROM usuarios WHERE nombre = ?", (nombre,))
    resultado = cursor.fetchone()

    if resultado is None:
        conn.close()
        return False, "Usuario no registrado en DB."

    saldo_actual = float(resultado[0])
    if saldo_actual >= monto_a_cobrar:
        nuevo_saldo = saldo_actual - monto_a_cobrar
        cursor.execute("UPDATE usuarios SET saldo = ? WHERE nombre = ?", (nuevo_saldo, nombre))
        conn.commit()
        conn.close()
        return True, nuevo_saldo

    conn.close()
    return False, f"Saldo insuficiente. Tiene ${saldo_actual:.2f}"


def registrar_usuario_en_db(nombre, saldo_inicial=1000.0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO usuarios (nombre, saldo) VALUES (?, ?)", (nombre, saldo_inicial))
    conn.commit()
    conn.close()


def modificar_usuario(nombre_actual, nuevo_nombre=None, nuevo_saldo=None):
    nombre_nuevo = nombre_actual if nuevo_nombre is None else nuevo_nombre.strip()
    if not nombre_nuevo:
        return False, "El nombre no puede quedar vacío."

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM usuarios WHERE nombre = ?", (nombre_actual,))
    if cursor.fetchone() is None:
        conn.close()
        return False, "El usuario ya no existe."

    if nombre_nuevo != nombre_actual:
        cursor.execute("SELECT 1 FROM usuarios WHERE nombre = ?", (nombre_nuevo,))
        if cursor.fetchone() is not None:
            conn.close()
            return False, f"Ya existe un usuario llamado '{nombre_nuevo}'."

    ruta_origen = obtener_ruta_imagen_usuario(nombre_actual)
    ruta_destino = None
    if ruta_origen:
        ruta_destino = DATASET_DIR / f"{nombre_nuevo}{ruta_origen.suffix}"
        if nombre_nuevo != nombre_actual and ruta_destino.exists():
            conn.close()
            return False, f"Ya existe el archivo de dataset para '{nombre_nuevo}'."

    try:
        if nombre_nuevo != nombre_actual:
            cursor.execute("UPDATE usuarios SET nombre = ? WHERE nombre = ?", (nombre_nuevo, nombre_actual))
        if nuevo_saldo is not None:
            cursor.execute("UPDATE usuarios SET saldo = ? WHERE nombre = ?", (float(nuevo_saldo), nombre_nuevo))
        conn.commit()
        conn.close()

        if ruta_origen and ruta_destino and ruta_origen != ruta_destino:
            ruta_origen.rename(ruta_destino)

        return True, nombre_nuevo
    except Exception as error:
        conn.rollback()
        conn.close()
        return False, f"No se pudo modificar el usuario: {error}"


def ingresar_dinero_usuario(nombre, monto):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT saldo FROM usuarios WHERE nombre = ?", (nombre,))
    resultado = cursor.fetchone()
    if resultado is None:
        conn.close()
        return False, "Usuario no registrado en DB."

    nuevo_saldo = float(resultado[0]) + float(monto)
    cursor.execute("UPDATE usuarios SET saldo = ? WHERE nombre = ?", (nuevo_saldo, nombre))
    conn.commit()
    conn.close()
    return True, nuevo_saldo


def eliminar_usuario_completo(nombre):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM usuarios WHERE nombre = ?", (nombre,))
    conn.commit()
    conn.close()

    if not DATASET_DIR.exists():
        return True

    for extension in (".jpg", ".jpeg", ".png"):
        ruta = DATASET_DIR / f"{nombre}{extension}"
        if ruta.exists():
            ruta.unlink()

    return True


# --- Dataset facial ------------------------------------------------------
def obtener_ruta_imagen_usuario(nombre):
    for extension in (".jpg", ".jpeg", ".png"):
        ruta = DATASET_DIR / f"{nombre}{extension}"
        if ruta.exists():
            return ruta
    return None


def cargar_dataset(ruta_dataset=DATASET_DIR):
    encodings_conocidos = []
    nombres_conocidos = []

    ruta_dataset = Path(ruta_dataset)
    ruta_dataset.mkdir(parents=True, exist_ok=True)

    for archivo in ruta_dataset.iterdir():
        if archivo.suffix.lower() in (".jpg", ".jpeg", ".png"):
            imagen = face_recognition.load_image_file(str(archivo))
            encodings = face_recognition.face_encodings(imagen, num_jitters=10)
            if encodings:
                encodings_conocidos.append(encodings[0])
                nombre = archivo.stem
                nombres_conocidos.append(nombre)
                registrar_usuario_en_db(nombre)

    return encodings_conocidos, nombres_conocidos


def recargar_datos():
    global ENCODINGS_CONOCIDOS, NOMBRES_CONOCIDOS
    ENCODINGS_CONOCIDOS, NOMBRES_CONOCIDOS = cargar_dataset()


# --- Registros con OpenCV desktop ---------------------------------------
def _worker_registro(conn, nombre_usuario, encodings_conocidos, nombres_conocidos, dataset_dir, db_path):
    """Corre en proceso separado para poder usar cv2.imshow en cualquier SO."""
    ruta_imagen = dataset_dir / f"{nombre_usuario}.jpg"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        conn.send((False, "No se pudo acceder a la camara."))
        conn.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                conn.send((False, "No se pudo capturar video desde la camara."))
                return

            frame_pequeno = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame_pequeno = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)
            ubicaciones_caras = face_recognition.face_locations(rgb_frame_pequeno, model="hog")

            for top, right, bottom, left in ubicaciones_caras:
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.putText(frame, "SPACE para capturar  |  Q para cancelar", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Usuario: {nombre_usuario}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            estado = "Rostro detectado" if ubicaciones_caras else "No se detecta rostro"
            color_e = (0, 255, 0) if ubicaciones_caras else (0, 0, 255)
            cv2.putText(frame, estado, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_e, 2)

            cv2.imshow(f"Registro - {nombre_usuario}", frame)
            tecla = cv2.waitKey(1) & 0xFF

            if tecla == ord(" "):
                if not ubicaciones_caras:
                    continue

                encodings_captura = face_recognition.face_encodings(rgb_frame_pequeno, ubicaciones_caras)
                if encodings_captura and encodings_conocidos:
                    distancias = face_recognition.face_distance(encodings_conocidos, encodings_captura[0])
                    mejor_idx = int(np.argmin(distancias))
                    if distancias[mejor_idx] < 0.45:
                        conn.send((False, f"Esta persona ya existe como '{nombres_conocidos[mejor_idx]}'."))
                        return

                dataset_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(ruta_imagen), frame)

                # Registro en DB desde el subproceso (SQLite soporta acceso por archivo)
                import sqlite3 as _sqlite3
                c = _sqlite3.connect(db_path)
                try:
                    c.execute(
                        "INSERT OR IGNORE INTO usuarios (nombre, saldo) VALUES (?, ?)",
                        (nombre_usuario, 1000.0),
                    )
                    c.commit()
                finally:
                    c.close()

                conn.send((True, f"Usuario '{nombre_usuario}' registrado con saldo inicial de $1000."))
                return

            if tecla == ord("q"):
                conn.send((False, "Registro cancelado."))
                return
    except Exception as e:
        conn.send((False, str(e)))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        conn.close()


def registrar_nuevo_usuario_desktop(nombre_usuario, encodings_conocidos, nombres_conocidos):
    nombre_usuario = (nombre_usuario or "").strip()
    if not nombre_usuario:
        return False, "Registro cancelado: no ingresaste nombre."

    ruta_imagen = DATASET_DIR / f"{nombre_usuario}.jpg"
    if ruta_imagen.exists():
        return False, f"El usuario '{nombre_usuario}' ya esta registrado."

    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_worker_registro,
        args=(child_conn, nombre_usuario, encodings_conocidos, nombres_conocidos, DATASET_DIR, DB_PATH),
        daemon=True,
    )
    p.start()
    child_conn.close()

    try:
        exito, mensaje = parent_conn.recv()
    except EOFError:
        return False, "El proceso de registro termino inesperadamente."
    finally:
        parent_conn.close()
        p.join(timeout=120)

    if exito:
        recargar_datos()
    return exito, mensaje


# --- rPPG ---------------------------------------------------------------
def _senal_diff(r, g, b):
    diff = (g - r) / (g + r + b + 1e-6)
    return diff - diff.mean()


def _analizar_ventana(senal, fps):
    n = len(senal)
    if n == 0:
        return 0.0, 0.0, 0.0

    ventana = (senal - np.mean(senal)) * np.hanning(n)
    espectro = np.abs(np.fft.rfft(ventana))
    freq = np.fft.rfftfreq(n, d=1.0 / fps)
    m_hr = (freq >= 0.7) & (freq <= 3.0)
    m_util = (freq >= 0.15) & (freq <= 5.0)

    pot_hr = np.sum(espectro[m_hr] ** 2)
    pot_util = np.sum(espectro[m_util] ** 2)
    ratio_hr = pot_hr / pot_util if pot_util > 0 else 0.0

    hr_espectro = espectro[m_hr]
    if hr_espectro.size > 0 and np.mean(hr_espectro ** 2) > 0:
        peak_snr = float(np.max(hr_espectro) ** 2 / np.mean(hr_espectro ** 2))
        freq_dom = float(freq[m_hr][np.argmax(hr_espectro)])
    else:
        peak_snr, freq_dom = 0.0, 0.0

    return ratio_hr, peak_snr, freq_dom


def verificar_liveness_rppg(cap, duracion_segundos=10, encodings_conocidos=None, nombres_conocidos=None):
    señales_r, señales_g, señales_b = [], [], []
    tiempo_inicio = time.time()
    frame_count = 0
    votos_recono = {}
    votos_min_recono = 5

    while True:
        t = time.time() - tiempo_inicio
        if t >= duracion_segundos:
            break

        ret, frame = cap.read()
        if not ret:
            break

        frame_d = frame.copy()
        h_f, w_f = frame_d.shape[:2]
        frame_count += 1

        frame_peq = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_peq = cv2.cvtColor(frame_peq, cv2.COLOR_BGR2RGB)
        ubicaciones = face_recognition.face_locations(rgb_peq, model="hog")

        if ubicaciones:
            top, right, bottom, left = ubicaciones[0]
            top, right, bottom, left = top * 4, right * 4, bottom * 4, left * 4

            roi = frame[top:bottom, left:right]
            if roi.size > 0:
                h_r, w_r = roi.shape[:2]
                iy, ix = int(h_r * 0.15), int(w_r * 0.10)
                roi_piel = roi[iy:h_r - iy, ix:w_r - ix]
                if roi_piel.size > 0:
                    señales_b.append(float(np.mean(roi_piel[:, :, 0])))
                    señales_g.append(float(np.mean(roi_piel[:, :, 1])))
                    señales_r.append(float(np.mean(roi_piel[:, :, 2])))

            cv2.rectangle(frame_d, (left, top), (right, bottom), (0, 200, 255), 2)
            cv2.putText(frame_d, "Leyendo pulso...", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

            if frame_count % 5 == 0 and encodings_conocidos:
                rgb_full = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_full, [(top, right, bottom, left)])
                if encs:
                    dist = face_recognition.face_distance(encodings_conocidos, encs[0])
                    if len(dist) > 0:
                        idx = int(np.argmin(dist))
                        if dist[idx] < 0.45:
                            nom = nombres_conocidos[idx]
                            votos_recono[nom] = votos_recono.get(nom, 0) + 1

            if votos_recono:
                mejor_nom = max(votos_recono, key=votos_recono.get)
                v = votos_recono[mejor_nom]
                color_id = (50, 255, 50) if v >= votos_min_recono else (0, 200, 255)
                cv2.putText(frame_d, f"ID: {mejor_nom}  ({v}/{votos_min_recono})", (left, bottom + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_id, 2)
        else:
            cv2.putText(frame_d, "Acerque el rostro a la camara", (10, h_f // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        cv2.rectangle(frame_d, (0, 0), (w_f, 45), (30, 30, 30), -1)
        cv2.putText(frame_d, "VERIFICANDO PERSONA REAL  (rPPG)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.putText(frame_d, "Permanezca quieto con buena iluminacion", (10, h_f - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        bar_y = h_f - 30
        progreso = min(t / duracion_segundos, 1.0)
        barra_x = 10 + int((w_f - 20) * progreso)
        cv2.rectangle(frame_d, (10, bar_y), (w_f - 10, bar_y + 18), (60, 60, 60), -1)
        cv2.rectangle(frame_d, (10, bar_y), (barra_x, bar_y + 18), (0, 180, 100), -1)
        cv2.rectangle(frame_d, (10, bar_y), (w_f - 10, bar_y + 18), (160, 160, 160), 1)
        seg_rest = max(0, int(duracion_segundos - t + 0.99))
        cv2.putText(frame_d, f"{seg_rest}s", (w_f - 42, bar_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        n_sig = min(len(señales_g), len(señales_r), len(señales_b))
        if n_sig > 5:
            sig_disp = _senal_diff(
                np.array(señales_r[-80:]),
                np.array(señales_g[-80:]),
                np.array(señales_b[-80:]),
            )
            gx, gy, gw, gh = w_f - 158, h_f - 128, 143, 62
            cv2.rectangle(frame_d, (gx - 2, gy - 18), (gx + gw + 2, gy + gh + 2), (40, 40, 40), -1)
            cv2.putText(frame_d, "Senal diferencial G-R", (gx, gy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 255, 150), 1)
            s_min, s_max = sig_disp.min(), sig_disp.max()
            s_rng = max(s_max - s_min, 1e-6)
            pts = []
            for i, v in enumerate(sig_disp):
                px = gx + int(i * gw / len(sig_disp))
                py = gy + gh - int((v - s_min) / s_rng * gh)
                pts.append([px, py])
            pts_arr = np.array(pts, dtype=np.int32)
            if len(pts_arr) > 1:
                cv2.polylines(frame_d, [pts_arr], isClosed=False, color=(0, 230, 130), thickness=1)

        cv2.imshow("Verificacion Biometrica", frame_d)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            cv2.destroyWindow("Verificacion Biometrica")
            return False, 0.0, "Verificacion cancelada.", None

    cv2.destroyWindow("Verificacion Biometrica")

    n_min = min(len(señales_g), len(señales_r), len(señales_b))
    if n_min < 60:
        return False, 0.0, "Pocas muestras", None

    R = np.array(señales_r[:n_min], dtype=np.float64)
    G = np.array(señales_g[:n_min], dtype=np.float64)
    B = np.array(señales_b[:n_min], dtype=np.float64)
    fps_real = n_min / duracion_segundos

    diff = _senal_diff(R, G, B)
    std_diff = float(np.std(diff))
    ratio_hr, peak_snr, _ = _analizar_ventana(diff, fps_real)

    win_len = int(4.0 * fps_real)
    win_paso = int(1.5 * fps_real)
    freqs_dom = []
    i = 0
    while i + win_len <= n_min:
        _, _, fd = _analizar_ventana(diff[i:i + win_len], fps_real)
        if fd > 0:
            freqs_dom.append(fd)
        i += win_paso

    consistencia = max(0.0, 1.0 - np.std(freqs_dom) / 0.40) if len(freqs_dom) >= 3 else 0.0

    score = round(
        0.25 * min(std_diff / 0.0008, 1.0)
        + 0.25 * min(ratio_hr / 0.25, 1.0)
        + 0.20 * min(peak_snr / 4.0, 1.0)
        + 0.30 * consistencia,
        3,
    )

    ok_std = std_diff >= 0.0002
    ok_ratio = ratio_hr >= 0.10
    ok_snr = peak_snr >= 1.5
    ok_consist = consistencia >= 0.20
    es_vivo = ok_std and ok_ratio and ok_snr and ok_consist

    diag = (
        f"std={std_diff:.5f}({'OK' if ok_std else 'FAIL'})  "
        f"ratio={ratio_hr:.3f}({'OK' if ok_ratio else 'FAIL'})  "
        f"snr={peak_snr:.2f}({'OK' if ok_snr else 'FAIL'})  "
        f"consist={consistencia:.3f}({'OK' if ok_consist else 'FAIL'})\n"
        f"muestras={n_min}  fps={fps_real:.1f}"
    )
    print(f"[rPPG] {diag}")

    usuario_reconocido = None
    if votos_recono:
        mejor_nom = max(votos_recono, key=votos_recono.get)
        if votos_recono[mejor_nom] >= votos_min_recono:
            usuario_reconocido = mejor_nom

    return es_vivo, score, diag, usuario_reconocido


# --- Acciones desktop ----------------------------------------------------
def _worker_escaneo(conn, encodings_conocidos, nombres_conocidos):
    """Corre en proceso separado para poder usar cv2.imshow en cualquier SO."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        conn.send((False, 0.0, "No se pudo acceder a la camara.", None))
        conn.close()
        return
    try:
        resultado = verificar_liveness_rppg(
            cap,
            encodings_conocidos=encodings_conocidos,
            nombres_conocidos=nombres_conocidos,
        )
    except Exception as e:
        resultado = (False, 0.0, str(e), None)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    conn.send(resultado)
    conn.close()


def iniciar_escaneo_desktop(monto, encodings_conocidos, nombres_conocidos):
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    p = multiprocessing.Process(
        target=_worker_escaneo,
        args=(child_conn, encodings_conocidos, nombres_conocidos),
        daemon=True,
    )
    p.start()
    child_conn.close()

    try:
        es_vivo, score_liveness, diag_liveness, usuario_reconocido = parent_conn.recv()
    except EOFError:
        return False, {
            "titulo": "Error",
            "detalle": "El proceso de escaneo termino inesperadamente.",
            "exito": False,
        }
    finally:
        parent_conn.close()
        p.join(timeout=60)

    if not es_vivo:
        return False, {
            "titulo": "Verificacion fallida",
            "detalle": (
                f"No se detecto senal de vida (score: {score_liveness:.3f})\n\n"
                f"{diag_liveness}\n\n"
                "Causas posibles:\n"
                "- Foto o pantalla\n"
                "- Poca iluminacion\n"
                "- Movimiento excesivo"
            ),
            "exito": False,
        }

    if not usuario_reconocido:
        return False, {
            "titulo": "No identificado",
            "detalle": (
                "Se verifico que eres una persona real, pero no se pudo identificar a nadie.\n\n"
                "Asegurate de estar registrado y mirar directamente a la camara durante la verificacion."
            ),
            "exito": False,
        }

    exito, resultado = actualizar_saldo(usuario_reconocido, monto)
    if exito:
        return True, {
            "titulo": "Pago aprobado",
            "detalle": (
                f"Cobro exitoso a {usuario_reconocido}\n"
                f"Monto cobrado: ${monto:.2f}\n"
                f"Saldo restante: ${resultado:.2f}\n\n"
                f"Score de liveness: {score_liveness:.3f}"
            ),
            "exito": True,
            "usuario": usuario_reconocido,
            "monto": monto,
            "saldo_restante": resultado,
            "score_liveness": score_liveness,
        }

    return False, {
        "titulo": "Pago rechazado",
        "detalle": f"Error con {usuario_reconocido}: {resultado}",
        "exito": False,
    }


# --- Utilidades visuales -------------------------------------------------
def _placeholder_png():
    imagen = Image.new("RGB", (160, 160), (230, 233, 238))
    dibujo = ImageDraw.Draw(imagen)
    dibujo.rectangle((18, 18, 142, 142), outline=(160, 168, 178), width=3)
    dibujo.text((44, 70), "SIN FOTO", fill=(96, 102, 112))
    buffer = BytesIO()
    imagen.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


# --- Flask ---------------------------------------------------------------
@app.route("/")
def index():
    usuarios = obtener_usuarios_db()
    return render_template(
        "index.html",
        total_usuarios=len(usuarios),
        cantidad_dataset=len(NOMBRES_CONOCIDOS),
    )


@app.route("/foto/<nombre>")
def foto_usuario(nombre):
    ruta = obtener_ruta_imagen_usuario(nombre)
    if ruta and ruta.exists():
        return send_file(ruta)
    return send_file(_placeholder_png(), mimetype="image/png")


@app.route("/registrar", methods=["GET", "POST"])
def registrar_page():
    if request.method == "POST":
        nombre = request.form.get("nombre_usuario", "")
        exito, mensaje = registrar_nuevo_usuario_desktop(nombre, ENCODINGS_CONOCIDOS, NOMBRES_CONOCIDOS)
        if exito:
            recargar_datos()
        # url_for() sirve para construir una URL a partir del nombre de una ruta de Flask.
        # Sintaxis: url_for("nombre_de_la_ruta", parametro=valor).
        # Ejemplos: url_for("index") -> "/" y url_for("foto_usuario", nombre="Ana") -> "/foto/Ana".
        # Básicamente busca el decorador @app.route asociado a la función que le pasamos y construye la URL correspondiente.
        # En este caso, url_for("registrar_page") devuelve "/registrar" (por el app.route ser /registrar y def registrar_page).
        return render_template(
            "resultado.html",
            titulo="Registro de usuario",
            detalle=mensaje,
            exito=exito,
            volver_url=url_for("registrar_page"),
            inicio_url=url_for("index"),
        )

    return render_template("registrar.html")


@app.route("/cobrar", methods=["GET", "POST"])
def cobrar_page():
    if request.method == "POST":
        monto_texto = request.form.get("monto", "0").strip()
        try:
            monto = float(monto_texto)
        except ValueError:
            return render_template(
                "resultado.html",
                titulo="Monto invalido",
                detalle="El monto debe ser numerico.",
                exito=False,
                volver_url=url_for("cobrar_page"),
                inicio_url=url_for("index"),
            )

        if monto <= 0:
            return render_template(
                "resultado.html",
                titulo="Monto invalido",
                detalle="El monto debe ser mayor que cero.",
                exito=False,
                volver_url=url_for("cobrar_page"),
                inicio_url=url_for("index"),
            )

        exito, resultado = iniciar_escaneo_desktop(monto, ENCODINGS_CONOCIDOS, NOMBRES_CONOCIDOS)
        if exito:
            recargar_datos()

        return render_template(
            "resultado.html",
            titulo=resultado["titulo"],
            detalle=resultado["detalle"],
            exito=resultado["exito"],
            volver_url=url_for("cobrar_page"),
            inicio_url=url_for("index"),
        )

    return render_template("cobrar.html")


@app.route("/usuarios")
def usuarios_page():
    usuarios_raw = obtener_usuarios_db()
    usuarios = [{"id": uid, "nombre": nombre, "saldo": saldo} for uid, nombre, saldo in usuarios_raw]
    return render_template("usuarios.html", usuarios=usuarios)


@app.route("/usuarios/<int:usuario_id>/editar", methods=["GET", "POST"])
def editar_usuario_page(usuario_id):
    usuario = obtener_usuario_por_id(usuario_id)
    if usuario is None:
        abort(404)

    usuario_dict = {"id": usuario[0], "nombre": usuario[1], "saldo": usuario[2]}

    if request.method == "POST":
        nuevo_nombre = request.form.get("nombre", usuario_dict["nombre"]).strip()
        saldo_texto = request.form.get("saldo", "").strip()
        nuevo_saldo = None
        if saldo_texto:
            try:
                nuevo_saldo = float(saldo_texto)
            except ValueError:
                return render_template(
                    "editar_usuario.html",
                    usuario=usuario_dict,
                    error="El saldo debe ser numerico.",
                )

        exito, mensaje = modificar_usuario(usuario_dict["nombre"], nuevo_nombre=nuevo_nombre, nuevo_saldo=nuevo_saldo)
        if exito:
            recargar_datos()
            flash(f"Usuario actualizado: {mensaje}", "success")
            return redirect(url_for("usuarios_page"))

        return render_template("editar_usuario.html", usuario=usuario_dict, error=mensaje)

    return render_template("editar_usuario.html", usuario=usuario_dict)


@app.route("/usuarios/<int:usuario_id>/depositar", methods=["GET", "POST"])
def depositar_usuario_page(usuario_id):
    usuario = obtener_usuario_por_id(usuario_id)
    if usuario is None:
        abort(404)

    usuario_dict = {"id": usuario[0], "nombre": usuario[1], "saldo": usuario[2]}

    if request.method == "POST":
        monto_texto = request.form.get("monto_ingreso", "").strip()
        try:
            monto = float(monto_texto)
        except ValueError:
            return render_template(
                "depositar_usuario.html",
                usuario=usuario_dict,
                error="El monto debe ser numerico.",
            )

        if monto <= 0:
            return render_template(
                "depositar_usuario.html",
                usuario=usuario_dict,
                error="El monto debe ser mayor que cero.",
            )

        exito, resultado = ingresar_dinero_usuario(usuario_dict["nombre"], monto)
        if exito:
            recargar_datos()
            flash(f"Nuevo saldo de {usuario_dict['nombre']}: ${resultado:.2f}", "success")
            return redirect(url_for("usuarios_page"))

        return render_template("depositar_usuario.html", usuario=usuario_dict, error=resultado)

    return render_template("depositar_usuario.html", usuario=usuario_dict)


@app.route("/usuarios/<int:usuario_id>/eliminar", methods=["POST"])
def eliminar_usuario_page(usuario_id):
    usuario = obtener_usuario_por_id(usuario_id)
    if usuario is None:
        abort(404)

    eliminar_usuario_completo(usuario[1])
    recargar_datos()
    flash(f"Usuario '{usuario[1]}' eliminado junto con su dataset.", "success")
    return redirect(url_for("usuarios_page"))


@app.route("/salud")
def salud():
    return {"status": "ok", "usuarios": len(NOMBRES_CONOCIDOS)}


if __name__ == "__main__":
    multiprocessing.freeze_support()
    inicializar_db()
    recargar_datos()
    app.run(host="127.0.0.1", port=5000, debug=True)

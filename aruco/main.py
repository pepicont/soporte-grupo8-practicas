import cv2
import numpy as np
import threading
import time
from kivy.app import App
from kivy.uix.image import Image as KivyImage
from kivy.clock import Clock
from kivy.graphics.texture import Texture

MARKER_SIZE_CM = 5.0
ALPHA = 0.65    # transparencia del cubo (0=invisible, 1=sólido)

_S = MARKER_SIZE_CM
_H = _S / 2

# Las 4 esquinas del marcador ArUco en coordenadas 3D reales (en cm, Z=0 = plano del papel).
# Orden: TL, TR, BR, BL — mismo orden que devuelve el detector.
MARKER_OBJ_PTS = np.float32([
    [-_H,  _H, 0], [_H,  _H, 0], [_H, -_H, 0], [-_H, -_H, 0],
])

# 8 vértices del cubo: 4 en la base (Z=0, apoyados en el marcador) y 4 en el techo (Z=_S).
CUBE_VERTS = np.float32([
    [-_H, -_H,  0], [_H, -_H,  0], [_H,  _H,  0], [-_H,  _H,  0],  # base
    [-_H, -_H, _S], [_H, -_H, _S], [_H,  _H, _S], [-_H,  _H, _S],  # techo
])

# Cada cara: lista de índices en CUBE_VERTS + nombre de textura.
# Orden TL→TR→BR→BL por cara para que la tira de pasto quede arriba en los laterales.
FACES = [
    ([4, 5, 6, 7], 'top'),
    ([0, 1, 2, 3], 'bottom'),
    ([4, 5, 1, 0], 'side'),
    ([5, 6, 2, 1], 'side'),
    ([6, 7, 3, 2], 'side'),
    ([7, 4, 0, 3], 'side'),
]

def make_textures(size=32):
    """Genera imágenes pequeñas con ruido que imitan los bloques de Minecraft."""
    rng = np.random.default_rng(42)

    def noisy(color):
        # Crea un parche del color base y le suma ruido píxel a píxel.
        t = np.full((size, size, 3), color, dtype=np.int32)
        t += rng.integers(-28, 28, (size, size, 3))
        return np.clip(t, 0, 255).astype(np.uint8)

    side = noisy([134, 96, 67])
    side[:max(2, size // 8)] = noisy([88, 140, 74])[:max(2, size // 8)]  # tira de pasto arriba
    return {
        'top':    noisy([88, 140, 74]),
        'side':   side,
        'bottom': noisy([134, 96, 67]),
    }

def camera_matrix(w, h):
    """
    Matriz intrínseca de cámara aproximada cuando no hay calibración real.
    f = focal length estimada como el lado mayor de la imagen (en píxeles).
    cx, cy = centro óptico asumido en el centro del frame.
    """
    f = max(w, h)
    return np.float32([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])

def draw_minecraft_cube(frame, pts2d, verts_cam, textures):
    """
    Dibuja el cubo sobre el frame usando painter's algorithm:
    ordena las caras de más lejos a más cerca (por profundidad Z en cámara)
    y las pinta en ese orden para que las cercanas tapen a las lejanas.

    Para cada cara:
      1. getPerspectiveTransform: calcula la homografía textura → cuadrilátero 2D.
      2. warpPerspective: deforma la textura al cuadrilátero proyectado.
      3. fillPoly + addWeighted: aplica la textura con transparencia (ALPHA).
    """
    h_f, w_f = frame.shape[:2]
    # Ordena por Z promedio de la cara en espacio cámara (mayor Z = más lejos).
    faces_by_depth = sorted(
        [(np.mean([verts_cam[i][2] for i in vidx]), vidx, tname) for vidx, tname in FACES],
        reverse=True,  # painter's algorithm: más lejos primero
    )
    for _, vidx, tname in faces_by_depth:
        dst = np.array([pts2d[i] for i in vidx], dtype=np.float32)
        tex = textures[tname]
        t_h, t_w = tex.shape[:2]
        src = np.float32([[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(tex, M, (w_f, h_f), flags=cv2.INTER_NEAREST)
        mask = np.zeros((h_f, w_f), dtype=np.uint8)
        cv2.fillPoly(mask, [dst.astype(int)], 255)
        blended = cv2.addWeighted(frame, 1 - ALPHA, warped, ALPHA, 0)
        frame[mask > 0] = blended[mask > 0]

class AR3DApp(App):
    title = "ArUco 3D - Minecraft"

    def build(self):
        """Inicializa la app: widget de imagen, detector ArUco, hilo de cámara y timer de UI."""
        self.img_widget = KivyImage()
        self.textures = make_textures()
        self.dist = np.zeros(4)   # coeficientes de distorsión (sin calibración = cero)
        self.K = None             # matriz de cámara, se calcula al recibir el primer frame
        self.poses = {}           # {marker_id: [rvec, tvec, frames_sin_detectar]}
        self.kv_tex = None        # textura Kivy reutilizable (evita crearla cada frame)
        self._frame = None        # último frame procesado, compartido entre threads
        self._lock = threading.Lock()

        # Configura el detector ArUco con parámetros ajustados para reducir falsos positivos.
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod    = cv2.aruco.CORNER_REFINE_SUBPIX  # refina esquinas a subpíxel
        params.minMarkerPerimeterRate      = 0.05   # mínimo ~96px de perímetro en 1920px — descarta ruido pequeño
        params.adaptiveThreshWinSizeMin    = 7
        params.adaptiveThreshWinSizeMax    = 23
        params.adaptiveThreshWinSizeStep   = 8
        params.polygonalApproxAccuracyRate = 0.03
        params.errorCorrectionRate         = 0.6    # 0.9 era el causante principal de falsos positivos
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250), params
        )

        # El hilo de cámara corre separado para no bloquear la UI de Kivy.
        threading.Thread(target=self._capture_loop, daemon=True).start()
        # Clock.schedule_interval llama a update() 60 veces por segundo desde el hilo principal de Kivy.
        Clock.schedule_interval(self.update, 1 / 60)
        return self.img_widget

    def _capture_loop(self):
        """
        Hilo dedicado a la cámara: captura frames, detecta marcadores ArUco,
        calcula la pose con solvePnP y dibuja el cubo antes de pasar el frame a la UI.
        """
        # ponytail: cap debe abrirse en el mismo thread que lo lee (requisito AVFoundation/macOS)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: no se pudo abrir la cámara")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)  # ponytail: evita spin loop que ahorca el GIL mientras la cámara inicializa
                continue
            if self.K is None:
                h, w = frame.shape[:2]
                self.K = camera_matrix(w, h)

            # detectMarkers devuelve: lista de esquinas (una por marcador), IDs y marcadores rechazados.
            corners, ids, _ = self.detector.detectMarkers(frame)
            detected_ids = set()
            if ids is not None:
                for corner, mid in zip(corners, ids.flatten()):
                    if cv2.contourArea(corner[0]) < 400:  # descarta marcadores demasiado pequeños
                        continue
                    # solvePnP: dado puntos 3D del marcador y sus esquinas 2D en la imagen,
                    # calcula rvec (rotación) y tvec (traslación) de la cámara respecto al marcador.
                    ok, rvec, tvec = cv2.solvePnP(
                        MARKER_OBJ_PTS, corner[0], self.K, self.dist,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE,  # método optimizado para marcadores cuadrados
                    )
                    if ok:
                        self.poses[mid] = [rvec, tvec, 0]
                        detected_ids.add(mid)

            # Si un marcador no se detectó en este frame, incrementa su contador de ausencia.
            # A los 8 frames sin detectarlo, se elimina su pose (evita que el cubo quede "flotando").
            for mid in list(self.poses):
                if mid not in detected_ids:
                    self.poses[mid][2] += 1
                    if self.poses[mid][2] > 8:
                        del self.poses[mid]

            for rvec, tvec, _ in self.poses.values():
                # projectPoints: proyecta los vértices 3D del cubo al plano 2D de la imagen
                # usando la pose (rvec, tvec) y la matriz de cámara K.
                pts2d, _ = cv2.projectPoints(CUBE_VERTS, rvec, tvec, self.K, self.dist)
                pts2d = pts2d.reshape(-1, 2)
                # Rodrigues convierte el vector de rotación (rvec) a matriz de rotación 3x3.
                R, _ = cv2.Rodrigues(rvec)
                # Transforma los vértices al espacio de la cámara para calcular profundidad por cara.
                verts_cam = (R @ CUBE_VERTS.T + tvec).T
                draw_minecraft_cube(frame, pts2d, verts_cam, self.textures)

            # OpenCV usa BGR; Kivy espera RGB.
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Kivy tiene el origen (0,0) abajo a la izquierda; OpenCV arriba a la izquierda → flip vertical.
            frame = cv2.flip(frame, 0)
            with self._lock:
                self._frame = frame

    def update(self, dt):
        """Llamada 60 veces/seg por Kivy: copia el último frame y lo sube a la textura de la UI."""
        with self._lock:
            if self._frame is None:
                return
            frame = self._frame.copy()
        h, w, _ = frame.shape
        if self.kv_tex is None:
            self.kv_tex = Texture.create(size=(w, h))
        # blit_buffer: vuelca los bytes del array numpy directamente en la textura de GPU.
        self.kv_tex.blit_buffer(frame.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        self.img_widget.texture = self.kv_tex
        self.img_widget.canvas.ask_update()


if __name__ == "__main__":
    AR3DApp().run()

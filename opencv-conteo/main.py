# importaciones
import cv2
import glob
import os
import numpy as np
from ultralytics import YOLO

# LINK DEL VIDEO:
# https://www.pexels.com/video/busy-shopping-mall-escalator-scene-36108273/
# Lo descargas y lo metes así nomás en la carpeta donde tenés main.py y calibrar-roi.py y listo, no hace falta cambiar nada en el código.

# ACLARACIONES IMPORTANTES:
# El venv pesa bastante por Torch para poder usar el modelo YOLO (alrededor de 1.5 GB).
# En vez de hacer el conteo de los autos optamos por diferenciarnos y hacer un conteo en tiempo real
# en este caso de personas en una escalera mecánica.
# En una primera etapa usamos el detector HOG pero nos encontramos con problemas porque las personas estaban de perfil
# y no eran reconocidas como personas. Por eso optamos por usar un modelo YOLO preentrenado para detección de personas, que es mucho más robusto y preciso.
# calibrar-roi.py explicado abajo en los puntos pero es para calcular bien los rois (las regiones de interés)
# el archivo yolov8n.pt que seguro se te cree es el modelo que agarra de github en tiempo de ejecución (la primera vez) para usar

# Cargar el modelo YOLO para detección de personas
model = YOLO('yolov8n.pt')


# Buscar automáticamente el primer archivo .mp4 en la carpeta actual
# Esto es para que vos arrojes el video dentro de la carpeta donde main.py y ande sin problemas profe.
mp4_files = glob.glob(os.path.join(os.getcwd(), '*.mp4'))
if not mp4_files:
	print('No se encontró ningún archivo .mp4 en la carpeta actual')
	exit()
VIDEO_PATH = mp4_files[0]

# Abrir el video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
	print('No se pudo abrir el video')
	exit()

# Definir las regiones de interés (ROI) para las dos escaleras
# Redimensionamos el video en memoria para procesarlo MUCHO más rápido
ret, frame = cap.read()
if not ret:
	print('No se pudo leer el primer frame')
	exit()

# Redimensionar para acelerar el procesamiento y ver todo en pantalla
# Básicamente bajamos la calidad para poder procesar más rápido y que el video vaya más rápido.
original_height, original_width = frame.shape[:2]
max_height = 1080  # Aumentado a 1080 para mejorar la resolución y detección de personas
if original_height > max_height:
	scale = max_height / original_height
	frame = cv2.resize(frame, (int(original_width*scale), max_height))

height, width, _ = frame.shape


# Definir ROIs como trapecios inclinados (paralelos a las escaleras)
# Ajustados visualmente para la inclinación real (de arriba-derecha a abajo-izquierda)
roi1_pts = np.array([
    [int(width*0.29), int(height*0.82)],  # punto 1
    [int(width*1.0),  int(height*0.29)],  # punto 2
    [int(width*0.91), int(height*0.13)],  # punto 3
    [int(width*0.0),  int(height*0.71)],  # punto 4
], np.int32)

roi2_pts = np.array([
    [int(width*0.0),  int(height*0.66)],  # punto 1
    [int(width*0.86), int(height*0.12)],  # punto 2
    [int(width*0.77), int(height*0.08)],  # punto 3
    [int(width*0.0),  int(height*0.48)],  # punto 4
], np.int32)
#Para definir bien estas ROIS, corrimos calibrar.py por primera vez y marcamos los 4 puntos en sentido horario
#del primer rectangulo y luego del segundo y al salir, por consola nos dio las coordenadas exactas para pegar en main.py (ya habíamos renegado bastante para establecer las rois bien)

# Contadores
subiendo = 0
bajando = 0


# Función para dibujar ROI como polígono
def draw_roi_poly(img, pts, color, label):
	cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
	# Poner el label cerca del primer punto
	x, y = pts[2]
	cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Función para verificar si un punto está dentro de un polígono
def point_in_poly(point, poly):
	# Asegura que poly sea float32 y el punto también
	poly = poly.astype(np.float32)
	pt = (float(point[0]), float(point[1]))
	return cv2.pointPolygonTest(poly, pt, False) >= 0

# Bucle principal
while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Redimensionar el frame igual que al inicio para mantener consistencia y velocidad
	if original_height > max_height:
		frame = cv2.resize(frame, (width, height))

	# Detección de personas usando YOLO (clase 0)
	results = model(frame, classes=[0], conf=0.35, verbose=False)

	# Dibujar ROIs como polígonos inclinados
	draw_roi_poly(frame, roi1_pts, (255,0,0), 'Escalera 1 (baja)')
	draw_roi_poly(frame, roi2_pts, (0,255,0), 'Escalera 2 (sube)')

	# Contadores temporales para este frame
	subiendo_frame = 0
	bajando_frame = 0

	# Analizar cada persona detectada
	for box in results[0].boxes:
		x1, y1, x2, y2 = map(int, box.xyxy[0])
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
		color = (0,0,255)
		# Verificar en qué polígono está el centroide
		if point_in_poly((cx, cy), roi1_pts):
			bajando_frame += 1
			color = (255,0,0)
		elif point_in_poly((cx, cy), roi2_pts):
			subiendo_frame += 1
			color = (0,255,0)
		cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
		cv2.circle(frame, (cx, cy), 5, color, -1)


	# Mostrar conteo en pantalla
	cv2.putText(frame, f'Subiendo: {subiendo_frame}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
	cv2.putText(frame, f'Bajando: {bajando_frame}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

	cv2.imshow('Conteo de personas', frame)
	key = cv2.waitKey(1) & 0xFF  # Menor delay para mayor velocidad
	if key == 27:  # ESC para salir
		break

cap.release()
cv2.destroyAllWindows()

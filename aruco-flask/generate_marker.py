import cv2
import cv2.aruco as aruco

# Generar un marcador ArUco para imprimir o mostrar en pantalla
# Usamos el diccionario 6X6_250 que tienes configurado en el script principal
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
# Generamos el ID 0
marker_img = aruco.generateImageMarker(aruco_dict, 0, 400)

cv2.imwrite("marker_6x6_id0.png", marker_img)
print("Se ha guardado 'marker_6x6_id0.png'.")
print("IMPORTANTE: Debes usar un marcador del diccionario 6x6 para que aruco-qr.py lo detecte.")

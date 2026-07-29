import cv2
import cv2.aruco as aruco
import numpy as np
import os

def main():
    # --- CONFIGURACIÓN ---
    # Ruta de la imagen que quieres proyectar. 
    # Asegúrate de poner la imagen en una carpeta 'assets' o ajustar esta ruta.
    # Por defecto buscaré una imagen en el workspace si existe.
    image_path = os.path.join(os.path.dirname(__file__), '..', 'practica-kivy', 'assets', 'cat-thinking.jpeg')
    
    if not os.path.exists(image_path):
        # Si no existe, intentamos buscar cualquier imagen en la carpeta assets de practica-kivy
        print(f"Advertencia: No se encontró {image_path}. Verificando alternativas...")
        image_path = "/home/pepi/Documents/Facultad/4to/Soporte/practica-kivy/assets/calculadora.png"

    overlay_img = cv2.imread(image_path)
    
    if overlay_img is None:
        print(f"Error: No se pudo cargar ninguna imagen. Asegúrate de que la ruta sea correcta.")
        return

    # 1. Configurar el diccionario ArUco
    try:
        # Usamos 6x6_250 como pediste
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        
        # --- AJUSTES PARA MEJORAR DETECCIÓN Y EVITAR LUCES ---
        parameters.adaptiveThreshWinSizeMin = 3
        parameters.adaptiveThreshWinSizeMax = 23
        parameters.adaptiveThreshWinSizeStep = 10
        parameters.adaptiveThreshConstant = 7
        
        # Los fluorescentes son rectangulares, ArUco es cuadrado.
        # Bajamos este valor para ser más estrictos con la forma cuadrada.
        parameters.polygonalApproxAccuracyRate = 0.03 
        parameters.minMarkerPerimeterRate = 0.1

        # Detector
        detector = aruco.ArucoDetector(aruco_dict, parameters)
    except AttributeError:
        # Fallback para versiones antiguas de OpenCV
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters_create()
        detector = None 

    # 2. Iniciar captura de video
    cap = cv2.VideoCapture(0)

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Espejar la imagen para que sea más natural (opcional)
        frame = cv2.flip(frame, 1)

        # 3. Detectar marcadores
        if detector:
            corners, ids, rejected = detector.detectMarkers(frame)
        else:
            # Fallback para versiones antiguas
            corners, ids, rejected = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)

        if ids is not None:
            # Filtro opcional: Solo procesar si el ID es bajo (por si el ruido genera IDs aleatorios altos)
            # o si el área del marcador es razonable.
            for i in range(len(ids)):
                # Dibujar ID detectado para saber qué está viendo
                cv2.putText(frame, f"ID: {ids[i][0]}", 
                            tuple(corners[i][0][0].astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Obtener las esquinas del marcador actual
                # marker_corners tiene forma (1, 4, 2)
                marker_corners = corners[i][0]
                
                # Los puntos de destino son las esquinas del ArUco detectado
                dst_pts = marker_corners.astype(np.float32)

                # Definir los puntos de origen (las esquinas de la imagen a proyectar)
                h, w = overlay_img.shape[:2]
                src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

                # 4. Calcular la transformación de perspectiva
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

                # 5. Transformar la imagen para que encaje
                warped_img = cv2.warpPerspective(overlay_img, matrix, (frame.shape[1], frame.shape[0]))

                # 6. Superponer la imagen usando una máscara
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, dst_pts.astype(np.int32), 255)

                mask_inv = cv2.bitwise_not(mask)

                # Fondo: el frame original sin el área del marcador
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

                # Frente: la imagen transformada recortada con la máscara
                frame_fg = cv2.bitwise_and(warped_img, warped_img, mask=mask)

                # Combinación final
                frame = cv2.add(frame_bg, frame_fg)

        cv2.imshow("Realidad Aumentada ArUco", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

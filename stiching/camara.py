import cv2
from datetime import datetime
from collections import deque
import threading
import time
import os

cap = cv2.VideoCapture(0)

# Obtener propiedades de la cámara
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30  # Forzar 30 fps (las webcams web suelen devolver 0 o valores bajos)

# Buffer para los últimos 5 segundos (150 frames a 30 fps)
buffer_seconds = 5
buffer_frames = deque(maxlen=buffer_seconds * fps)

recording = False
frames_buffer = []
countdown_active = False
countdown_start_time = 0

def reproducir_video(filename):
    """Reproduce un video en una ventana separada"""
    cap_video = cv2.VideoCapture(filename)
    while True:
        ret, frame = cap_video.read()
        if not ret:
            break
        cv2.imshow("Reproduciendo", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break
    cap_video.release()
    cv2.destroyWindow("Reproduciendo")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Espejar horizontalmente para reflejar la orientación verdadera de la cámara
    frame = cv2.flip(frame, 1)
    
    # Agregar frame al buffer de últimos 5 segundos
    buffer_frames.append(frame.copy())
    
    # Mostrar estado de grabación
    if recording:
        cv2.putText(frame, "GRABANDO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frames_buffer.append(frame.copy())
    
    # Mostrar y controlar el countdown para foto
    if countdown_active:
        elapsed = time.time() - countdown_start_time
        remaining = max(0, 5 - int(elapsed))
        
        # Mostrar el contador en pantalla
        cv2.putText(frame, f"FOTO EN: {remaining}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # Si el countdown terminó, tomar la foto
        if remaining == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            photo_filename = f"foto_{timestamp}.jpg"
            cv2.imwrite(photo_filename, frame)
            print(f"Foto guardada: {photo_filename}")
            countdown_active = False
    
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):  # Presionar 'r' para iniciar/detener grabación
        if not recording:
            recording = True
            frames_buffer = []
            print("Grabación iniciada...")
        else:
            recording = False
            # Guardar el video
            if frames_buffer:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"videoultimo_{timestamp}.mp4"
                
                out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
                for f in frames_buffer:
                    out.write(f)
                out.release()
                print(f"Video guardado: {filename}")
                frames_buffer = []
    
    elif key == ord('d'):  # Presionar 'd' para guardar últimos 5 segundos
        if buffer_frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ultimos_5seg_{timestamp}.mp4"
            
            out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            for f in buffer_frames:
                out.write(f)
            out.release()
            print(f"Video de últimos 5 segundos guardado: {filename}")
            
            # Reproducir el video en un hilo separado
            thread = threading.Thread(target=reproducir_video, args=(filename,))
            thread.daemon = True
            thread.start()
    
    elif key == ord('p'):  # Presionar 'p' para iniciar countdown y tomar foto
        countdown_active = True
        countdown_start_time = time.time()
        print("Countdown iniciado... Foto en 5 segundos")
    
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

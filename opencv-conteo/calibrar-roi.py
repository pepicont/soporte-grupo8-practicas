# calibrar_roi.py - Corré esto una sola vez para obtener los puntos exactos
import cv2
import numpy as np
import glob, os

mp4_files = glob.glob(os.path.join(os.getcwd(), '*.mp4'))
cap = cv2.VideoCapture(mp4_files[0])
ret, frame = cap.read()
cap.release()

original_height, original_width = frame.shape[:2]
max_height = 1080
if original_height > max_height:
    scale = max_height / original_height
    frame = cv2.resize(frame, (int(original_width * scale), max_height))

height, width = frame.shape[:2]

points = []
current_roi = [1]

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        idx = len(points)
        cv2.putText(frame, str(idx), (x+8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow('Calibrar ROI', frame)

        if len(points) == 4:
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
            cv2.imshow('Calibrar ROI', frame)
            print(f"\n# ROI {current_roi[0]} - copiá esto en main.py:")
            for i, (px, py) in enumerate(points):
                fx = round(px / width, 2)
                fy = round(py / height, 2)
                print(f"    [int(width*{fx}), int(height*{fy})],  # punto {i+1}")
            points.clear()
            current_roi[0] += 1
            print(f"\nAhora hacé click en los 4 vértices del ROI {current_roi[0]} (o ESC para salir)")

cv2.imshow('Calibrar ROI', frame)
cv2.setMouseCallback('Calibrar ROI', click_event)
print("Hacé click en los 4 vértices del ROI 1 (en orden: sup-izq, sup-der, inf-der, inf-izq)")
cv2.waitKey(0)
cv2.destroyAllWindows()
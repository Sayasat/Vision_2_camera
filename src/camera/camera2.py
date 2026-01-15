import cv2
import time
import numpy as np
from driver import Gemini335LECamera

cam1 = Gemini335LECamera(serial_number="CPE345P0008A")
    # Камера 2
cam2 = Gemini335LECamera(serial_number="CPE345P0004A")

cam1.start()
cam2.start()

print("Камеры запущены. Нажмите 'q' для выхода.")

try:
    while True:
        color1, _ = cam1.get_frames()
        color2, _ = cam2.get_frames()

        if color1 is not None:
            cv2.imshow("Camera 0008A", color1)
        if color2 is not None:
            cv2.imshow("Camera 0004A", color2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam1.stop()
    cam2.stop()
    cv2.destroyAllWindows()
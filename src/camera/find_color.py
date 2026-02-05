import cv2
import numpy as np
from driver import Gemini335LECamera

# Параметры окна
window_name = "Box Color Detector"
cv2.namedWindow(window_name)

# Переменные для хранения результатов
low_bound = np.array([0, 0, 0])
high_bound = np.array([0, 0, 0])
selected = False

def pick_color(event, x, y, flags, param):
    global low_bound, high_bound, selected
    if event == cv2.EVENT_LBUTTONDOWN:
        frame_hsv = param
        hsv_pixel = frame_hsv[y, x]
        
        # Создаем диапазон: Hue +- 10, Saturation +- 40, Value +- 50
        # Ограничиваем значения в пределах [0-180] для H и [0-255] для S, V
        h, s, v = hsv_pixel
        
        low_bound = np.array([max(0, h-10), max(50, s-50), max(50, v-50)])
        high_bound = np.array([min(180, h+10), 255, 255])
        
        selected = True
        print(f"\n--- Настройки для cv2.inRange ---")
        print(f"lower = np.array([{low_bound[0]}, {low_bound[1]}, {low_bound[2]}])")
        print(f"upper = np.array([{high_bound[0]}, {high_bound[1]}, {high_bound[2]}])")

cam1 = Gemini335LECamera(serial_number="CPE345P0004A")
cam1.start()

try:
    while True:
        frame, _ = cam1.get_frames()
        if frame is None: continue

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cv2.setMouseCallback(window_name, pick_color, param=hsv)

        if selected:
            # Показываем маску, чтобы проверить, как работает фильтр
            mask = cv2.inRange(hsv, low_bound, high_bound)
            # Выводим значения на экран
            cv2.putText(frame, f"Low: {low_bound}", (10, 30), 1, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"High: {high_bound}", (10, 60), 1, 1, (0, 255, 0), 2)
            cv2.imshow("Mask (Result)", mask)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam1.stop()
    cv2.destroyAllWindows()
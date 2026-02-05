# import numpy as np
# import cv2
# from pyorbbecsdk import Pipeline, OBFormat, PointCloudFilter, Context, Config
# from utils import frame_to_bgr_image
# import json
# import os
# import time

# class Gemini335LECamera:

#     def __init__(self, serial_number=None, min_depth=20, max_depth=10000, log_file="picked_items.json"):
#         self.ctx = Context()
#         self.pipeline = None
#         self.min_depth = min_depth
#         self.max_depth = max_depth
#         self.log_file = log_file
        
#         # Поиск устройства по серийному номеру
#         device_list = self.ctx.query_devices()
#         found_device = None
        
#         if serial_number:
#             for i in range(device_list.get_count()):
#                 dev = device_list.get_device_by_index(i)
#                 if dev.get_device_info().get_serial_number() == serial_number:
#                     found_device = dev
#                     break
            
#             if found_device:
#                 self.pipeline = Pipeline(found_device)
#                 print(f"Подключено к камере SN: {serial_number}")
#             else:
#                 raise RuntimeError(f"Камера с серийным номером {serial_number} не найдена!")
#         else:
#             # Если SN не указан, берем первое доступное устройство
#             self.pipeline = Pipeline()
#             print("SN не указан, подключена первая доступная камера.")

#         self.pointcloud_filter = PointCloudFilter()
#         self.pointcloud_filter.set_create_point_format(OBFormat.POINT)
#     # def __init__(self, min_depth=20, max_depth=10000, log_file="picked_items.json"):
#     #     self.pipeline = Pipeline()
#     #     self.min_depth = min_depth
#     #     self.max_depth = max_depth
#     #     self.log_file = log_file
#     #     self.pointcloud_filter = PointCloudFilter()
#     #     self.pointcloud_filter.set_create_point_format(OBFormat.POINT)

#     def start(self):
#         self.pipeline.start()
        
#     def stop(self):
#         self.pipeline.stop()

#     def get_frames(self):
#         frames = self.pipeline.wait_for_frames(100)
#         if frames is None:
#             return None, None

#         # Color frame
#         color_frame = frames.get_color_frame()
#         if color_frame is None:
#             return None, None
#         color_image = frame_to_bgr_image(color_frame)

#         # Depth frame
#         depth_frame = frames.get_depth_frame()
#         if depth_frame is None or depth_frame.get_format() != OBFormat.Y16:
#             return color_image, None

#         width = depth_frame.get_width()
#         height = depth_frame.get_height()
#         scale = depth_frame.get_depth_scale()

#         raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
#         depth_data = raw_depth.astype(np.float32) * scale
#         depth_filtered = np.where((depth_data > self.min_depth) & (depth_data < self.max_depth), depth_data, 0)

#         return color_image, depth_filtered.astype(np.uint16)
    
#     def get_pointcloud(self):
#         frames = self.pipeline.wait_for_frames(100)
#         if frames is None:
#             return None, None

#         color_frame = frames.get_color_frame()
#         depth_frame = frames.get_depth_frame()

#         if color_frame is None or depth_frame is None:
#             return None, None

#         color_image = frame_to_bgr_image(color_frame)

#         pointcloud_frame = self.pointcloud_filter.process(depth_frame)
#         if pointcloud_frame is None:
#             return color_image, None

#         width = depth_frame.get_width()
#         height = depth_frame.get_height()
#         pc_data = np.frombuffer(pointcloud_frame.get_data(), dtype=np.float32)
#         pc_data = pc_data.reshape((height, width, 3))

#         return color_image, pc_data
    
#     def get_box_focus(self, color_image, depth_data):
#         if color_image is None or depth_data is None:
#             return None

#         # 1. Сглаживаем изображение, чтобы убрать цветовой шум
#         blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
#         hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

#         # 2. Настройка порогов (Зеленый цвет)
#         # H (Hue): 35-85 — диапазон зеленого
#         # S (Saturation): поднят до 100, чтобы игнорировать серые/белесые тени
#         # V (Value): поднят до 50, чтобы игнорировать слишком темные участки (глубокие тени)
#         lower_green = np.array([35, 100, 50]) 
#         upper_green = np.array([90, 255, 255])

#         # 3. Создание маски
#         mask = cv2.inRange(hsv, lower_green, upper_green)

#         # 4. Морфологическая очистка (убираем мелкие точки и соединяем части ящика)
#         kernel = np.ones((5, 5), np.uint8)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
#         mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)

#         # 5. Поиск контуров
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         best_box = None
#         max_area = 0

#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < 1000: # Игнорируем слишком мелкие объекты
#                 continue

#             # Проверка формы: ящик обычно похож на прямоугольник
#             # Это дополнительно отсекает бесформенные пятна/тени
#             peri = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            
#             # Если у объекта от 4 до 8 углов — это скорее всего наш ящик
#             if 4 <= len(approx) <= 8:
#                 if area > max_area:
#                     max_area = area
#                     best_box = cnt

#         if best_box is not None:
#             M = cv2.moments(best_box)
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])
                
#                 # Получаем глубину и фильтруем нулевые значения
#                 dist = depth_data[cY, cX]
#                 if dist == 0:
#                     # Если попали в "дырку" в данных, смотрим среднее в радиусе 5 пикселей
#                     roi = depth_data[max(0, cY-5):cY+5, max(0, cX-5):cX+5]
#                     dist = np.mean(roi[roi > 0]) if np.any(roi > 0) else 0

#                 return {
#                     "center": (cX, cY),
#                     "distance_mm": dist,
#                     "contour": best_box
#                 }
        
#         return None
    
#     def find_objects_inside_box(self, color_image, depth_data, box_contour):
#         # 1. Создаем маску ящика
#         mask_box = np.zeros(color_image.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask_box, [box_contour], -1, 255, -1)

#         # 2. СРЕЗАЕМ КРАЯ (Эрозия)
#         # Увеличивай (30, 30), если хочешь срезать еще больше от бортов
#         kernel = np.ones((30, 60), np.uint8)
#         eroded_mask = cv2.erode(mask_box, kernel, iterations=1)
        
#         # Получаем контур новой "сжатой" зоны для рисовки
#         inner_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         safe_zone_contour = inner_contours[0] if inner_contours else None

#         # 3. Фильтр глубины (Z)
#         box_depths = depth_data[eroded_mask > 0]
#         valid_z = box_depths[box_depths > 0]
#         z_limit = np.median(valid_z) + 15 if valid_z.size else 700

#         # # 4. Фильтр цвета (НЕ зеленый)
#         # hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
#         # green_mask = cv2.inRange(hsv, (35, 50, 30), (105, 255, 255))
        
#         # # 5. Итоговая маска
#         # final_mask = cv2.bitwise_and(cv2.bitwise_not(green_mask), eroded_mask)
#         # depth_mask = cv2.inRange(depth_data.astype(np.float32), 10, z_limit)
#         # final_mask = cv2.bitwise_and(final_mask, depth_mask.astype(np.uint8))

#         hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
#     # Захватываем красный с обоих концов шкалы разом (от 170 до 10 через 180)
#         # В OpenCV проще сделать два диапазона и объединить
#         m1 = cv2.inRange(hsv, (0, 40, 40), (10, 255, 255))
#         m2 = cv2.inRange(hsv, (170, 40, 40), (180, 255, 255))
#         red_mask = cv2.bitwise_or(m1, m2)
        
#         # 4. Итоговая маска: (НЕ красный) И (внутри ящика) И (выше дна)
#         depth_mask = cv2.inRange(depth_data.astype(np.float32), 10, z_limit)
#         final_mask = cv2.bitwise_and(cv2.bitwise_not(red_mask), eroded_mask)
#         final_mask = cv2.bitwise_and(final_mask, depth_mask.astype(np.uint8))
            
#         # Поиск объектов
#         obj_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         # detected = [c for c in obj_contours if cv2.contourArea(c) > 300]
#         detected_data = []
#         for cnt in obj_contours:
#             if cv2.contourArea(cnt) > 300:
#                 center = self.get_object_center(cnt)
#                 if center:
#                     # cX, cY = center
#                     # # --- ВОТ ЭТО НУЖНО ДОБАВИТЬ ---
#                     # obj_z = depth_data[cY, cX]
#                     detected_data.append({
#                         "contour": cnt,
#                         "center": center,
#                         # "z_mm": obj_z
#                     })

#         return detected_data, safe_zone_contour
    
#     def get_object_center(self, contour):
#         """Вычисляет центр контура через моменты."""
#         M = cv2.moments(contour)
#         if M["m00"] != 0:
#             cX = int(M["m10"] / M["m00"])
#             cY = int(M["m01"] / M["m00"])
#             return (cX, cY)
#         return None
    

#     def set_calibration_params(self):
#         # ТВОЯ БАЗОВАЯ ТОЧКА И МАТРИЦА
#         self.cam_ref = np.array([311.867, 25.709, 695.000])
#         self.rob_ref = np.array([168.0, -295.0, -298.0]) 

#         self.R = np.eye(3)
#         self.R[1, 1] = -1  # Инверсия Y
#         self.R[2, 2] = -1  # Инверсия Z
        
#         self.t = self.rob_ref - (self.R @ self.cam_ref)
        
#         # ЗАГРУЗКА ТАБЛИЦЫ ОШИБОК (твои 12 точек)
#         # self.error_map = [
#         #     {"c": [276.2, 55.2, 648.0], "rz": -251.0},
#         #     {"c": [133.0, 161.9, 641.0], "rz": -244.0},
#         #     {"c": [429.9, 169.3, 638.0], "rz": -241.0},
#         #     {"c": [440.6, -98.3, 657.0], "rz": -260.0},
#         #     {"c": [286.8, -98.4, 658.0], "rz": -261.0},
#         #     {"c": [280.6, 165.5, 639.0], "rz": -242.0},
#         #     {"c": [177.4, 79.9, 645.0], "rz": -248.0},
#         #     {"c": [382.2, -11.5, 651.0], "rz": -254.0},
#         #     {"c": [344.9, 77.7, 644.0], "rz": -247.0},
#         #     {"c": [510.4, 30.3, 649.0], "rz": -252.0},
#         #     {"c": [472.9, 44.9, 649.0], "rz": -252.0},
#         #     {"c": [153.7, 28.2, 650.0], "rz": -253.0}
#         # ]
#         self.error_map = [
#             {"c": [260.558, 80.531, 648.0], "rz": -263.437},
#             {"c": [250.911, 189.689, 638.0], "rz": -261.373},
#             {"c": [261.243, -9.33, 659.0], "rz": -264.247},
#             {"c": [418.819, -15.55, 659.0], "rz": -263.681},
#             {"c": [421.37, 88.549, 647.0], "rz": -261.166},
#             {"c": [424.211, 183.091, 636.0], "rz": -258.78},
#             {"c": [139.277, 87.048, 651.0], "rz": -265.519},
#             {"c": [135.382, -9.358, 662.0], "rz": -267.869},
#             {"c": [121.381, 166.899, 643.0], "rz": -264.288}
#         ]

        
#         print(f"Калибровка и карта ошибок загружены.")

#     def get_local_z_correction(self, cx, cy):
#         """Находит, какая ошибка Z была в ближайшей точке из списка"""
#         dists = []
#         for p in self.error_map:
#             d = np.sqrt((cx - p['c'][0])**2 + (cy - p['c'][1])**2)
#             dists.append(d)
        
#         idx = np.argmin(dists)
#         closest_point = self.error_map[idx]
        
#         # Вычисляем, какой Z выдал бы базовый алгоритм (без правок) для этой точки
#         base_cam = np.array(closest_point['c'])
#         base_rob = (self.R @ base_cam) + self.t
        
#         # Ошибка = (Тот кривой Z из списка) - (-255.0)
#         error = closest_point['rz'] - (-255.0)
        
#         return error

#     def camera_to_robot_xyz(self, cx, cy, cz):
#         if not hasattr(self, 't'):
#             self.set_calibration_params()

#         if cz <= 0:
#             # Если Z <= 0, возвращаем базовую точку с нормализованным значением
#             return np.array([self.rob_ref[0], self.rob_ref[1], cz])

#         # 1. Стандартный расчет
#         cam_pt = np.array([cx, cy, cz], dtype=float)
#         rob_pt = (self.R @ cam_pt) + self.t
        
#         # 2. Динамическая коррекция
#         z_error = self.get_local_z_correction(cx, cy)
        
#         # Вычитаем ошибку из Z, теперь без ограничения на -255
#         rob_pt[2] = rob_pt[2] - z_error

#         # Теперь мы не ограничиваем Z жестко
#         return rob_pt
    
#     def log_picked_object(self, robot_coords):
#         """
#         Записывает координаты захваченного SKU в JSON файл.
#         robot_coords: массив или список [x, y, z]
#         """
#         history = []
        
#         # 1. Читаем существующие данные, если файл есть
#         if os.path.exists(self.log_file):
#             try:
#                 with open(self.log_file, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
#             except (json.JSONDecodeError, IOError):
#                 history = []

#         # 2. Формируем запись
#         new_entry = {
#             "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#             "robot_x": round(float(robot_coords[0]), 2),
#             "robot_y": round(float(robot_coords[1]), 2),
#             "robot_z": round(float(robot_coords[2]), 2)
#         }
        
#         # 3. Добавляем и сохраняем
#         history.append(new_entry)
        
#         try:
#             with open(self.log_file, 'w', encoding='utf-8') as f:
#                 json.dump(history, f, indent=4, ensure_ascii=False)
#             print(f"[LOG] Данные SKU сохранены в {self.log_file}")
#         except IOError as e:
#             print(f"[ERROR] Не удалось записать в файл: {e}")

#     def log_dimensions_only(self, length, width, filename="object_dimensions.json"):
#         """Записывает только габариты объекта в отдельный JSON файл."""
#         new_data = {
#             "timestamp": time.strftime("%H:%M:%S"),
#             "length_mm": length,
#             "width_mm": width
#         }
        
#         history = []
#         if os.path.exists(filename):
#             try:
#                 with open(filename, 'r', encoding='utf-8') as f:
#                     history = json.load(f)
#             except:
#                 history = []

#         history.append(new_data)

#         with open(filename, 'w', encoding='utf-8') as f:
#             json.dump(history, f, indent=4, ensure_ascii=False)

#     def get_object_dimensions(self, contour, pc_data):
#         """
#         Вычисляет физическую длину и ширину объекта в мм.
#         """
#         # 1. Получаем ориентированный прямоугольник (MinAreaRect)
#         # Это позволит измерить объект, даже если он повернут
#         rect = cv2.minAreaRect(contour)
#         box = cv2.boxPoints(rect)
#         box = np.int0(box)

#         # 2. Берем характерные точки (ширина и высота в пикселях)
#         # p0-p1 (ширина), p1-p2 (длина)
#         p0, p1, p2 = box[0], box[1], box[2]

#         # 3. Извлекаем 3D координаты этих точек из облака точек
#         # Важно: используем [y, x] для обращения к матрице pc_data
#         pt0_3d = pc_data[p0[1], p0[0]]
#         pt1_3d = pc_data[p1[1], p1[0]]
#         pt2_3d = pc_data[p2[1], p2[0]]

#         # Проверяем, что данные о глубине валидны (не нули)
#         if np.all(pt0_3d) and np.all(pt1_3d) and np.all(pt2_3d):
#             # Вычисляем евклидово расстояние между точками в мм
#             width_mm = np.linalg.norm(pt0_3d - pt1_3d)
#             length_mm = np.linalg.norm(pt1_3d - pt2_3d)
            
#             return round(float(length_mm), 1), round(float(width_mm), 1)
        
#         return 0, 0
    
    
# # ✅ Пример использования при запуске как main
# if __name__ == "__main__":
#     cam = Gemini335LECamera()
#     cam.start()
#     print("Press 'q' to exit.")

#     try:
#         while True:
#             color, depth = cam.get_frames()
#             if color is not None:
#                 cv2.imshow("Color", color)
#             if depth is not None:
#                 depth_vis = cv2.convertScaleAbs(depth, alpha=0.03)
#                 depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
#                 cv2.imshow("Depth", depth_vis)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#     finally:
#         cam.stop()
#         cv2.destroyAllWindows()
#         print("Camera stopped.")




# import numpy as np
# import cv2
# from pyorbbecsdk import Pipeline, OBFormat, PointCloudFilter, Context, Config
# from utils import frame_to_bgr_image
# import json
# import os
# import time

# class Gemini335LECamera:
#     def __init__(self, serial_number=None, min_depth=20, max_depth=10000, log_file="picked_items.json"):
#         self.ctx = Context()
#         self.pipeline = None
#         self.min_depth = min_depth
#         self.max_depth = max_depth
#         self.log_file = log_file
        
#         device_list = self.ctx.query_devices()
#         found_device = None
        
#         if serial_number:
#             for i in range(device_list.get_count()):
#                 dev = device_list.get_device_by_index(i)
#                 if dev.get_device_info().get_serial_number() == serial_number:
#                     found_device = dev
#                     break
            
#             if found_device:
#                 self.pipeline = Pipeline(found_device)
#                 print(f"Подключено к камере SN: {serial_number}")
#             else:
#                 raise RuntimeError(f"Камера с серийным номером {serial_number} не найдена!")
#         else:
#             self.pipeline = Pipeline()
#             print("SN не указан, подключена первая доступная камера.")

#         self.pointcloud_filter = PointCloudFilter()
#         self.pointcloud_filter.set_create_point_format(OBFormat.POINT)

#     def start(self):
#         self.pipeline.start()
        
#     def stop(self):
#         self.pipeline.stop()

#     def get_frames(self):
#         frames = self.pipeline.wait_for_frames(100)
#         if frames is None: return None, None
#         color_frame = frames.get_color_frame()
#         if color_frame is None: return None, None
#         color_image = frame_to_bgr_image(color_frame)
#         depth_frame = frames.get_depth_frame()
#         if depth_frame is None: return color_image, None

#         width, height = depth_frame.get_width(), depth_frame.get_height()
#         scale = depth_frame.get_depth_scale()
#         raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
#         depth_data = raw_depth.astype(np.float32) * scale
#         depth_filtered = np.where((depth_data > self.min_depth) & (depth_data < self.max_depth), depth_data, 0)
#         return color_image, depth_filtered.astype(np.uint16)
    
#     def get_pointcloud(self):
#         frames = self.pipeline.wait_for_frames(100)
#         if frames is None: return None, None
#         color_frame = frames.get_color_frame()
#         depth_frame = frames.get_depth_frame()
#         if color_frame is None or depth_frame is None: return None, None
#         color_image = frame_to_bgr_image(color_frame)
#         pointcloud_frame = self.pointcloud_filter.process(depth_frame)
#         if pointcloud_frame is None: return color_image, None
#         pc_data = np.frombuffer(pointcloud_frame.get_data(), dtype=np.float32).reshape((depth_frame.get_height(), depth_frame.get_width(), 3))
#         return color_image, pc_data

#     def set_calibration_params(self):
#         # Твои новые 10 пар точек
#         data = {
#             "pairs": [
#                 { "cam": [419.847, 188.277, 626.000], "rob": [285, -400, -264.048] },
#                 { "cam": [289.883, 180.798, 628.000], "rob": [149.757, -396.850, -264.770] },
#                 { "cam": [309.138, -11.606, 656.000], "rob": [151.506, -193.225, -264.063] },
#                 { "cam": [164.689, 96.327, 644.000], "rob": [13.231, -307.941, -264.074] },
#                 { "cam": [155.521, 179.916, 632.000], "rob": [13.143, -397.705, -264.979] },
#                 { "cam": [320.126, 97.788, 640.000], "rob": [177.628, -307.447, -264.382] },
#                 { "cam": [456.627, 97.482, 638.000], "rob": [320.336, -302.666, -264.800] },
#                 { "cam": [468.940, 97.482, 638.000], "rob": [335.215, -299.580, -264.032] },
#                 { "cam": [466.647, -7.341, 652.000], "rob": [316.485, -190.202, -264.566] },
#                 { "cam": [445.728, 38.443, 646.000], "rob": [305.573, -240.578, -264.633] },
#                 { "cam": [393.821, 113.543, 636.000], "rob": [254.124, -322.909, -264.828] },
#                 { "cam": [322.540, 30.271, 649.000], "rob": [168.304, -232.418, -264.520] },
#                 { "cam": [346.767, 129.910, 636.000], "rob": [208.674, -341.180, -265.371] },
#                 { "cam": [206.900, 26.256, 653.000], "rob": [48.252, -229.346, -264.761] },
#                 { "cam": [176.134, 65.659, 648.000], "rob": [23.737, -275.445, -264.780] },
#                 { "cam": [156.259, 160.344, 635.000], "rob": [11.737, -379.294, -264.926] },
#                 { "cam": [168.269, -3.175, 658.000], "rob": [3.182, -203.232, -264.232] }
#             ]
#         }
#         cam_pts = np.array([p['cam'] for p in data['pairs']])
#         rob_pts = np.array([p['rob'] for p in data['pairs']])

#         # Центрирование
#         c_cam = np.mean(cam_pts, axis=0)
#         c_rob = np.mean(rob_pts, axis=0)
        
#         # Расчет матрицы вращения через SVD (Singular Value Decomposition)
#         H = (cam_pts - c_cam).T @ (rob_pts - c_rob)
#         U, S, Vt = np.linalg.svd(H)
#         self.R = Vt.T @ U.T

#         if np.linalg.det(self.R) < 0:
#             Vt[2, :] *= -1
#             self.R = Vt.T @ U.T

#         # Смещение
#         self.t = c_rob - self.R @ c_cam
#         print("Калибровка по SVD успешно завершена.")

#     def camera_to_robot_xyz(self, cx, cy, cz):
#         if not hasattr(self, 'R'): self.set_calibration_params()
#         cam_pt = np.array([cx, cy, cz], dtype=float)
#         return self.R @ cam_pt + self.t

#     def get_box_focus(self, color_image, depth_data):
#         blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
#         hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#         mask = cv2.inRange(hsv, np.array([35, 100, 50]), np.array([90, 255, 255]))
#         mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         best_box, max_area = None, 0
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area > 1000:
#                 approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
#                 if 4 <= len(approx) <= 8 and area > max_area:
#                     max_area, best_box = area, cnt
#         if best_box is not None:
#             M = cv2.moments(best_box)
#             if M["m00"] != 0:
#                 cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
#                 return {"center": (cX, cY), "distance_mm": depth_data[cY, cX], "contour": best_box}
#         return None

#     def find_objects_inside_box(self, color_image, depth_data, box_contour):
#         mask_box = np.zeros(color_image.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask_box, [box_contour], -1, 255, -1)
#         eroded_mask = cv2.erode(mask_box, np.ones((30, 60), np.uint8), iterations=1)
#         hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
#         green_mask = cv2.inRange(hsv, (35, 50, 30), (105, 255, 255))
#         final_mask = cv2.bitwise_and(cv2.bitwise_not(green_mask), eroded_mask)
#         obj_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         detected = []
#         for cnt in obj_contours:
#             if cv2.contourArea(cnt) > 300:
#                 M = cv2.moments(cnt)
#                 if M["m00"] != 0:
#                     detected.append({"contour": cnt, "center": (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))})
#         return detected, None

#     def get_object_dimensions(self, contour, pc_data):
#         rect = cv2.minAreaRect(contour)
#         box = np.int0(cv2.boxPoints(rect))
#         pts = [pc_data[p[1], p[0]] for p in box[:3]]
#         if all(np.any(p) for p in pts):
#             return round(float(np.linalg.norm(pts[1]-pts[2])), 1), round(float(np.linalg.norm(pts[0]-pts[1])), 1)
#         return 0, 0
    
#     def log_picked_object(self, robot_coords):
#         entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "x": round(robot_coords[0], 2), "y": round(robot_coords[1], 2), "z": round(robot_coords[2], 2)}
#         history = []
#         if os.path.exists(self.log_file):
#             try:
#                 with open(self.log_file, 'r') as f: history = json.load(f)
#             except: history = []
#         history.append(entry)
#         with open(self.log_file, 'w') as f: json.dump(history, f, indent=4)








import numpy as np
import cv2
from pyorbbecsdk import Pipeline, OBFormat, PointCloudFilter, Context, Config
from utils import frame_to_bgr_image
import json
import os
import time

class Gemini335LECamera:
    def __init__(self, serial_number=None, min_depth=20, max_depth=10000, log_file="picked_items.json"):
        self.ctx = Context()
        self.pipeline = None
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.log_file = log_file
        
        device_list = self.ctx.query_devices()
        found_device = None
        
        if serial_number:
            for i in range(device_list.get_count()):
                dev = device_list.get_device_by_index(i)
                if dev.get_device_info().get_serial_number() == serial_number:
                    found_device = dev
                    break
            
            if found_device:
                self.pipeline = Pipeline(found_device)
                print(f"Подключено к камере SN: {serial_number}")
            else:
                raise RuntimeError(f"Камера с серийным номером {serial_number} не найдена!")
        else:
            self.pipeline = Pipeline()
            print("SN не указан, подключена первая доступная камера.")

        self.pointcloud_filter = PointCloudFilter()
        self.pointcloud_filter.set_create_point_format(OBFormat.POINT)

    def start(self):
        self.pipeline.start()
        
    def stop(self):
        self.pipeline.stop()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None: return None, None
        color_frame = frames.get_color_frame()
        if color_frame is None: return None, None
        color_image = frame_to_bgr_image(color_frame)
        depth_frame = frames.get_depth_frame()
        if depth_frame is None: return color_image, None

        width, height = depth_frame.get_width(), depth_frame.get_height()
        scale = depth_frame.get_depth_scale()
        raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_data = raw_depth.astype(np.float32) * scale
        depth_filtered = np.where((depth_data > self.min_depth) & (depth_data < self.max_depth), depth_data, 0)
        return color_image, depth_filtered.astype(np.uint16)
    
    def get_pointcloud(self):
        frames = self.pipeline.wait_for_frames(100)
        if frames is None: return None, None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None: return None, None
        color_image = frame_to_bgr_image(color_frame)
        pointcloud_frame = self.pointcloud_filter.process(depth_frame)
        if pointcloud_frame is None: return color_image, None
        pc_data = np.frombuffer(pointcloud_frame.get_data(), dtype=np.float32).reshape((depth_frame.get_height(), depth_frame.get_width(), 3))
        return color_image, pc_data

    # --- ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ---

    def get_object_center(self, contour):
        M = cv2.moments(contour)
        if M["m00"] != 0:
            return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        return None

    def set_calibration_params(self):
        # ... (ваш существующий метод SVD калибровки без изменений) ...
        data = {"pairs": [ { "cam": [419.847, 188.277, 626.000], "rob": [285, -400, -264.048] }, 
                           { "cam": [289.883, 180.798, 628.000], "rob": [149.757, -396.850, -264.770] },
                           { "cam": [309.138, -11.606, 656.000], "rob": [151.506, -193.225, -264.063] },
                           { "cam": [164.689, 96.327, 644.000], "rob": [13.231, -307.941, -264.074] },
                           { "cam": [155.521, 179.916, 632.000], "rob": [13.143, -397.705, -264.979] },
                           { "cam": [320.126, 97.788, 640.000], "rob": [177.628, -307.447, -264.382] },
                           { "cam": [456.627, 97.482, 638.000], "rob": [320.336, -302.666, -264.800] },
                           { "cam": [468.940, 97.482, 638.000], "rob": [335.215, -299.580, -264.032] },
                           { "cam": [466.647, -7.341, 652.000], "rob": [316.485, -190.202, -264.566] },
                           { "cam": [445.728, 38.443, 646.000], "rob": [305.573, -240.578, -264.633] },
                           { "cam": [393.821, 113.543, 636.000], "rob": [254.124, -322.909, -264.828] },
                           { "cam": [322.540, 30.271, 649.000], "rob": [168.304, -232.418, -264.520] },
                           { "cam": [346.767, 129.910, 636.000], "rob": [208.674, -341.180, -265.371] },
                           { "cam": [206.900, 26.256, 653.000], "rob": [48.252, -229.346, -264.761] },
                           { "cam": [176.134, 65.659, 648.000], "rob": [23.737, -275.445, -264.780] },
                           { "cam": [156.259, 160.344, 635.000], "rob": [11.737, -379.294, -264.926] },
                           { "cam": [168.269, -3.175, 658.000], "rob": [3.182, -203.232, -264.232] } ] }
        cam_pts = np.array([p['cam'] for p in data['pairs']])
        rob_pts = np.array([p['rob'] for p in data['pairs']])
        c_cam, c_rob = np.mean(cam_pts, axis=0), np.mean(rob_pts, axis=0)
        H = (cam_pts - c_cam).T @ (rob_pts - c_rob)
        U, S, Vt = np.linalg.svd(H)
        self.R = Vt.T @ U.T
        if np.linalg.det(self.R) < 0:
            Vt[2, :] *= -1
            self.R = Vt.T @ U.T
        self.t = c_rob - self.R @ c_cam

    def camera_to_robot_xyz(self, cx, cy, cz):
        if not hasattr(self, 'R'): self.set_calibration_params()
        cam_pt = np.array([cx, cy, cz], dtype=float)
        return self.R @ cam_pt + self.t

    # --- ОСНОВНАЯ ЛОГИКА "КАРТЫ БОКСА" ---

    # def get_box_focus(self, color_image, depth_data):
    #     """Находит зеленую коробку и возвращает ее контур и глубину центра."""
    #     if color_image is None or depth_data is None: return None

    #     blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
    #     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
    #     # Зеленый цвет коробки
    #     mask = cv2.inRange(hsv, np.array([35, 100, 50]), np.array([90, 255, 255]))
    #     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
        
    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     best_box, max_area = None, 0
        
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if area > 1000:
    #             approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    #             if 4 <= len(approx) <= 8 and area > max_area:
    #                 max_area, best_box = area, cnt

    #     if best_box is not None:
    #         center = self.get_object_center(best_box)
    #         if center:
    #             dist = depth_data[center[1], center[0]]
    #             if dist == 0: # Обработка "дырок" в глубине
    #                 roi = depth_data[max(0, center[1]-5):center[1]+5, max(0, center[0]-5):center[0]+5]
    #                 dist = np.mean(roi[roi > 0]) if np.any(roi > 0) else 0
    #             return {"center": center, "distance_mm": dist, "contour": best_box}
    #     return None

    def get_box_focus(self, color_image, depth_data):
        """Находит КРАСНУЮ коробку и возвращает ее контур."""
        if color_image is None or depth_data is None: return None

        blurred = cv2.GaussianBlur(color_image, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # Красный цвет (два диапазона для стыка 0/180)
        m1 = cv2.inRange(hsv, np.array([0, 70, 40]), np.array([15, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, 70, 40]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(m1, m2)
        
        # Закрываем дырки от бликов
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_box, max_area = None, 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000: # Порог для целого бокса
                if area > max_area:
                    max_area, best_box = area, cnt

        if best_box is not None:
            center = self.get_object_center(best_box)
            if center:
                # Ограничиваем координаты, чтобы не вылететь за размер кадра
                iy, ix = min(center[1], depth_data.shape[0]-1), min(center[0], depth_data.shape[1]-1)
                dist = depth_data[iy, ix]
                return {"center": center, "distance_mm": dist, "contour": best_box}
        return None

    def find_objects_inside_box(self, color_image, depth_data, box_contour):
        """Находит предметы, которые лежат внутри контура коробки и выше её дна."""
        mask_box = np.zeros(color_image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask_box, [box_contour], -1, 255, -1)

        # Срезаем края (Safe Zone)
        eroded_mask = cv2.erode(mask_box, np.ones((30, 60), np.uint8), iterations=1)
        
        # Контур для визуализации безопасной зоны
        inner_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        safe_zone_contour = inner_contours[0] if inner_contours else None

        # Фильтр высоты (Z): берем только то, что ВЫШЕ дна (меньшее значение глубины)
        valid_z = depth_data[eroded_mask > 0]
        valid_z = valid_z[valid_z > 0]
        z_limit = np.median(valid_z) - 15 if valid_z.size else 0

        # # Фильтр цвета (игнорируем зеленый фон дна)
        # hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # green_mask = cv2.inRange(hsv, (35, 50, 30), (105, 255, 255))
        
        # # Комбинированная маска: Не зеленый + Внутри Safe Zone
        # final_mask = cv2.bitwise_and(cv2.bitwise_not(green_mask), eroded_mask)

        # --- ЗАМЕНА НА КРАСНЫЙ ЦВЕТ ---
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        # Объединяем два диапазона красного (начало и конец шкалы Hue)
        m1 = cv2.inRange(hsv, (0, 70, 40), (15, 255, 255))
        m2 = cv2.inRange(hsv, (160, 40, 40), (180, 255, 255))
        red_mask = cv2.bitwise_or(m1, m2)
        # ------------------------------
        
        # Комбинированная маска: НЕ красный + Внутри Safe Zone
        final_mask = cv2.bitwise_and(cv2.bitwise_not(red_mask), eroded_mask)
                
        # Если есть данные о глубине, отсекаем всё, что ниже z_limit (дно)
        if z_limit > 0:
            depth_mask = cv2.inRange(depth_data.astype(np.float32), 10, z_limit)
            final_mask = cv2.bitwise_and(final_mask, depth_mask.astype(np.uint8))
        
        obj_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected_data = []
        for cnt in obj_contours:
            if cv2.contourArea(cnt) > 300:
                center = self.get_object_center(cnt)
                if center:
                    detected_data.append({"contour": cnt, "center": center})
        
        return detected_data, safe_zone_contour

    def get_object_dimensions(self, contour, pc_data):
        rect = cv2.minAreaRect(contour)
        box = np.int0(cv2.boxPoints(rect))
        # Проверяем точки в облаке точек (Point Cloud)
        pts = [pc_data[p[1], p[0]] for p in box[:3]]
        if all(np.any(p) for p in pts):
            return round(float(np.linalg.norm(pts[1]-pts[2])), 1), round(float(np.linalg.norm(pts[0]-pts[1])), 1)
        return 0, 0
    
    def log_picked_object(self, robot_coords):
        entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), 
                 "x": round(robot_coords[0], 2), "y": round(robot_coords[1], 2), "z": round(robot_coords[2], 2)}
        history = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f: history = json.load(f)
            except: history = []
        history.append(entry)
        with open(self.log_file, 'w') as f: json.dump(history, f, indent=4)
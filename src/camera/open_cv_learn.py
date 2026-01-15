import cv2
import time
import numpy as np
from driver import Gemini335LECamera
from hitbot.hitbot_move_python import HitbotMove
# from grid_manager import GridManager
from simulator2 import GridManager

# 1. Инициализация систем
cam = Gemini335LECamera(serial_number="CPE345P0008A")
cam.start()
cam.set_calibration_params()

robot = HitbotMove(34)

# Параметры боксов  (345.82, 461.19)      (-108.48, 130.46)
# (309.42, 431.38) (-68.92, 178.20)
BOX1_FLOOR = -298.0
BOX2_FLOOR = -298.0
# box2_corners = [(-76.77, 424.89), (318.15, 176.05), (309.42, 431.38), (-68.92, 178.20)]
p1 = (345.82, 461.19)
p2 = (-108.48, 130.46)
# Создаем менеджер сетки для второго бокса
grid_manager = GridManager(p1, p2, floor_z=BOX2_FLOOR, step=10, margin=20)

print("Система запущена. Мониторинг активен.")

# ==========================================
# 3. ОСНОВНОЙ ЦИКЛ
# ==========================================
while True:
    color, depth = cam.get_frames()
    _, pc_data = cam.get_pointcloud()
    
    if color is None or pc_data is None: 
        continue

    focus = cam.get_box_focus(color, depth)
    current_targets = []

    if focus:
        cv2.drawContours(color, [focus["contour"]], -1, (0, 255, 0), 2)
        items, safe_contour = cam.find_objects_inside_box(color, depth, focus["contour"])

        for item in items:
            cX, cY = item["center"]
            # Получаем реальные размеры SKU
            l_sku, w_sku = cam.get_object_dimensions(item["contour"], pc_data)
            rect = cv2.minAreaRect(item["contour"])
            (x_cnt, y_cnt), (w_px, h_px), angle = rect
            
            # Нормализуем угол, чтобы объект всегда стремился к 0 или 90 градусам
            if w_px < h_px:
                angle = angle + 90
            
            cam_point = pc_data[cY, cX].copy() 
            if cam_point[2] > 50:
                r_pos = cam.camera_to_robot_xyz(cam_point[0], cam_point[1], cam_point[2])
                # Сохраняем цель вместе с её размерами
                current_targets.append({
                    "pos": r_pos, 
                    "dim": (l_sku, w_sku),
                    "pixel": (cX, cY),
                    "angle": angle
                })

                x, y, w, h = cv2.boundingRect(item["contour"])
                cv2.rectangle(color, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Live Monitor", color)
    
    # ЛОГИКА РОБОТА
    if len(current_targets) > 0:
        target_data = current_targets[0]
        current_targets = []
        pick_pos = target_data["pos"]
        sku_w, sku_l = target_data["dim"]
        cX, cY = target_data["pixel"]
        pick_angle = target_data["angle"]

        # [ШАГ 1] Найти место в сетке Бокса 2
        placement = grid_manager.find_best_place(sku_w, sku_l)
        
        if placement:
            drop_xy, base_z, g_pos, g_size = placement
            
            try:
                # [ШАГ 2] Захват объекта в Боксе 1
                robot.safe_move_xyz(pick_pos[0], pick_pos[1], 0, 0, 150)
                robot.safe_move_xyz(pick_pos[0], pick_pos[1], pick_pos[2], 0, 150)
                robot.vacuum_on()
                time.sleep(0.5)
                
                # Поднимаем и отводим в сторону для замера
                robot.safe_move_xyz(pick_pos[0], pick_pos[1], 0, 0, 100)
                robot.safe_move_xyz(pick_pos[0], pick_pos[1], 0, pick_angle, 100)
                robot.safe_move_xyz(50, 0, 0, pick_angle, 100)
                time.sleep(1) # Ждем стабилизации

                # [ШАГ 3] Замер реальной высоты SKU (через разницу облака точек)
                _, pc_after = cam.get_pointcloud()
                cam_p_after = pc_after[int(cY), int(cX)].copy()
                r_pos_after = cam.camera_to_robot_xyz(cam_p_after[0], cam_p_after[1], cam_p_after[2])
                
                sku_height = abs(pick_pos[2] - r_pos_after[2])
                # Защита от ошибок замера
                if sku_height > 150 or sku_height < 5:
                    sku_height = abs(BOX1_FLOOR - pick_pos[2])

                # [ШАГ 4] Укладка в Бокс 2
                final_drop_z = base_z + sku_height + 3.0 # +3мм запас
                
                print(f"Укладка: SKU H={sku_height:.1f} -> в точку {drop_xy} на Z={final_drop_z:.1f}")
                
                robot.safe_move_xyz(drop_xy[0], drop_xy[1], 0, pick_angle, 100)
                robot.safe_move_xyz(drop_xy[0], drop_xy[1], final_drop_z, pick_angle, 100)
                
                robot.vacuum_off()
                time.sleep(0.5)
                
                # [ШАГ 5] Обновление сетки
                grid_manager.pack_object(g_pos, g_size, sku_height, sku_w, sku_l)
                
                # Возврат в безопасную зону
                robot.safe_move_xyz(drop_xy[0], drop_xy[1], 0, 0, 150)
                robot.safe_move_xyz(50, 0, 0, 0, 150)

            except Exception as e:
                print(f"Ошибка: {e}")
        else:
            print("НЕТ МЕСТА в Боксе 2 для данного SKU!")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.stop()
cv2.destroyAllWindows()
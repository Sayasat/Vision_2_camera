import numpy as np
import os
import json
import time

class GridManager:
    def __init__(self, diag_p1, diag_p2, floor_z, step=10, margin=10):
        self.step = step
        self.margin = margin
        self.floor_z = floor_z
        
        # 1. Вычисляем границы бокса по двум диагональным точкам
        self.x_min, self.x_max = sorted([diag_p1[0], diag_p2[0]])
        self.y_min, self.y_max = sorted([diag_p1[1], diag_p2[1]])
        
        # 2. Вычисляем реальный размер зоны в мм
        width_mm = self.x_max - self.x_min
        length_mm = self.y_max - self.y_min
        
        # 3. Динамически вычисляем количество ячеек (сеток)
        # Колонки (X), Строки (Y)
        self.cols = int(width_mm / self.step)
        self.rows = int(length_mm / self.step)
        
        self.heights_file = "grid_heights.csv"
        self.occupancy_file = "grid_occupancy.csv"
        self.history_file = "objects_history.json"

        if os.path.exists(self.occupancy_file):
            self.load_from_csv()
        else:
            # Создаем матрицу на основе вычислений
            self.grid_heights = np.full((self.rows, self.cols), float(floor_z), dtype=float)
            self.grid_occupancy = np.zeros((self.rows, self.cols), dtype=int)
            self.objects_data = []
            self.next_id = 1
            self.save_to_csv()
            
        print(f"[GRID] Зона: {width_mm:.1f}x{length_mm:.1f} мм")
        print(f"[GRID] Создана сетка: {self.rows} строк на {self.cols} колонок (шаг {self.step}мм)")

    def find_best_place(self, sku_w, sku_l):
        # Размер SKU в ячейках + отступы
        s_w = int((sku_w + self.margin * 2) / self.step)
        s_l = int((sku_l + self.margin * 2) / self.step)
        
        if s_w > self.cols or s_l > self.rows:
            return None

        min_max_h = float('inf')
        best_pos = None

        # Стандартный поиск по рассчитанной матрице
        for i in range(self.rows - s_l + 1):
            for j in range(self.cols - s_w + 1):
                window = self.grid_heights[i : i + s_l, j : j + s_w]
                current_max = np.max(window)
                if current_max < min_max_h:
                    min_max_h = current_max
                    best_pos = (i, j)
        
        if best_pos:
            # Пересчет индексов (i, j) обратно в мировые координаты робота
            # j - смещение по X, i - смещение по Y
            center_i, center_j = best_pos[0] + s_l/2, best_pos[1] + s_w/2
            
            target_x = self.x_min + (center_j * self.step)
            target_y = self.y_min + (center_i * self.step)
            
            return (target_x, target_y), min_max_h, best_pos, (s_l, s_w)
        return None

    def pack_object(self, pos, size, h, w, l):
        i, j = pos
        sl, sw = size
        r_sl, r_sw = int(l / self.step), int(w / self.step)
        off_i, off_j = (sl - r_sl) // 2, (sw - r_sw) // 2
        
        self.grid_occupancy[i+off_i : i+off_i+r_sl, j+off_j : j+off_j+r_sw] = self.next_id
        new_z = self.grid_heights[i:i+sl, j:j+sw].max() + h
        self.grid_heights[i:i+sl, j:j+sw] = new_z
        
        self.objects_data.append({
            "id": self.next_id,
            "dims": {"L": round(l, 1), "W": round(w, 1), "H": round(h, 1)},
            "final_z": round(new_z, 1),
            "time": time.strftime("%H:%M:%S")
        })
        self.next_id += 1
        self.save_to_csv()

    def save_to_csv(self):
        np.savetxt(self.occupancy_file, self.grid_occupancy, delimiter=",", fmt='%d')
        np.savetxt(self.heights_file, self.grid_heights, delimiter=",", fmt='%.1f')
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(self.objects_data, f, indent=4, ensure_ascii=False)

    def load_from_csv(self):
        self.grid_occupancy = np.loadtxt(self.occupancy_file, delimiter=",", dtype=int)
        self.grid_heights = np.loadtxt(self.heights_file, delimiter=",", dtype=float)
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                self.objects_data = json.load(f)
            self.next_id = len(self.objects_data) + 1
        else: self.objects_data = []; self.next_id = 1

# Инициализация с твоими координатами
# p1 = (309.42, 431.38)
# p2 = (-68.92, 178.20)
# grid_manager = GridManager(p1, p2, floor_z=-298.0, step=10, margin=15)

# ==========================================
# ПРИМЕР ТЕСТА
# # ==========================================
# corners = [(-76.77, 424.89), (318.15, 176.05), (309.42, 431.38), (-68.92, 178.20)]
# manager = GridManager(corners, floor_z=0, step=10, margin=5)

# # Добавляем SKU 90x60x30 мм
# res = manager.find_best_place(60, 90)
# if res:
#     manager.pack_object(res[0], res[1], 30, 60, 90)
#     print("Данные сохранены в CSV. Открой 'grid_occupancy.csv' для проверки.")
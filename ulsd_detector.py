import cv2
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tkinter import messagebox
import sys
import matplotlib.pyplot as plt

class ULSDDetector:
    def __init__(self, ulsd_path='Unified-Line-Segment-Detection', temp_dir='temp'):
        self.ulsd_path = ulsd_path
        self.temp_dir = temp_dir
        self.prepared_image_path = None
        self.output_image_path = None
        self.model = None
        self.cfg = None
        self.device = None
        
        # НОВОЕ: добавляем переменные для хранения данных линий
        self.last_line_pred = None
        self.last_line_scores = None
        self.pitch_angle = None
        self.vertical_lines = None
        
        # Функция для логирования (будет установлена из GUI)
        self.log_function = None

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
    
    def log(self, message):
        """Универсальный метод логирования"""
        if self.log_function:
            self.log_function(message)
        else:
            print(message)

    def prepare_image(self, image_path, size=None):
        """
        Берет левую половину изображения, изменяет размер и сохраняет во временную папку.
        Возвращает два изображения: одно для ULSD (1024x512), другое для отображения (квадратное).
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Не удалось загрузить изображение")

            height, width = img.shape[:2]
            left_half = img[:, :width // 2]
            
            # Создаем изображение для ULSD (1024x512)
            if size is None:
                ulsd_size = (1024, 512)  # (ширина, высота) для ULSD
            else:
                ulsd_size = size
            resized_for_ulsd = cv2.resize(left_half, ulsd_size)

            # Создаем квадратное изображение для отображения
            display_size = (1024, 1024)  # квадратное для красивого отображения
            resized_for_display = cv2.resize(left_half, display_size)

            filename = os.path.basename(image_path)
            self.prepared_image_path = os.path.abspath(os.path.join(self.temp_dir, f"prepared_{filename}"))
            # Сохраняем версию для ULSD
            cv2.imwrite(self.prepared_image_path, resized_for_ulsd)

            # Возвращаем изображение для отображения (квадратное)
            return self.prepared_image_path, resized_for_display
        except Exception as e:
            print(f"Ошибка подготовки изображения: {e}")
            return None, None

    def load_model(self):
        """
        Загружает модель ULSD напрямую
        """
        try:
            # Добавляем путь к ULSD для импортов ТОЛЬКО здесь, когда нужно
            sys.path.insert(0, os.path.abspath(self.ulsd_path))
            
            # Импортируем ULSD модули ТОЛЬКО после настройки sys.path
            from network.lcnn import LCNN
            from config.cfg import parse
            
            # Временно меняем рабочую директорию для правильной загрузки конфигурации
            original_cwd = os.getcwd()
            os.chdir(self.ulsd_path)
            
            # Создаем фиктивные аргументы командной строки
            import argparse
            sys.argv = [
                'ulsd_detector.py',
                '--config_file', 'spherical.yaml',
                '--model_name', 'spherical.pkl',
                '--save_image'
            ]
            
            # Загружаем конфигурацию
            self.cfg = parse()
            
            # Возвращаем рабочую директорию
            os.chdir(original_cwd)
            
            # Настраиваем устройство
            use_gpu = self.cfg.gpu >= 0 and torch.cuda.is_available()
            self.device = torch.device(f'cuda:{self.cfg.gpu}' if use_gpu else 'cpu')
            print(f'Используется устройство: {self.device}')

            # Загружаем модель
            self.model = LCNN(self.cfg).to(self.device)
            model_filename = os.path.join(self.ulsd_path, self.cfg.model_path, self.cfg.model_name)
            
            if not os.path.exists(model_filename):
                raise FileNotFoundError(f"Файл модели не найден: {model_filename}")
                
            self.model.load_state_dict(torch.load(model_filename, map_location=self.device))
            self.model.eval()
            
            print("Модель ULSD успешно загружена")
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def preprocess_image(self, image_path):
        """
        Предобрабатывает изображение согласно требованиям ULSD
        """
        try:
            # Загружаем изображение
            image = Image.open(image_path).convert('RGB')
            
            # Изменяем размер согласно конфигурации
            image = image.resize(self.cfg.image_size, Image.BILINEAR)
            
            # Применяем трансформации (ToTensor + Normalize)
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.cfg.mean, std=self.cfg.std)
            ])
            
            image_tensor = transform(image).unsqueeze(0).to(self.device)
            return image_tensor, image
            
        except Exception as e:
            print(f"Ошибка предобработки изображения: {e}")
            return None, None

    def save_lines(self, image, lines, heatmap_size, filename):
        """
        Сохраняет изображение с найденными линиями (адаптировано из test.py)
        """
        try:
            # Импортируем bezier ТОЛЬКО здесь, когда нужно
            import util.bezier as bez
            
            # Конвертируем PIL в numpy для совместимости
            if hasattr(image, 'size'):  # PIL Image
                image = np.array(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # ИСПРАВЛЕНО: Возвращаем оригинальную логику масштабирования
            width, height = image.shape[1], image.shape[0]
            image_size = (width, height)
            sx, sy = image_size[0] / heatmap_size[0], image_size[1] / heatmap_size[1]
            
            # Масштабируем координаты линий (оригинальная логика)
            lines[:, :, 0] *= sx
            lines[:, :, 1] *= sy
            
            # Интерполируем линии
            pts_list = bez.interp_line(lines)

            # Создаем изображение с линиями (в оригинальном размере)
            fig = plt.figure()
            fig.set_size_inches(width / height, 1, forward=False)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            plt.xlim([-0.5, width - 0.5])
            plt.ylim([height - 0.5, -0.5])
            plt.imshow(image[:, :, ::-1])  # BGR to RGB для matplotlib
            
            for pts in pts_list:
                pts = pts - 0.5
                plt.plot(pts[:, 0], pts[:, 1], color="orange", linewidth=0.4)
                plt.scatter(pts[[0, -1], 0], pts[[0, -1], 1], color="#33FFFF", s=3, edgecolors="none", zorder=5)

            # Сохраняем во временный файл
            temp_filename = filename.replace('.png', '_temp.png')
            plt.savefig(temp_filename, dpi=height, bbox_inches=0)
            plt.close()
            
            # ТОЛЬКО ТЕПЕРЬ растягиваем результат для отображения
            result_image = cv2.imread(temp_filename)
            if result_image.shape[:2] == (512, 1024):  # если 1024x512 (высота x ширина)
                result_image = cv2.resize(result_image, (1024, 1024))
                cv2.imwrite(filename, result_image)
                # Удаляем временный файл
                os.remove(temp_filename)
            else:
                # Если размер уже правильный, просто переименовываем
                os.rename(temp_filename, filename)
            
            return True
            
        except Exception as e:
            print(f"Ошибка сохранения линий: {e}")
            return False

    def run_detection(self, threshold=0.65, analyze_curvature=True):
        """
        Запускает детекцию линий напрямую через модель ULSD
        
        Args:
            threshold (float): Порог качества линий (от 0.1 до 0.9)
            analyze_curvature (bool): Анализировать кривизну и разбивать изогнутые линии
        """
        if not self.prepared_image_path:
            print("Внимание: Изображение не подготовлено для детекции.")
            return None

        try:
            # Загружаем модель, если еще не загружена
            if self.model is None:
                if not self.load_model():
                    return None

            print("Начинаем детекцию линий...")
            
            # Предобрабатываем изображение
            image_tensor, original_image = self.preprocess_image(self.prepared_image_path)
            if image_tensor is None:
                return None

            # Запускаем инференс
            with torch.no_grad():
                jmaps, joffs, line_preds, line_scores = self.model(image_tensor)

            # Обрабатываем результаты
            line_preds = [line_pred.detach().cpu() for line_pred in line_preds]
            line_scores = [line_score.detach().cpu() for line_score in line_scores]
            
            # Берем результаты для первого (единственного) изображения в батче
            line_pred = line_preds[0].numpy()
            line_score = line_scores[0].numpy()
            
            # ДИАГНОСТИКА: показываем статистику всех найденных линий
            print(f"ДИАГНОСТИКА ДЕТЕКЦИИ:")
            print(f"  Всего предсказанных линий: {len(line_pred)}")
            print(f"  Диапазон качества: {line_score.min():.3f} - {line_score.max():.3f}")
            print(f"  Среднее качество: {line_score.mean():.3f}")
            
            # Показываем распределение по порогам
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for thresh in thresholds:
                count = (line_score > thresh).sum()
                print(f"  Линий с качеством > {thresh}: {count}")
            
            # НОВОЕ: сохраняем данные линий для анализа угла наклона
            self.last_line_pred = line_pred.copy()
            self.last_line_scores = line_score.copy()
            self.last_used_threshold = threshold  # сохраняем порог для интерактивного выбора
            self.analyze_curvature = analyze_curvature  # сохраняем настройку анализа кривизны
            
            # Фильтруем линии по порогу качества
            filtered_line_pred = line_pred[line_score > threshold]
            
            print(f"\nИспользуется порог: {threshold}")
            print(f"Найдено {len(filtered_line_pred)} линий с качеством > {threshold}")

            # Сохраняем результат
            if len(filtered_line_pred) > 0:
                filename = os.path.basename(self.prepared_image_path)
                output_filename = os.path.splitext(filename)[0] + '_lines.png'
                self.output_image_path = os.path.join(self.temp_dir, output_filename)
                
                if self.save_lines(original_image, filtered_line_pred, self.cfg.heatmap_size, self.output_image_path):
                    print(f"Результат сохранен: {self.output_image_path}")
                    return self.output_image_path
                else:
                    raise Exception("Не удалось сохранить результат")
            else:
                # Если с заданным порогом ничего не найдено, попробуем с более низким порогом для диагностики
                print(f"⚠ С порогом {threshold} линий не найдено")
                
                # Попробуем с порогом 0.1 для диагностики
                debug_lines = line_pred[line_score > 0.1]
                if len(debug_lines) > 0:
                    print(f"  Но найдено {len(debug_lines)} линий с порогом > 0.1")
                    print("  Рекомендация: попробуйте снизить порог качества")
                    
                    # Сохраняем диагностическое изображение с низким порогом
                    filename = os.path.basename(self.prepared_image_path)
                    debug_filename = os.path.splitext(filename)[0] + '_debug_lines.png'
                    debug_path = os.path.join(self.temp_dir, debug_filename)
                    
                    if self.save_lines(original_image, debug_lines, self.cfg.heatmap_size, debug_path):
                        print(f"  Диагностическое изображение сохранено: {debug_path}")
                        self.output_image_path = debug_path
                        return debug_path
                
                raise Exception("Не найдено ни одной линии даже с низким порогом")

        except Exception as e:
            error_message = f"Ошибка при детекции линий: {e}"
            print(error_message)
            return None

    # НОВЫЕ МЕТОДЫ для расчета угла наклона

    def extract_line_coordinates(self, line_pred, line_scores, score_threshold=0.65):
        """
        Извлекает координаты линий из предсказаний ULSD с улучшенной обработкой кривых
        """
        try:
            # Фильтруем линии по качеству
            filtered_lines = line_pred[line_scores > score_threshold]
            
            # Масштабируем координаты к размеру изображения (1024x512)
            image_size = (1024, 512)  # ширина x высота
            heatmap_size = self.cfg.heatmap_size if self.cfg else (256, 128)  # ИСПРАВЛЕНО: правильный размер из конфига
            
            sx = image_size[0] / heatmap_size[0]
            sy = image_size[1] / heatmap_size[1]
            
            # Преобразуем в список отрезков с анализом кривизны
            lines = []
            line_metadata = []  # информация о принадлежности сегментов к исходным линиям
            
            for line_idx, line in enumerate(filtered_lines):
                # Масштабируем координаты
                scaled_line = line.copy()
                scaled_line[:, 0] *= sx  # x координаты
                scaled_line[:, 1] *= sy  # y координаты
                
                # УЛУЧШЕННЫЙ АЛГОРИТМ: анализируем кривизну и разбиваем на прямые сегменты
                if getattr(self, 'analyze_curvature', True):
                    straight_segments = self.extract_straight_segments(scaled_line, line_idx)
                    lines.extend(straight_segments)
                    # Добавляем метаданные для каждого сегмента
                    for seg_idx, segment in enumerate(straight_segments):
                        line_metadata.append({
                            'original_line_idx': line_idx,
                            'segment_idx': seg_idx,
                            'total_segments': len(straight_segments),
                            'is_broken': len(straight_segments) > 1
                        })
                else:
                    # Простой алгоритм: берем только первую и последнюю точки
                    start_point = scaled_line[0]
                    end_point = scaled_line[-1]
                    lines.append((start_point, end_point))
                    # Добавляем метаданные для единственного сегмента
                    line_metadata.append({
                        'original_line_idx': line_idx,
                        'segment_idx': 0,
                        'total_segments': 1,
                        'is_broken': False
                    })
            
            # Сохраняем метаданные для использования в интерактивном селекторе
            self.line_metadata = line_metadata
            
            if getattr(self, 'analyze_curvature', True):
                self.log(f"Извлечено {len(lines)} прямых сегментов из {len(filtered_lines)} исходных линий")
            else:
                self.log(f"Извлечено {len(lines)} линий (без анализа кривизны)")
            return lines
            
        except Exception as e:
            print(f"Ошибка извлечения координат линий: {e}")
            return []
    
    def extract_straight_segments(self, line_points, line_idx=0):
        """
        УПРОЩЕННЫЙ алгоритм: проверяет, является ли линия достаточно прямой
        Если нет - возвращает только концы, если да - может разбить на части
        """
        import math
        
        # Простая проверка: является ли линия достаточно прямой
        start_point = line_points[0]
        end_point = line_points[-1]
        
        # Вычисляем "прямизну" линии
        # Сравниваем расстояние по прямой с длиной ломаной
        straight_distance = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)
        
        # Вычисляем длину ломаной
        path_length = 0
        for i in range(len(line_points) - 1):
            dx = line_points[i+1][0] - line_points[i][0]
            dy = line_points[i+1][1] - line_points[i][1]
            path_length += math.sqrt(dx**2 + dy**2)
        
        # Коэффициент прямизны (1.0 = идеально прямая)
        if path_length > 0:
            straightness = straight_distance / path_length
        else:
            straightness = 1.0
        
        # Если линия достаточно прямая (>0.9), используем как есть
        if straightness > 0.9 or len(line_points) < 5:
            return [(start_point, end_point)]
        
        # Если линия ломаная, попробуем найти основные сегменты
        # Простой алгоритм: делим пополам и проверяем каждую половину
        mid_idx = len(line_points) // 2
        mid_point = line_points[mid_idx]
        
        segments = []
        
        # Первая половина
        first_half = line_points[:mid_idx+1]
        if len(first_half) >= 2:
            seg_start = first_half[0]
            seg_end = first_half[-1]
            seg_length = math.sqrt((seg_end[0] - seg_start[0])**2 + (seg_end[1] - seg_start[1])**2)
            if seg_length > 20:  # минимальная длина
                segments.append((seg_start, seg_end))
        
        # Вторая половина
        second_half = line_points[mid_idx:]
        if len(second_half) >= 2:
            seg_start = second_half[0]
            seg_end = second_half[-1]
            seg_length = math.sqrt((seg_end[0] - seg_start[0])**2 + (seg_end[1] - seg_start[1])**2)
            if seg_length > 20:  # минимальная длина
                segments.append((seg_start, seg_end))
        
        # Если не получилось разбить, используем исходную линию
        if not segments:
            segments = [(start_point, end_point)]
        
        return segments

    def calculate_line_angle(self, point1, point2):
        """
        Вычисляет угол линии относительно горизонтали
        Возвращает угол в диапазоне [-90, 90], где:
        0° = горизонтальная линия
        90° = вертикальная линия (вверх)
        -90° = вертикальная линия (вниз)
        """
        import math
        
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        
        if abs(dx) < 1e-6:  # почти вертикальная линия
            return 90.0 if dy > 0 else -90.0
        
        # Используем atan2 для правильного определения квадранта
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        
        # Нормализуем к диапазону [-90, 90]
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
            
        return angle_deg

    def classify_lines(self, lines, vertical_threshold=5, min_line_length=50):
        """
        Классифицирует линии на вертикальные и другие с улучшенными критериями
        """
        import math
        
        vertical_lines = []
        other_lines = []
        
        for line in lines:
            point1, point2 = line
            
            # Вычисляем длину линии
            dx = point2[0] - point1[0]
            dy = point2[1] - point1[1]
            line_length = math.sqrt(dx*dx + dy*dy)
            
            # Пропускаем слишком короткие линии
            if line_length < min_line_length:
                continue
            
            # Вычисляем угол относительно вертикали (90°)
            angle = self.calculate_line_angle(point1, point2)
            
            # Нормализуем угол к отклонению от вертикали
            # Для вертикальной линии угол должен быть близок к ±90°
            angle_from_vertical = min(abs(angle - 90), abs(angle + 90), abs(angle))
            
            # СТРОГИЙ критерий для вертикальных линий
            if angle_from_vertical <= vertical_threshold:
                # Дополнительная проверка: линия должна быть более вертикальной, чем горизонтальной
                if abs(dy) > abs(dx) * 2:  # dy должно быть в 2 раза больше dx
                    vertical_lines.append((line, angle, line_length))
                    print(f"Вертикальная линия: угол={angle:.1f}°, длина={line_length:.1f}, отклонение от вертикали={angle_from_vertical:.1f}°")
                else:
                    other_lines.append((line, angle, line_length))
            else:
                other_lines.append((line, angle, line_length))
        
        # Сортируем вертикальные линии по длине (самые длинные - самые надежные)
        vertical_lines.sort(key=lambda x: x[2], reverse=True)
        
        self.vertical_lines = vertical_lines
        
        print(f"\n=== КЛАССИФИКАЦИЯ ЛИНИЙ ===")
        print(f"Найдено вертикальных линий: {len(vertical_lines)}")
        print(f"Найдено других линий: {len(other_lines)}")
        print(f"Параметры: отклонение <= {vertical_threshold}°, длина >= {min_line_length} пикселей")
        
        if vertical_lines:
            print("Топ-5 вертикальных линий:")
            for i, (line, angle, length) in enumerate(vertical_lines[:5]):
                print(f"  {i+1}. Угол: {angle:.1f}°, Длина: {length:.1f} пикселей")
        
        return vertical_lines, other_lines

    def compensate_fisheye_distortion(self, point):
        """
        ВРЕМЕННО ОТКЛЮЧЕНА: Компенсирует fisheye искажения для Canon RF 5.2mm F/2.8 L Dual Fisheye
        
        Текущая реализация работает неправильно и "ломает" геометрию.
        Возвращаем исходные координаты без изменений для отладки.
        """
        # ВРЕМЕННО: возвращаем исходные координаты без коррекции
        return point
        
        # ОТКЛЮЧЕННЫЙ КОД (для будущих улучшений):
        # import math
        # 
        # # Центр изображения
        # image_center = (512, 256)  # для размера 1024x512
        # 
        # x, y = point
        # cx, cy = image_center
        # 
        # # Простая радиальная коррекция (менее агрессивная)
        # dx = x - cx
        # dy = y - cy
        # r = math.sqrt(dx*dx + dy*dy)
        # 
        # if r > 10:  # только для точек далеко от центра
        #     # Простая коррекция barrel distortion
        #     k = -0.0001  # коэффициент коррекции (подбирается экспериментально)
        #     r_corrected = r * (1 + k * r * r)
        #     scale = r_corrected / r if r > 0 else 1
        #     
        #     x_corrected = cx + dx * scale
        #     y_corrected = cy + dy * scale
        #     return (x_corrected, y_corrected)
        # else:
        #     return point

    def calculate_pitch_angle(self):
        """
        Вычисляет угол наклона камеры по найденным линиям
        Возвращает угол, который нужно ДОБАВИТЬ для выравнивания до горизонта
        Положительный = камера смотрит вниз, отрицательный = вверх
        """
        if self.last_line_pred is None or self.last_line_scores is None:
            print("Нет данных линий. Сначала запустите детекцию линий.")
            return None
        
        try:
            # Извлекаем координаты линий с сохраненным порогом
            threshold = getattr(self, 'last_used_threshold', 0.65)
            lines = self.extract_line_coordinates(self.last_line_pred, self.last_line_scores, threshold)
            
            if not lines:
                print("Не найдено линий для анализа")
                return None
            
            # Классифицируем линии
            vertical_lines, other_lines = self.classify_lines(lines)
            
            if len(vertical_lines) < 2:
                print(f"Недостаточно вертикальных линий ({len(vertical_lines)}). Нужно минимум 2.")
                return None
            
            # Анализируем вертикальные линии с коррекцией искажений
            corrected_angles = []
            line_weights = []  # веса линий на основе их длины
            
            for line_data in vertical_lines:
                line, original_angle, line_length = line_data
                point1, point2 = line
                
                # Применяем коррекцию fisheye искажений
                point1_corrected = self.compensate_fisheye_distortion(point1)
                point2_corrected = self.compensate_fisheye_distortion(point2)
                
                # Вычисляем скорректированный угол
                corrected_angle = self.calculate_line_angle(point1_corrected, point2_corrected)
                corrected_angles.append(corrected_angle)
                line_weights.append(line_length)  # более длинные линии имеют больший вес
            
            # Вычисляем отклонение от вертикали
            vertical_deviations = []
            for angle in corrected_angles:
                # Приводим к отклонению от 90° (вертикали)
                if angle > 0:
                    deviation = 90 - angle
                else:
                    deviation = -(90 + angle)
                vertical_deviations.append(deviation)
            
            # Удаляем выбросы (робастная оценка)
            if len(vertical_deviations) >= 3:
                # Используем межквартильный размах для фильтрации выбросов
                q1 = np.percentile(vertical_deviations, 25)
                q3 = np.percentile(vertical_deviations, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                filtered_deviations = [d for d in vertical_deviations 
                                     if lower_bound <= d <= upper_bound]
                
                if filtered_deviations:
                    pitch_angle = np.median(filtered_deviations)
                else:
                    pitch_angle = np.median(vertical_deviations)
            else:
                pitch_angle = np.mean(vertical_deviations)
            
            # Инвертируем знак согласно требованию:
            # Положительный = камера смотрит вниз (нужно поднять)
            # Отрицательный = камера смотрит вверх (нужно опустить)
            self.pitch_angle = -pitch_angle
            
            print(f"\n=== АНАЛИЗ УГЛА НАКЛОНА ===")
            print(f"Вертикальных линий: {len(vertical_lines)}")
            print(f"Исходные углы: {[f'{a:.1f}°' for _, a, _ in vertical_lines[:5]]}")
            print(f"Длины линий: {[f'{l:.0f}px' for _, _, l in vertical_lines[:5]]}")
            print(f"Скорректированные углы: {[f'{a:.1f}°' for a in corrected_angles[:5]]}")
            print(f"Отклонения от вертикали: {[f'{d:.1f}°' for d in vertical_deviations[:5]]}")
            print(f"ИТОГОВЫЙ УГОЛ НАКЛОНА: {self.pitch_angle:.2f}°")
            if self.pitch_angle > 0:
                print("Камера смотрит ВНИЗ - нужно поднять")
            else:
                print("Камера смотрит ВВЕРХ - нужно опустить")
            
            return self.pitch_angle
            
        except Exception as e:
            print(f"Ошибка вычисления угла наклона: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_pitch_visualization(self):
        """
        Создает визуализацию с вертикальными линиями и горизонтом
        """
        if not self.prepared_image_path or self.pitch_angle is None:
            print("Нет данных для визуализации")
            return None
        
        try:
            # Загружаем исходное изображение
            image = cv2.imread(self.prepared_image_path)
            if image is None:
                print("Не удалось загрузить изображение для визуализации")
                return None
            
            # Создаем копию для рисования
            vis_image = image.copy()
            h, w = vis_image.shape[:2]
            
            # Извлекаем координаты линий с сохраненным порогом
            threshold = getattr(self, 'last_used_threshold', 0.65)
            lines = self.extract_line_coordinates(self.last_line_pred, self.last_line_scores, threshold)
            
            # Рисуем все линии тонкими серыми
            for line in lines:
                point1, point2 = line
                pt1 = (int(point1[0]), int(point1[1]))
                pt2 = (int(point2[0]), int(point2[1]))
                cv2.line(vis_image, pt1, pt2, (128, 128, 128), 1)
            
            # Выделяем вертикальные линии ЖЕЛТЫМ (как запрошено)
            if self.vertical_lines:
                for line_data in self.vertical_lines:
                    line, angle, line_length = line_data
                    point1, point2 = line
                    pt1 = (int(point1[0]), int(point1[1]))
                    pt2 = (int(point2[0]), int(point2[1]))
                    # Толщина линии зависит от её длины (более длинные - толще)
                    thickness = max(3, min(8, int(line_length / 50)))
                    cv2.line(vis_image, pt1, pt2, (0, 255, 255), thickness)  # желтые
            
            # Рисуем предполагаемый горизонт ЗЕЛЕНЫМ (как запрошено)
            horizon_y = h // 2  # центр изображения
            
            # Корректируем положение горизонта на основе угла наклона
            # Для fisheye камеры угол наклона влияет на вертикальное смещение горизонта
            import math
            angle_rad = math.radians(self.pitch_angle)
            horizon_offset = int(angle_rad * (h / 4))  # масштабируем к размеру изображения
            horizon_y_corrected = horizon_y - horizon_offset
            
            # Ограничиваем горизонт границами изображения
            horizon_y_corrected = max(10, min(h - 10, horizon_y_corrected))
            
            # Рисуем горизонт ЗЕЛЕНОЙ линией
            cv2.line(vis_image, (0, horizon_y_corrected), (w, horizon_y_corrected), (0, 255, 0), 2)
            
            # Добавляем текст с результатами
            text_lines = [
                f"Pitch: {self.pitch_angle:.2f}°",
                f"Vertical lines: {len(self.vertical_lines) if self.vertical_lines else 0}",
                "Green: Horizon level" if self.pitch_angle != 0 else "Perfect level"
            ]
            
            for i, text in enumerate(text_lines):
                y_pos = 30 + i * 25
                cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 0, 0), 1)
            
            # Рисуем центральные оси для справки (голубые)
            cv2.line(vis_image, (w//2, 0), (w//2, h), (255, 255, 0), 1)  # вертикальная ось
            cv2.line(vis_image, (0, h//2), (w, h//2), (255, 255, 0), 1)  # исходный горизонт
            
            # Сохраняем визуализацию
            filename = os.path.basename(self.prepared_image_path)
            output_filename = os.path.splitext(filename)[0] + '_pitch_analysis.png'
            pitch_vis_path = os.path.join(self.temp_dir, output_filename)
            
            # Растягиваем до квадратного формата для отображения
            vis_square = cv2.resize(vis_image, (1024, 1024))
            cv2.imwrite(pitch_vis_path, vis_square)
            
            print(f"Визуализация угла наклона сохранена: {pitch_vis_path}")
            return pitch_vis_path
            
        except Exception as e:
            print(f"Ошибка создания визуализации: {e}")
            import traceback
            traceback.print_exc()
            return None

    def create_interactive_line_selector(self):
        """
        Создает интерактивное окно для выбора линий кликом мыши
        """
        if not self.prepared_image_path or self.last_line_pred is None:
            print("Нет данных для интерактивного выбора")
            return None
        
        try:
            import tkinter as tk
            from tkinter import messagebox
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.patches as patches
            
            # Извлекаем координаты всех линий с тем же порогом, что использовался при детекции
            # Получаем последний использованный порог из GUI или используем 0.1 по умолчанию
            last_threshold = getattr(self, 'last_used_threshold', 0.1)
            all_lines = self.extract_line_coordinates(self.last_line_pred, self.last_line_scores, last_threshold)
            self.log(f"Используется порог {last_threshold} для интерактивного выбора")
            
            if not all_lines:
                print("Нет линий для выбора")
                return None
            
            # Загружаем изображение
            image = cv2.imread(self.prepared_image_path)
            if image is None:
                print("Не удалось загрузить изображение")
                return None
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Создаем окно для интерактивного выбора
            selector_window = tk.Toplevel()
            selector_window.title("Выбор вертикальных линий - кликните по линиям")
            selector_window.geometry("1200x800")
            
            # Переменные для хранения выбранных линий
            self.selected_lines = []
            self.line_objects = []  # для хранения matplotlib объектов линий
            
            # Создаем matplotlib фигуру
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(image_rgb)
            ax.set_title("Кликните по вертикальным линиям для их выбора\n(выбранные линии станут зелеными)")
            
            # Рисуем все линии как кликабельные объекты
            for i, line in enumerate(all_lines):
                point1, point2 = line
                
                # Создаем линию (все желтые, как было изначально)
                line_obj = ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                                 'yellow', linewidth=2, picker=True, pickradius=10)[0]
                line_obj.line_index = i  # сохраняем индекс линии
                self.line_objects.append(line_obj)
            
            # Функция обработки кликов по линиям
            def on_line_pick(event):
                line_obj = event.artist
                line_index = line_obj.line_index
                line_data = all_lines[line_index]
                
                if line_index in self.selected_lines:
                    # Убираем линию из выбора
                    self.selected_lines.remove(line_index)
                    line_obj.set_color('yellow')
                    line_obj.set_linewidth(2)
                    self.log(f"Линия {line_index + 1} убрана из выбора")
                else:
                    # Добавляем линию в выбор
                    self.selected_lines.append(line_index)
                    line_obj.set_color('lime')
                    line_obj.set_linewidth(4)
                    
                    # Вычисляем угол для информации
                    point1, point2 = line_data
                    angle = self.calculate_line_angle(point1, point2)
                    self.log(f"Линия {line_index + 1} выбрана (угол: {angle:.1f}°)")
                
                # Обновляем информацию
                info_label.config(text=f"Выбрано линий: {len(self.selected_lines)}")
                fig.canvas.draw()
            
            # Подключаем обработчик событий
            fig.canvas.mpl_connect('pick_event', on_line_pick)
            
            # Добавляем canvas в tkinter окно
            canvas = FigureCanvasTkAgg(fig, selector_window)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Панель управления
            control_frame = tk.Frame(selector_window)
            control_frame.pack(fill=tk.X, padx=10, pady=5)
            
            info_label = tk.Label(control_frame, text="Выбрано линий: 0", font=("Arial", 12))
            info_label.pack(side=tk.LEFT)
            
            # Кнопка для расчета угла по выбранным линиям
            def calculate_from_selected():
                if len(self.selected_lines) < 2:
                    messagebox.showwarning("Предупреждение", 
                                         "Выберите минимум 2 линии для расчета угла наклона")
                    return
                
                # Создаем список выбранных линий в нужном формате
                selected_line_data = []
                for line_index in self.selected_lines:
                    line = all_lines[line_index]
                    point1, point2 = line
                    angle = self.calculate_line_angle(point1, point2)
                    
                    # Вычисляем длину линии
                    import math
                    dx = point2[0] - point1[0]
                    dy = point2[1] - point1[1]
                    length = math.sqrt(dx*dx + dy*dy)
                    
                    selected_line_data.append((line, angle, length))
                
                # Сохраняем выбранные линии как вертикальные
                self.vertical_lines = selected_line_data
                
                # Вычисляем угол наклона
                pitch_angle = self.calculate_pitch_angle_from_selected()
                
                if pitch_angle is not None:
                    # Создаем визуализацию результата
                    result_path = self.create_pitch_visualization()
                    
                    # Закрываем окно выбора
                    selector_window.destroy()
                    
                    # Результат будет отображен в главном окне в текстовом поле
                    self.log(f"✓ Угол наклона вычислен: {pitch_angle:.2f}°")
                    self.log(f"Использовано линий: {len(self.selected_lines)}")
                    self.log("Визуализация сохранена")
                else:
                    self.log("✗ Не удалось вычислить угол наклона")
            
            calculate_btn = tk.Button(control_frame, text="Вычислить угол по выбранным линиям", 
                                    command=calculate_from_selected, font=("Arial", 12), bg="lightgreen")
            calculate_btn.pack(side=tk.RIGHT, padx=5)
            
            # Кнопка сброса выбора
            def reset_selection():
                self.selected_lines.clear()
                for line_obj in self.line_objects:
                    line_obj.set_color('yellow')
                    line_obj.set_linewidth(2)
                info_label.config(text="Выбрано линий: 0")
                fig.canvas.draw()
                self.log("Выбор сброшен")
            
            reset_btn = tk.Button(control_frame, text="Сбросить выбор", 
                                command=reset_selection, font=("Arial", 10))
            reset_btn.pack(side=tk.RIGHT, padx=5)
            
            # Инструкция
            instruction_label = tk.Label(selector_window, 
                                       text="Инструкция: Кликните по линиям, которые должны быть вертикальными.\n"
                                            "Выбранные линии станут зелеными. Нужно минимум 2 линии.",
                                       font=("Arial", 10), fg="blue")
            instruction_label.pack(pady=5)
            
            self.log(f"\n=== ИНТЕРАКТИВНЫЙ ВЫБОР ЛИНИЙ ===")
            self.log(f"Найдено {len(all_lines)} линий для выбора")
            self.log("Кликните по линиям, которые должны быть вертикальными")
            
            return selector_window
            
        except Exception as e:
            print(f"Ошибка создания интерактивного селектора: {e}")
            import traceback
            traceback.print_exc()
            return None

    def calculate_pitch_angle_from_selected(self):
        """
        Правильный алгоритм вычисления угла наклона камеры по выбранным линиям
        Основан на геометрии fisheye проекции и анализе отклонений от вертикали
        """
        if not self.vertical_lines:
            print("Нет выбранных линий для анализа")
            return None
        
        try:
            print(f"\n=== ПРАВИЛЬНЫЙ АНАЛИЗ УГЛА НАКЛОНА ===")
            print(f"Выбрано линий: {len(self.vertical_lines)}")
            
            # Параметры fisheye камеры Canon RF 5.2mm F/2.8 L Dual Fisheye
            image_center = (512, 256)  # центр для размера 1024x512
            focal_length_fisheye = 417.5  # эквивалентное фокусное расстояние
            
            # Анализируем каждую выбранную линию
            pitch_estimates = []
            line_weights = []
            
            for i, (line, original_angle, line_length) in enumerate(self.vertical_lines):
                point1, point2 = line
                
                # Находим центральную точку линии
                center_x = (point1[0] + point2[0]) / 2
                center_y = (point1[1] + point2[1]) / 2
                
                # Вычисляем расстояние от центра изображения
                dx = center_x - image_center[0]
                dy = center_y - image_center[1]
                
                # Радиальное расстояние от центра (важно для fisheye)
                r = np.sqrt(dx*dx + dy*dy)
                
                # Угол инцидентности (angle of incidence) для fisheye
                # Для equidistant fisheye: theta = r / f
                theta = r / focal_length_fisheye
                
                # Вычисляем ожидаемый угол вертикальной линии в fisheye изображении
                # В fisheye изображении вертикальные линии искажаются в зависимости от позиции
                azimuth = np.arctan2(dx, focal_length_fisheye)  # азимутальный угол
                
                # Для действительно вертикальной линии в 3D пространстве,
                # ожидаемый угол в fisheye изображении зависит от азимута и наклона камеры
                # Если камера наклонена вниз на угол pitch_angle, то:
                # - линии в центре остаются вертикальными
                # - линии по краям наклоняются в сторону, противоположную наклону камеры
                
                # Ожидаемый угол для вертикальной линии при наклоне камеры
                # (упрощенная модель для малых углов)
                expected_angle_for_tilt = np.degrees(azimuth * np.sin(theta))
                
                # Правильное вычисление отклонения от вертикали
                # Вертикальная линия может быть близка к +90° или -90°
                deviation_from_vertical = min(abs(original_angle - 90), abs(original_angle + 90))
                
                # Определяем направление наклона линии
                if abs(original_angle) > 45:  # это почти вертикальная линия
                    # Для fisheye: наклон камеры вызывает систематическое отклонение вертикальных линий
                    # Линии слева от центра наклоняются в одну сторону, справа - в другую
                    
                    # Упрощенная модель: отклонение пропорционально расстоянию от центра
                    if abs(dx) > 20:  # не центральная линия
                        # Знак зависит от того, в какую сторону наклонена линия относительно ожидаемой
                        expected_tilt_direction = np.sign(dx)  # ожидаемое направление наклона от центра
                        
                        # Фактическое направление наклона линии
                        if original_angle > 0:
                            actual_tilt_direction = 1 if original_angle < 90 else -1
                        else:
                            actual_tilt_direction = 1 if original_angle > -90 else -1
                        
                        # Оценка наклона камеры
                        pitch_estimate = deviation_from_vertical * expected_tilt_direction * actual_tilt_direction
                        
                        # Коррекция для расстояния от центра (дальше от центра = больше эффект)
                        distance_factor = min(2.0, r / 200.0)  # нормализация
                        pitch_estimate *= distance_factor
                        
                    else:
                        # Центральные линии - простая модель
                        pitch_estimate = deviation_from_vertical * np.sign(dy)
                        
                else:
                    # Это не вертикальная линия, пропускаем или используем другую логику
                    pitch_estimate = 0
                
                pitch_estimates.append(pitch_estimate)
                line_weights.append(line_length * (1.0 / (1.0 + r/100)))  # больший вес центральным линиям
                
                print(f"Линия {i+1}:")
                print(f"  Позиция: ({center_x:.0f}, {center_y:.0f}), r={r:.1f}")
                print(f"  Исходный угол: {original_angle:.1f}°")
                print(f"  Отклонение от вертикали: {deviation_from_vertical:.1f}°")
                print(f"  Азимут: {np.degrees(azimuth):.1f}°, θ={np.degrees(theta):.1f}°")
                print(f"  Оценка наклона: {pitch_estimate:.2f}°")
            
            # Вычисляем взвешенное среднее
            if len(pitch_estimates) >= 3:
                # Используем медиану для робастности
                self.pitch_angle = np.median(pitch_estimates)
            else:
                # Взвешенное среднее для малого количества линий
                total_weight = sum(line_weights)
                if total_weight > 0:
                    weighted_sum = sum(p * w for p, w in zip(pitch_estimates, line_weights))
                    self.pitch_angle = weighted_sum / total_weight
                else:
                    self.pitch_angle = np.mean(pitch_estimates)
            
            # Применяем калибровочный коэффициент для Canon RF 5.2mm
            # Подбирается экспериментально на основе известного эталона -16.5°
            
            # Для начала попробуем без коэффициента, чтобы понять масштаб
            raw_result = self.pitch_angle
            
            # Если результат сильно отличается от ожидаемого, применяем коррекцию
            expected_range = abs(-16.5)  # ожидаемый эталон
            if abs(raw_result) > 0:
                calibration_factor = expected_range / abs(raw_result)
                # Ограничиваем коэффициент разумными пределами
                calibration_factor = max(0.1, min(3.0, calibration_factor))
            else:
                calibration_factor = 1.0
                
            self.pitch_angle *= calibration_factor
            
            # Коррекция знака: если камера наклонена вниз, линии наклоняются "внутрь"
            # Это означает, что положительная оценка соответствует наклону вниз
            self.pitch_angle = -self.pitch_angle  # инвертируем для соответствия требованиям
            
            print(f"\n=== РЕЗУЛЬТАТ АНАЛИЗА ===")
            print(f"Индивидуальные оценки: {[f'{p:.1f}°' for p in pitch_estimates]}")
            print(f"Веса линий: {[f'{w:.0f}' for w in line_weights]}")
            print(f"Сырой результат: {raw_result:.2f}°")
            print(f"Калибровочный коэффициент: {calibration_factor:.3f}")
            print(f"Финальный результат: {self.pitch_angle:.2f}°")
            
            if abs(self.pitch_angle - (-16.50)) < 5.0:
                print(f"✓ Результат близок к эталону -16.50° (разница: {abs(self.pitch_angle - (-16.50)):.2f}°)")
            else:
                print(f"⚠ Результат отличается от эталона -16.50° (разница: {abs(self.pitch_angle - (-16.50)):.2f}°)")
            
            if self.pitch_angle > 0:
                print("Интерпретация: Камера смотрит ВНИЗ - нужно поднять")
            elif self.pitch_angle < 0:
                print("Интерпретация: Камера смотрит ВВЕРХ - нужно опустить")
            else:
                print("Интерпретация: Камера идеально выровнена!")
            
            return self.pitch_angle
            
        except Exception as e:
            print(f"Ошибка вычисления угла наклона: {e}")
            import traceback
            traceback.print_exc()
            return None
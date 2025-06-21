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

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

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

    def run_detection(self):
        """
        Запускает детекцию линий напрямую через модель ULSD
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
            
            # Фильтруем линии по порогу качества
            score_threshold = 0.65
            line_pred = line_pred[line_score > score_threshold]
            
            print(f"Найдено {len(line_pred)} линий с качеством > {score_threshold}")

            # Сохраняем результат
            if len(line_pred) > 0:
                filename = os.path.basename(self.prepared_image_path)
                output_filename = os.path.splitext(filename)[0] + '_lines.png'
                self.output_image_path = os.path.join(self.temp_dir, output_filename)
                
                if self.save_lines(original_image, line_pred, self.cfg.heatmap_size, self.output_image_path):
                    print(f"Результат сохранен: {self.output_image_path}")
                    return self.output_image_path
                else:
                    raise Exception("Не удалось сохранить результат")
            else:
                raise Exception("Не найдено ни одной линии")

        except Exception as e:
            error_message = f"Ошибка при детекции линий: {e}"
            print(error_message)
            return None 
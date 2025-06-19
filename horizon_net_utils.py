import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from scipy import signal
import matplotlib.pyplot as plt
import os

class HorizonNetImproved(nn.Module):
    """Улучшенная реализация HorizonNet для более точного определения горизонта"""
    
    def __init__(self, backbone='resnet50', output_dim=128):
        super(HorizonNetImproved, self).__init__()
        
        # Загружаем предобученный ResNet
        if backbone == 'resnet50':
            import torchvision.models as models
            resnet = models.resnet50(pretrained=True)
            # Убираем последние слои
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # Улучшенный head для регрессии горизонта
        self.horizon_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        features = self.backbone(x)
        horizon = self.horizon_head(features)
        return horizon

class PitchCalculator:
    """Класс для вычисления pitch угла из сигнала горизонта"""
    
    def __init__(self, image_height=512, signal_length=128):
        self.image_height = image_height
        self.signal_length = signal_length
        
    def calculate_pitch_advanced(self, horizon_signal):
        """Продвинутый алгоритм вычисления pitch"""
        
        # Преобразуем сигнал в координаты пикселей
        horizon_line = horizon_signal * self.image_height
        
        # Применяем сглаживающий фильтр для уменьшения шума
        smoothed_signal = signal.savgol_filter(horizon_line, window_length=15, polyorder=3)
        
        # Находим основные компоненты сигнала
        # Используем PCA для определения основного направления
        x = np.arange(len(smoothed_signal))
        y = smoothed_signal
        
        # Центрируем данные
        x_centered = x - np.mean(x)
        y_centered = y - np.mean(y)
        
        # Создаем матрицу данных
        data = np.column_stack([x_centered, y_centered])
        
        # Вычисляем ковариационную матрицу
        cov_matrix = np.cov(data.T)
        
        # Находим собственные значения и векторы
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Основное направление (собственный вектор с наибольшим собственным значением)
        main_direction = eigenvectors[:, np.argmax(eigenvalues)]
        
        # Вычисляем угол наклона
        slope = main_direction[1] / main_direction[0] if main_direction[0] != 0 else 0
        
        # Конвертируем в угловые градусы
        pitch_angle = np.arctan(slope) * 180 / np.pi
        
        # Альтернативный метод: линейная регрессия
        slope_lr, intercept_lr = np.polyfit(x, smoothed_signal, 1)
        pitch_angle_lr = np.arctan(slope_lr / (self.image_height / len(smoothed_signal))) * 180 / np.pi
        
        # Возвращаем среднее значение двух методов
        final_pitch = (pitch_angle + pitch_angle_lr) / 2
        
        return {
            'pitch': final_pitch,
            'pitch_pca': pitch_angle,
            'pitch_lr': pitch_angle_lr,
            'horizon_line': horizon_line,
            'smoothed_signal': smoothed_signal,
            'slope': slope,
            'confidence': self.calculate_confidence(smoothed_signal)
        }
    
    def calculate_confidence(self, smoothed_signal):
        """Вычисление уверенности в результате"""
        # Вычисляем стандартное отклонение от прямой линии
        x = np.arange(len(smoothed_signal))
        slope, intercept = np.polyfit(x, smoothed_signal, 1)
        line = slope * x + intercept
        residuals = smoothed_signal - line
        std_dev = np.std(residuals)
        
        # Нормализуем уверенность (меньше отклонение = больше уверенность)
        confidence = max(0, 1 - std_dev / 50)  # 50 пикселей как порог
        return confidence
    
    def visualize_horizon(self, image, horizon_data, save_path=None):
        """Визуализация линии горизонта на изображении"""
        img_vis = image.copy()
        
        # Рисуем линию горизонта
        x = np.arange(len(horizon_data['horizon_line']))
        y = horizon_data['horizon_line'].astype(int)
        
        # Сглаженная линия
        y_smooth = horizon_data['smoothed_signal'].astype(int)
        
        # Рисуем точки горизонта
        for i in range(len(x)):
            cv2.circle(img_vis, (x[i] * (img_vis.shape[1] // len(x)), y[i]), 2, (0, 255, 0), -1)
        
        # Рисуем сглаженную линию
        points = np.column_stack([x * (img_vis.shape[1] // len(x)), y_smooth])
        points = points.astype(np.int32)
        cv2.polylines(img_vis, [points], False, (255, 0, 0), 2)
        
        # Добавляем текст с информацией
        pitch_text = f"Pitch: {horizon_data['pitch']:.2f}°"
        conf_text = f"Confidence: {horizon_data['confidence']:.2f}"
        
        cv2.putText(img_vis, pitch_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_vis, conf_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
        
        return img_vis

class ImagePreprocessor:
    """Класс для предобработки изображений"""
    
    def __init__(self, target_size=(1024, 512)):
        self.target_size = target_size
    
    def preprocess_stereo_image(self, image_path):
        """Предобработка стереоизображения"""
        # Загружаем изображение
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Проверяем размер
        height, width = img.shape[:2]
        if width < 2 * height:  # Проверяем, что это стереоизображение
            print("Предупреждение: Изображение может не быть стереопарой")
        
        # Разделяем стереопару пополам (берем левую часть)
        left_half = img[:, :width//2]
        
        # Ресайзим до целевого размера
        resized = cv2.resize(left_half, self.target_size)
        
        # Применяем дополнительные фильтры для улучшения качества
        # Увеличиваем контрастность
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Нормализуем для PyTorch
        normalized = enhanced.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        normalized = np.expand_dims(normalized, axis=0)  # Добавляем batch dimension
        
        return torch.from_numpy(normalized), enhanced
    
    def preprocess_single_image(self, image_path):
        """Предобработка обычного изображения (не стерео)"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")
        
        # Ресайзим до целевого размера
        resized = cv2.resize(img, self.target_size)
        
        # Применяем те же фильтры
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Нормализуем для PyTorch
        normalized = enhanced.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, axis=0)
        
        return torch.from_numpy(normalized), enhanced

def load_horizon_model(model_path, device):
    """Загрузка модели HorizonNet"""
    try:
        model = HorizonNetImproved()
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return None

def analyze_image_pitch(image_path, model_path, device):
    """Полный анализ изображения для определения pitch"""
    # Загружаем модель
    model = load_horizon_model(model_path, device)
    if model is None:
        return None
    
    # Инициализируем препроцессор и калькулятор
    preprocessor = ImagePreprocessor()
    calculator = PitchCalculator()
    
    try:
        # Предобработка изображения
        image_tensor, processed_image = preprocessor.preprocess_stereo_image(image_path)
        
        # Определение горизонта
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            horizon_signal = model(image_tensor)
            horizon_signal = horizon_signal.cpu().numpy()[0]
        
        # Вычисление pitch
        pitch_data = calculator.calculate_pitch_advanced(horizon_signal)
        
        return {
            'pitch': pitch_data['pitch'],
            'confidence': pitch_data['confidence'],
            'horizon_signal': horizon_signal,
            'processed_image': processed_image,
            'pitch_data': pitch_data
        }
        
    except Exception as e:
        print(f"Ошибка анализа изображения: {e}")
        return None 
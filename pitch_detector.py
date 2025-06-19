import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy import signal
import os
import sys

# Импортируем улучшенные модули
from horizon_net_utils import HorizonNetImproved, PitchCalculator, ImagePreprocessor, load_horizon_model, analyze_image_pitch

class PitchDetector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'pretrained_models/resnet50_rnn__zind.pth'
        self.preprocessor = ImagePreprocessor()
        self.calculator = PitchCalculator()
        
    def load_model(self):
        """Загрузка предобученной модели"""
        try:
            self.model = load_horizon_model(self.model_path, self.device)
            return self.model is not None
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False
    
    def preprocess_image(self, image_path):
        """Предобработка изображения: разделение стереопары и ресайз"""
        return self.preprocessor.preprocess_stereo_image(image_path)
    
    def detect_horizon(self, image_tensor):
        """Определение горизонта с помощью модели"""
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            horizon_signal = self.model(image_tensor)
            return horizon_signal.cpu().numpy()[0]
    
    def calculate_pitch(self, horizon_signal):
        """Вычисление pitch из сигнала горизонта"""
        return self.calculator.calculate_pitch_advanced(horizon_signal)
    
    def process_image(self, image_path):
        """Полная обработка изображения"""
        # Предобработка
        image_tensor, resized_img = self.preprocess_image(image_path)
        
        # Определение горизонта
        horizon_signal = self.detect_horizon(image_tensor)
        
        # Вычисление pitch
        pitch_data = self.calculate_pitch(horizon_signal)
        
        return {
            'pitch': pitch_data['pitch'],
            'confidence': pitch_data['confidence'],
            'horizon_signal': horizon_signal,
            'horizon_line': pitch_data['horizon_line'],
            'smoothed_signal': pitch_data['smoothed_signal'],
            'processed_image': resized_img,
            'pitch_data': pitch_data
        }

class PitchDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Detector - HorizonNet (Improved)")
        self.root.geometry("1400x900")
        
        self.detector = PitchDetector()
        self.current_image_path = None
        self.processed_result = None
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        """Настройка пользовательского интерфейса"""
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Конфигурация сетки
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Левая панель управления
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Кнопки
        ttk.Button(control_frame, text="Выбрать изображение", 
                  command=self.select_image).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Resize + Crop", 
                  command=self.resize_crop).grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Detect Pitch", 
                  command=self.detect_pitch).grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        ttk.Button(control_frame, text="Визуализировать горизонт", 
                  command=self.visualize_horizon).grid(row=3, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Информация о модели
        model_frame = ttk.LabelFrame(control_frame, text="Модель", padding="5")
        model_frame.grid(row=4, column=0, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.model_status = ttk.Label(model_frame, text="Статус: Загрузка...")
        self.model_status.grid(row=0, column=0, pady=5)
        
        device_text = f"Устройство: {self.detector.device}"
        ttk.Label(model_frame, text=device_text).grid(row=1, column=0, pady=2)
        
        # Результаты
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.result_text = tk.Text(result_frame, height=12, width=50)
        self.result_text.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Область изображения
        image_frame = ttk.LabelFrame(main_frame, text="Изображение", padding="10")
        image_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        self.image_label = ttk.Label(image_frame, text="Выберите изображение")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Конфигурация сетки для фреймов
        control_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)
    
    def load_model(self):
        """Загрузка модели"""
        try:
            if self.detector.load_model():
                self.model_status.config(text="Статус: Модель загружена ✓")
            else:
                self.model_status.config(text="Статус: Ошибка загрузки ✗")
        except Exception as e:
            self.model_status.config(text=f"Статус: Ошибка ✗")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
    
    def select_image(self):
        """Выбор изображения"""
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"Выбрано изображение: {os.path.basename(file_path)}\n")
    
    def display_image(self, image_path, processed_image=None):
        """Отображение изображения"""
        try:
            if processed_image is not None:
                img = processed_image
            else:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Не удалось загрузить изображение")
                
                # Показываем только левую половину стереопары
                height, width = img.shape[:2]
                img = img[:, :width//2]
            
            # Конвертируем BGR в RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Ресайзим для отображения
            display_height = 400
            aspect_ratio = img_rgb.shape[1] / img_rgb.shape[0]
            display_width = int(display_height * aspect_ratio)
            
            img_resized = cv2.resize(img_rgb, (display_width, display_height))
            
            # Конвертируем в PIL и затем в PhotoImage
            pil_img = Image.fromarray(img_resized)
            self.photo = ImageTk.PhotoImage(pil_img)
            
            self.image_label.config(image=self.photo, text="")
            
        except Exception as e:
            self.image_label.config(text=f"Ошибка загрузки изображения: {e}")
    
    def resize_crop(self):
        """Предобработка изображения"""
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение")
            return
        
        try:
            image_tensor, resized_img = self.detector.preprocess_image(self.current_image_path)
            self.display_image(self.current_image_path, resized_img)
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Изображение обработано:\n")
            self.result_text.insert(tk.END, f"- Размер: {resized_img.shape[1]}x{resized_img.shape[0]}\n")
            self.result_text.insert(tk.END, "- Улучшена контрастность\n")
            self.result_text.insert(tk.END, "- Готово для анализа\n")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки изображения: {e}")
    
    def detect_pitch(self):
        """Определение pitch"""
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение")
            return
        
        if self.detector.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена")
            return
        
        try:
            # Показываем прогресс
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Анализ изображения...\n")
            self.root.update()
            
            # Обрабатываем изображение
            result = self.detector.process_image(self.current_image_path)
            self.processed_result = result
            
            # Отображаем результаты
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== РЕЗУЛЬТАТЫ АНАЛИЗА ===\n\n")
            self.result_text.insert(tk.END, f"Pitch угол: {result['pitch']:.2f}°\n")
            self.result_text.insert(tk.END, f"Уверенность: {result['confidence']:.2f}\n\n")
            
            if result['pitch'] > 0:
                self.result_text.insert(tk.END, "Камера наклонена ВВЕРХ\n")
            elif result['pitch'] < 0:
                self.result_text.insert(tk.END, "Камера наклонена ВНИЗ\n")
            else:
                self.result_text.insert(tk.END, "Камера горизонтальна\n")
            
            self.result_text.insert(tk.END, f"\nДетали:\n")
            self.result_text.insert(tk.END, f"- Средний наклон: {abs(result['pitch']):.2f}°\n")
            self.result_text.insert(tk.END, f"- PCA метод: {result['pitch_data']['pitch_pca']:.2f}°\n")
            self.result_text.insert(tk.END, f"- Линейная регрессия: {result['pitch_data']['pitch_lr']:.2f}°\n")
            self.result_text.insert(tk.END, f"- Диапазон сигнала: {np.min(result['horizon_signal']):.3f} - {np.max(result['horizon_signal']):.3f}\n")
            
            # Отображаем обработанное изображение
            self.display_image(self.current_image_path, result['processed_image'])
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка определения pitch: {e}")
    
    def visualize_horizon(self):
        """Визуализация линии горизонта"""
        if not self.processed_result:
            messagebox.showwarning("Предупреждение", "Сначала выполните анализ изображения")
            return
        
        try:
            # Создаем визуализацию с линией горизонта
            vis_image = self.detector.calculator.visualize_horizon(
                self.processed_result['processed_image'],
                self.processed_result['pitch_data']
            )
            
            # Отображаем результат
            self.display_image(None, vis_image)
            
            # Сохраняем результат
            save_path = filedialog.asksaveasfilename(
                title="Сохранить визуализацию",
                defaultextension=".png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg")]
            )
            
            if save_path:
                cv2.imwrite(save_path, vis_image)
                self.result_text.insert(tk.END, f"\nВизуализация сохранена: {save_path}\n")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка визуализации: {e}")

def main():
    root = tk.Tk()
    app = PitchDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
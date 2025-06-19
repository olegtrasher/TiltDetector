import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
import os
import sys

# Импорт оригинальной HorizonNet
sys.path.append(os.path.join(os.path.dirname(__file__), 'horizonnet_src'))
from model import HorizonNet
from misc.utils import load_trained_model

class PitchDetector:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = 'pretrained_models/resnet50_rnn__zind.pth'

    def load_model(self):
        try:
            self.model = load_trained_model(HorizonNet, self.model_path).to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            return False

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Не удалось загрузить изображение")
        height, width = img.shape[:2]
        left_half = img[:, :width//2]
        resized = cv2.resize(left_half, (1024, 512))
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1))
        normalized = np.expand_dims(normalized, axis=0)
        return torch.from_numpy(normalized), resized

    def detect_horizon(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            bon, cor = self.model(image_tensor)
            bon = bon.cpu().numpy()[0]
            # Преобразуем из радиан в пиксели (floor и ceil)
            H = image_tensor.shape[2]
            bon_pix = (bon / np.pi + 0.5) * H - 0.5
            return bon_pix, cor.cpu().numpy()[0]

    def calculate_pitch(self, bon_pix):
        floor_line = bon_pix[0]
        x = np.arange(len(floor_line))
        slope, intercept = np.polyfit(x, floor_line, 1)
        pitch_angle = np.arctan(slope / (512 / len(floor_line))) * 180 / np.pi
        return pitch_angle, floor_line

    def visualize_horizon(self, image, floor_line):
        img_vis = image.copy()
        h, w = img_vis.shape[:2]
        # Масштабируем floor_line под ширину изображения
        x_coords = np.linspace(0, w-1, len(floor_line)).astype(int)
        y_coords = np.clip(floor_line, 0, h-1).astype(int)
        for i in range(1, len(x_coords)):
            cv2.line(img_vis, (x_coords[i-1], y_coords[i-1]), (x_coords[i], y_coords[i]), (0, 255, 0), 2)
        return img_vis

    def process_image(self, image_path):
        image_tensor, resized_img = self.preprocess_image(image_path)
        bon_pix, cor = self.detect_horizon(image_tensor)
        pitch_angle, floor_line = self.calculate_pitch(bon_pix)
        vis_img = self.visualize_horizon(resized_img, floor_line)
        return {
            'pitch': pitch_angle,
            'horizon_line': floor_line,
            'processed_image': resized_img,
            'visualized_image': vis_img
        }

class PitchDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pitch Detector - HorizonNet (Оригинал)")
        self.root.geometry("1200x800")
        self.detector = PitchDetector()
        self.current_image_path = None
        self.processed_result = None
        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        control_frame.grid(row=0, column=0, rowspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        ttk.Button(control_frame, text="Выбрать изображение", command=self.select_image).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="Resize + Crop", command=self.resize_crop).grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(control_frame, text="Detect Pitch", command=self.detect_pitch).grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        model_frame = ttk.LabelFrame(control_frame, text="Модель", padding="5")
        model_frame.grid(row=3, column=0, pady=(20, 0), sticky=(tk.W, tk.E))
        self.model_status = ttk.Label(model_frame, text="Статус: Загрузка...")
        self.model_status.grid(row=0, column=0, pady=5)
        device_text = f"Устройство: {self.detector.device}"
        ttk.Label(model_frame, text=device_text).grid(row=1, column=0, pady=2)
        result_frame = ttk.LabelFrame(main_frame, text="Результаты", padding="10")
        result_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.result_text = tk.Text(result_frame, height=8, width=40)
        self.result_text.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        image_frame = ttk.LabelFrame(main_frame, text="Изображение", padding="10")
        image_frame.grid(row=1, column=1, rowspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        self.image_label = ttk.Label(image_frame, text="Выберите изображение")
        self.image_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.columnconfigure(0, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(0, weight=1)
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

    def load_model(self):
        try:
            if self.detector.load_model():
                self.model_status.config(text="Статус: Модель загружена ✓")
            else:
                self.model_status.config(text="Статус: Ошибка загрузки ✗")
        except Exception as e:
            self.model_status.config(text=f"Статус: Ошибка ✗")
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")

    def select_image(self):
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
        try:
            if processed_image is not None:
                img = processed_image
            else:
                img = cv2.imread(image_path)
                if img is None:
                    raise ValueError("Не удалось загрузить изображение")
                height, width = img.shape[:2]
                img = img[:, :width//2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            display_height = 400
            aspect_ratio = img_rgb.shape[1] / img_rgb.shape[0]
            display_width = int(display_height * aspect_ratio)
            img_resized = cv2.resize(img_rgb, (display_width, display_height))
            pil_img = Image.fromarray(img_resized)
            self.photo = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.photo, text="")
        except Exception as e:
            self.image_label.config(text=f"Ошибка загрузки изображения: {e}")

    def resize_crop(self):
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение")
            return
        try:
            image_tensor, resized_img = self.detector.preprocess_image(self.current_image_path)
            self.display_image(self.current_image_path, resized_img)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Изображение обработано:\n")
            self.result_text.insert(tk.END, f"- Размер: {resized_img.shape[1]}x{resized_img.shape[0]}\n")
            self.result_text.insert(tk.END, "- Готово для анализа\n")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки изображения: {e}")

    def detect_pitch(self):
        if not self.current_image_path:
            messagebox.showwarning("Предупреждение", "Сначала выберите изображение")
            return
        if self.detector.model is None:
            messagebox.showerror("Ошибка", "Модель не загружена")
            return
        try:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "Анализ изображения...\n")
            self.root.update()
            result = self.detector.process_image(self.current_image_path)
            self.processed_result = result
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "=== РЕЗУЛЬТАТЫ АНАЛИЗА ===\n\n")
            self.result_text.insert(tk.END, f"Pitch угол: {result['pitch']:.2f}°\n\n")
            if result['pitch'] > 0:
                self.result_text.insert(tk.END, "Камера наклонена ВВЕРХ\n")
            elif result['pitch'] < 0:
                self.result_text.insert(tk.END, "Камера наклонена ВНИЗ\n")
            else:
                self.result_text.insert(tk.END, "Камера горизонтальна\n")
            self.result_text.insert(tk.END, f"\nДетали:\n")
            self.result_text.insert(tk.END, f"- Средний наклон: {abs(result['pitch']):.2f}°\n")
            self.result_text.insert(tk.END, f"- Диапазон сигнала: {np.min(result['horizon_line']):.3f} - {np.max(result['horizon_line']):.3f}\n")
            # Автоматическая визуализация линии горизонта
            self.display_image(self.current_image_path, result['visualized_image'])
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка определения pitch: {e}")

def main():
    root = tk.Tk()
    app = PitchDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
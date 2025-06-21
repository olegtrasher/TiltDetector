import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
import cv2
from PIL import Image, ImageTk
import os
from ulsd_detector import ULSDDetector

class ULSD_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ULSD: Line Detection & Pitch Calculator")
        self.root.geometry("1200x800")

        self.detector = ULSDDetector()
        self.original_image_path = None
        
        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        main_frame.columnconfigure(1, weight=3) # Give more weight to image frame
        main_frame.rowconfigure(1, weight=1)

        # --- Control Frame ---
        control_frame = ttk.LabelFrame(main_frame, text="Управление", padding="10")
        control_frame.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 10))

        ttk.Button(control_frame, text="1. Выбрать изображение", command=self.select_image).pack(fill=tk.X, pady=5)
        ttk.Button(control_frame, text="2. Подготовить (Обрезать и изменить размер)", command=self.prepare_image).pack(fill=tk.X, pady=5)
        
        # Поле для ввода порога качества линий
        threshold_frame = ttk.Frame(control_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(threshold_frame, text="Порог качества линий:").pack(anchor=tk.W)
        threshold_input_frame = ttk.Frame(threshold_frame)
        threshold_input_frame.pack(fill=tk.X)
        
        self.threshold_var = tk.StringVar(value="0.65")
        self.threshold_entry = ttk.Entry(threshold_input_frame, textvariable=self.threshold_var, width=8)
        self.threshold_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Label(threshold_input_frame, text="(0.1 - 0.9)", font=("Arial", 8), foreground="gray").pack(side=tk.LEFT)
        
        # Чекбокс для анализа ломаных линий
        self.analyze_curvature_var = tk.BooleanVar(value=False)  # по умолчанию отключено
        curvature_checkbox = ttk.Checkbutton(control_frame, 
                                           text="Разбивать ломаные линии на прямые сегменты", 
                                           variable=self.analyze_curvature_var)
        curvature_checkbox.pack(anchor=tk.W, pady=2)
        
        ttk.Button(control_frame, text="3. Найти линии (ULSD)", command=self.detect_lines).pack(fill=tk.X, pady=5)
        
        # Кнопка 4: Интерактивный выбор линий для вычисления угла
        self.interactive_button = tk.Button(
            control_frame, 
            text="4. Вычислить угол наклона", 
            command=self.interactive_line_selection,
            font=("Arial", 12),
            bg="#E5FFE5",
            relief="solid",
            borderwidth=1,
            state="disabled"
        )
        self.interactive_button.pack(fill="x", pady=2)
        
        # Разделитель
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # НОВОЕ ТЕКСТОВОЕ ПОЛЕ для результатов
        ttk.Label(control_frame, text="Результат анализа:").pack(anchor=tk.W)
        self.result_text = scrolledtext.ScrolledText(control_frame, height=8, width=35, wrap=tk.WORD, 
                                                   state=tk.NORMAL, cursor="xterm")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Делаем текст выделяемым и копируемым
        self.result_text.bind("<Control-a>", self.select_all_text)
        self.result_text.bind("<Control-c>", lambda e: self.result_text.event_generate("<<Copy>>"))
        
        # Добавляем информацию о камере
        ttk.Separator(control_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        info_text = "Камера: Canon RF 5.2mm F/2.8 L\nDual Fisheye Lens\n\nТочность: ±1°\nДиапазон: 0° до 90°"
        ttk.Label(control_frame, text=info_text, font=("Arial", 8), foreground="gray").pack(anchor=tk.W)

        # --- Image Frame ---
        image_frame = ttk.LabelFrame(main_frame, text="Изображение", padding="10")
        image_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        
        # Настраиваем сетку внутри рамки, чтобы виджет мог растягиваться
        image_frame.columnconfigure(0, weight=1)
        image_frame.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(image_frame, text="Выберите стерео-изображение")
        # Используем grid и sticky, чтобы виджет растягивался
        self.image_label.grid(row=0, column=0, sticky="nsew")

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите стерео-изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            self.original_image_path = file_path
            self.display_image(file_path)
            self.log_result("✓ Изображение выбрано")

    def prepare_image(self):
        if not self.original_image_path:
            self.log_result("⚠ Сначала выберите изображение (Шаг 1)")
            return
        
        prepared_path, prepared_img_data = self.detector.prepare_image(self.original_image_path)
        if prepared_path:
            self.display_image(image_data=prepared_img_data)
            self.log_result("✓ Изображение подготовлено\n  Размер: 1024x512 для ULSD")
        else:
            self.log_result("✗ Ошибка подготовки изображения")

    def detect_lines(self):
        if not self.detector.prepared_image_path:
            self.log_result("⚠ Сначала подготовьте изображение (Шаг 2)")
            return

        # Получаем порог качества от пользователя
        try:
            threshold = float(self.threshold_var.get())
            if threshold < 0.1 or threshold > 0.9:
                self.log_result("⚠ Порог должен быть от 0.1 до 0.9")
                return
        except ValueError:
            self.log_result("⚠ Введите корректное числовое значение для порога")
            return

        self.log_result(f"🔍 Запуск детекции линий (порог: {threshold})...")
        self.root.update()

        # Передаем настройки анализа кривизны
        analyze_curvature = self.analyze_curvature_var.get()
        output_path = self.detector.run_detection(threshold=threshold, analyze_curvature=analyze_curvature)
        
        if output_path and os.path.exists(output_path):
            self.display_image(output_path)
            # Подсчитываем количество найденных линий
            line_count = len(self.detector.last_line_pred) if self.detector.last_line_pred is not None else 0
            self.log_result(f"✓ Детекция завершена\n  Найдено линий: {line_count}")
            
            # Активируем кнопки для анализа угла наклона
            self.interactive_button.config(state="normal")
        else:
            self.log_result("✗ Ошибка детекции линий")



    def interactive_line_selection(self):
        """Запускает интерактивный выбор линий"""
        try:
            self.log_result("🖱️ Запуск интерактивного выбора линий...")
            
            # Передаем функцию логирования в детектор
            self.detector.log_function = self.log_result
            
            # Создаем интерактивное окно
            selector_window = self.detector.create_interactive_line_selector()
            
            if selector_window:
                self.log_result("✓ Интерактивное окно открыто")
                self.log_result("  Инструкция:")
                self.log_result("  • Кликните по линиям, которые должны быть вертикальными")
                self.log_result("  • Выбранные линии станут зелеными")
                self.log_result("  • Нужно минимум 2 линии для расчета")
                self.log_result("  • Нажмите 'Вычислить угол' когда закончите выбор")
            else:
                self.log_result("✗ Не удалось открыть интерактивное окно")
                
        except Exception as e:
            self.log_result(f"✗ Ошибка интерактивного выбора: {e}")
            print(f"Ошибка интерактивного выбора: {e}")

    def log_result(self, message):
        """Добавляет сообщение в текстовое поле результатов"""
        self.result_text.insert(tk.END, message + "\n\n")
        self.result_text.see(tk.END)
        self.root.update()
    
    def select_all_text(self, event):
        """Выделяет весь текст в поле результатов"""
        self.result_text.tag_add(tk.SEL, "1.0", tk.END)
        self.result_text.mark_set(tk.INSERT, "1.0")
        self.result_text.see(tk.INSERT)
        return 'break'

    def display_image(self, image_path=None, image_data=None):
        try:
            # Принудительно обновляем геометрию, чтобы получить правильные размеры
            self.root.update_idletasks()

            if image_data is not None:
                img = image_data
            elif image_path:
                img = cv2.imread(image_path)
            else:
                return

            if img is None:
                raise ValueError("Не удалось загрузить данные изображения")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # --- Новая логика масштабирования ---
            container_width = self.image_label.winfo_width()
            container_height = self.image_label.winfo_height()
            # Задаем размеры по умолчанию, если окно еще не отрисовано
            if container_width < 10 or container_height < 10:
                container_width, container_height = 600, 600

            img_h, img_w = img_rgb.shape[:2]
            img_aspect = img_w / img_h
            container_aspect = container_width / container_height

            # Масштабируем, чтобы вписать в контейнер, сохраняя пропорции
            if img_aspect > container_aspect:
                # Изображение шире контейнера -> вписываем по ширине
                display_w = container_width
                display_h = int(display_w / img_aspect)
            else:
                # Изображение выше контейнера -> вписываем по высоте
                display_h = container_height
                display_w = int(display_h * img_aspect)

            # Предотвращаем нулевой размер
            if display_w < 1: display_w = 1
            if display_h < 1: display_h = 1
            
            img_resized = cv2.resize(img_rgb, (display_w, display_h))
            # ---
            
            pil_img = Image.fromarray(img_resized)
            self.photo = ImageTk.PhotoImage(pil_img)
            self.image_label.config(image=self.photo, text="")
        except Exception as e:
            self.log_result(f"✗ Ошибка отображения: {e}")


def main():
    root = tk.Tk()
    app = ULSD_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
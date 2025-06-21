import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import os
from ulsd_detector import ULSDDetector

class ULSD_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ULSD: Line Detection")
        self.root.geometry("1000x700")

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
        ttk.Button(control_frame, text="3. Найти линии (ULSD)", command=self.detect_lines).pack(fill=tk.X, pady=5)

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

    def prepare_image(self):
        if not self.original_image_path:
            print("Внимание: Сначала выберите изображение (Шаг 1).")
            return
        
        prepared_path, prepared_img_data = self.detector.prepare_image(self.original_image_path)
        if prepared_path:
            self.display_image(image_data=prepared_img_data)
            print(f"Изображение подготовлено и сохранено в: {prepared_path}")
        else:
            print("Ошибка: Не удалось подготовить изображение.")

    def detect_lines(self):
        if not self.detector.prepared_image_path:
            print("Внимание: Сначала подготовьте изображение (Шаг 2).")
            return

        print("Запускается процесс детекции. Это может занять некоторое время...")
        self.root.update()

        output_path = self.detector.run_detection()
        
        if output_path and os.path.exists(output_path):
            self.display_image(output_path)
            print(f"Детекция завершена. Результат в: {output_path}")
        else:
            print("Ошибка: Не удалось найти результат детекции. Проверьте консоль.")


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
                container_width, container_height = 700, 650

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
            messagebox.showerror("Ошибка отображения", f"Не удалось показать изображение: {e}")


def main():
    root = tk.Tk()
    app = ULSD_GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 
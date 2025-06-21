# Быстрая установка TiltDetector

## Шаг 1: Клонирование репозитория

```bash
git clone --recursive https://github.com/your-username/TiltDetector.git
cd TiltDetector
```

**Важно:** Используйте флаг `--recursive` для автоматического клонирования подмодуля ULSD.

## Шаг 2: Установка зависимостей

```bash
pip install -r requirements.txt
```

## Шаг 3: Скачивание моделей ULSD

Скачайте предобученные модели и поместите их в папку `Unified-Line-Segment-Detection/model/`:

- `spherical.pkl` - для панорамных изображений (основная)
- `fisheye.pkl` - для fisheye камер  
- `pinhole.pkl` - для обычных камер

**Ссылка на модели:** [ULSD Models](https://github.com/lh9171338/Unified-Line-Segment-Detection#models)

## Шаг 4: Запуск

```bash
python pitch_detector.py
```

## Если что-то пошло не так

### Проблема с подмодулем ULSD

Если папка `Unified-Line-Segment-Detection` пуста:

```bash
git submodule update --init --recursive
```

### Отсутствуют модели

Создайте папку для моделей:

```bash
mkdir Unified-Line-Segment-Detection/model
```

Затем скачайте и поместите туда файлы `.pkl`.

### Ошибки импорта

Убедитесь, что все зависимости установлены:

```bash
pip install torch torchvision opencv-python pillow matplotlib numpy
```

## Тестирование

Используйте любое стерео-изображение в формате Side-by-Side для тестирования работы системы. 
#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы Pitch Detector
"""

import os
import sys
import torch
import numpy as np
from horizon_net_utils import analyze_image_pitch, ImagePreprocessor, PitchCalculator

def test_image_preprocessing():
    """Тест предобработки изображения"""
    print("=== Тест предобработки изображения ===")
    
    # Проверяем наличие тестового изображения
    test_image = "test_621_sq2.png"
    if not os.path.exists(test_image):
        print(f"Ошибка: Файл {test_image} не найден")
        return False
    
    try:
        preprocessor = ImagePreprocessor()
        image_tensor, processed_image = preprocessor.preprocess_stereo_image(test_image)
        
        print(f"✓ Изображение загружено успешно")
        print(f"✓ Размер тензора: {image_tensor.shape}")
        print(f"✓ Размер обработанного изображения: {processed_image.shape}")
        print(f"✓ Диапазон значений тензора: {image_tensor.min():.3f} - {image_tensor.max():.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка предобработки: {e}")
        return False

def test_pitch_calculation():
    """Тест вычисления pitch"""
    print("\n=== Тест вычисления pitch ===")
    
    try:
        calculator = PitchCalculator()
        
        # Создаем тестовый сигнал горизонта (наклон вверх)
        test_signal = np.linspace(0.3, 0.7, 128)
        pitch_data = calculator.calculate_pitch_advanced(test_signal)
        
        print(f"✓ Pitch вычислен успешно")
        print(f"✓ Угол pitch: {pitch_data['pitch']:.2f}°")
        print(f"✓ Уверенность: {pitch_data['confidence']:.2f}")
        print(f"✓ PCA метод: {pitch_data['pitch_pca']:.2f}°")
        print(f"✓ Линейная регрессия: {pitch_data['pitch_lr']:.2f}°")
        
        return True
        
    except Exception as e:
        print(f"✗ Ошибка вычисления pitch: {e}")
        return False

def test_model_loading():
    """Тест загрузки модели"""
    print("\n=== Тест загрузки модели ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'pretrained_models/resnet50_rnn__zind.pth'
    
    if not os.path.exists(model_path):
        print(f"✗ Файл модели {model_path} не найден")
        return False
    
    try:
        from horizon_net_utils import load_horizon_model
        model = load_horizon_model(model_path, device)
        
        if model is not None:
            print(f"✓ Модель загружена успешно")
            print(f"✓ Устройство: {device}")
            print(f"✓ Количество параметров: {sum(p.numel() for p in model.parameters()):,}")
            return True
        else:
            print("✗ Не удалось загрузить модель")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка загрузки модели: {e}")
        return False

def test_full_analysis():
    """Тест полного анализа изображения"""
    print("\n=== Тест полного анализа ===")
    
    test_image = "test_621_sq2.png"
    model_path = 'pretrained_models/resnet50_rnn__zind.pth'
    
    if not os.path.exists(test_image):
        print(f"✗ Тестовое изображение не найдено")
        return False
    
    if not os.path.exists(model_path):
        print(f"✗ Файл модели не найден")
        return False
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        result = analyze_image_pitch(test_image, model_path, device)
        
        if result is not None:
            print(f"✓ Анализ выполнен успешно")
            print(f"✓ Pitch угол: {result['pitch']:.2f}°")
            print(f"✓ Уверенность: {result['confidence']:.2f}")
            print(f"✓ Размер обработанного изображения: {result['processed_image'].shape}")
            return True
        else:
            print("✗ Анализ не удался")
            return False
            
    except Exception as e:
        print(f"✗ Ошибка анализа: {e}")
        return False

def main():
    """Основная функция тестирования"""
    print("Запуск тестов Pitch Detector...\n")
    
    tests = [
        test_image_preprocessing,
        test_pitch_calculation,
        test_model_loading,
        test_full_analysis
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Неожиданная ошибка в тесте: {e}")
    
    print(f"\n=== Результаты тестирования ===")
    print(f"Пройдено тестов: {passed}/{total}")
    
    if passed == total:
        print("✓ Все тесты пройдены успешно!")
        return True
    else:
        print("✗ Некоторые тесты не пройдены")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
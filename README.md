# 🚖 InDrive Car Analyzer

Веб-приложение для анализа состояния автомобилей по фотографиям.  
Использует **YOLOv5** для детекции автомобилей и объектов, а также **Google Gemini API** для генерации итогового отчёта.

---

## ✨ Возможности
- 📷 Загрузка нескольких фото автомобиля через веб-интерфейс  
- 🤖 Определение автомобилей и объектов с помощью YOLOv5  
- 📝 Генерация структурированного отчёта через Gemini (марка, чистота, повреждения и т.д.)  
- 🎨 Современный интерфейс с анимацией и адаптивным дизайном  

---

## 🛠️ Технологии
- [Python 3.13](https://www.python.org/)  
- [FastAPI](https://fastapi.tiangolo.com/) — веб-фреймворк  
- [Uvicorn](https://www.uvicorn.org/) — ASGI-сервер  
- [PyTorch](https://pytorch.org/) + [YOLOv5](https://github.com/ultralytics/yolov5) — компьютерное зрение  
- [Google Generative AI](https://ai.google.dev/) (Gemini)  

---

## 🚀 Запуск проекта

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/username/indrive-car-analyzer.git
   cd indrive-car-analyzer

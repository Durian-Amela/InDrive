import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import base64
import google.generativeai as genai

# 🔑 API-ключ Gemini
GEMINI_API_KEY = "AIzaSyABAVvOfJxJSlbNa9dhfcgSYYFS-vNX80w"
genai.configure(api_key=GEMINI_API_KEY)

# Загрузка YOLOv5
MODEL_PATH = r"C:\Users\dulat\Desktop\data\yolov5s.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)

# FastAPI
app = FastAPI(title="InDrive Car Analyzer")

# ===== Функция анализа через Gemini =====
def gemini_analyze(images: list[Image.Image]) -> str:
    images_data = []
    for img in images:
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_bytes = buffered.getvalue()
        images_data.append({"mime_type": "image/jpeg", "data": base64.b64encode(img_bytes).decode()})
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    prompt = (
        "На основе предоставленных фотографий автомобиля составь единый отчет о его состоянии.\n"
        "Укажи: марку автомобиля (если можно определить), количество фотографий, "
        "состояние чистоты, видимые повреждения, итоговый анализ.\n"
        "Сделай красивое и структурированное оформление."
    )
    response = model_gemini.generate_content(images_data + [prompt])
    return response.text.strip() if response else "Нет ответа от Gemini"

# ===== Главная страница =====
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
    <head>
        <title>InDrive Car Analyzer</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Inter', sans-serif;
                margin:0; padding:0;
                background: linear-gradient(135deg,#e6f9f0,#f0fdf4);
                overflow-x:hidden;
            }
            /* Фон-узор такси */
            body::before {
                content: "";
                position: fixed;
                top:0; left:0; width:100%; height:100%;
                background: url('https://upload.wikimedia.org/wikipedia/commons/6/63/Taxi_icon.svg') repeat;
                opacity: 0.03; /* лёгкий фон */
                z-index:0;
            }
            header {
                display:flex;
                justify-content: space-between;
                align-items:center;
                padding:20px 40px;
                background:#00C853;
                color:white;
                box-shadow:0 4px 10px rgba(0,0,0,0.2);
                animation: slideDown 1s ease;
                position:relative; z-index:1;
            }
            header .logo {
                display:flex; align-items:center; gap:10px;
            }
            header img { height:40px; }
            nav a {
                color:white;
                margin-left:20px;
                text-decoration:none;
                font-weight:600;
                display:flex; align-items:center; gap:5px;
                transition: transform 0.3s, opacity 0.3s;
            }
            nav a svg { width:18px; height:18px; fill:white; }
            nav a:hover { transform: translateY(-3px); opacity:0.8; }
            .container {
                max-width:700px;
                margin:60px auto;
                background: rgba(255,255,255,0.95);
                padding:30px;
                border-radius:20px;
                box-shadow:0 10px 25px rgba(0,0,0,0.15);
                text-align:center;
                animation: fadeIn 1s ease;
                position:relative; z-index:1;
            }
            h2 { color:#00C853; margin-bottom:20px; }
            input[type="file"] {
                display:block;
                margin:20px auto;
                padding:12px;
                border:2px dashed #00C853;
                border-radius:10px;
                width:80%;
                cursor:pointer;
                transition: border-color 0.3s, transform 0.3s;
            }
            input[type="file"]:hover { border-color:#008f5a; transform: scale(1.02);}
            input[type="submit"] {
                background:#00C853;
                color:#fff;
                border:none;
                padding:14px 28px;
                border-radius:12px;
                cursor:pointer;
                font-size:16px;
                font-weight:600;
                transition: background 0.3s, transform 0.2s;
            }
            input[type="submit"]:hover {
                background:#008f5a;
                transform: translateY(-2px);
            }
            @keyframes fadeIn { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
            @keyframes slideDown { from {transform:translateY(-50px); opacity:0;} to {transform:translateY(0); opacity:1;} }
        </style>
    </head>
    <body>
        <header>
            <div class="logo">
                <img src="https://upload.wikimedia.org/wikipedia/commons/2/20/InDrive_Logo.svg" alt="InDrive Logo">
                <span>InDrive</span>
            </div>
            <nav>
                <a href="#"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/></svg>Главная</a>
                <a href="#"><svg viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16"/></svg>Услуги</a>
                <a href="#"><svg viewBox="0 0 24 24"><path d="M2 12h20"/></svg>Контакты</a>
                <a href="#"><svg viewBox="0 0 24 24"><path d="M12 2v20"/></svg>Адрес</a>
                <a href="#"><svg viewBox="0 0 24 24"><circle cx="12" cy="8" r="4"/><path d="M4 22c0-4 8-6 8-6s8 2 8 6"/></svg>Профиль</a>
            </nav>
        </header>
        <div class="container">
            <h2>Анализ состояния автомобиля</h2>
            <form action="/analyze" enctype="multipart/form-data" method="post">
                <input name="files" type="file" accept="image/*" multiple required>
                <input type="submit" value="Анализировать">
            </form>
        </div>
    </body>
    </html>
    """

# ===== Анализ =====
@app.post("/analyze", response_class=HTMLResponse)
async def analyze(files: list[UploadFile] = File(...)):
    images = []
    for file in files:
        img = Image.open(file.file).convert("RGB")
        images.append(img)

    # YOLO для объектов
    yolo_text_list = []
    for img in images:
        results = model(img)
        preds = results.pandas().xyxy[0]
        objects_list = [f"{row['name']} ({row['confidence']:.2f})" for _, row in preds.iterrows()]
        yolo_text_list.append(", ".join(objects_list) if objects_list else "Объекты не найдены")
    yolo_summary = "; ".join(yolo_text_list)

    # Gemini анализ
    gemini_text = gemini_analyze(images) if images else "На фотографиях не найдено машин."

    # Первое изображение
    buffered = io.BytesIO()
    images[0].save(buffered, format="JPEG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return f"""
    <html>
    <head>
        <title>Результаты InDrive</title>
        <style>
            body {{ font-family:'Inter',sans-serif; background: linear-gradient(135deg,#e6f9f0,#f0fdf4); padding:30px; }}
            .container {{ max-width:750px; margin:40px auto; background: rgba(255,255,255,0.95); padding:30px;
                          border-radius:20px; box-shadow:0 10px 25px rgba(0,0,0,0.2); animation: fadeIn 1s ease; }}
            h2 {{ color:#00C853; text-align:center; }}
            .image-block {{ text-align:center; margin-bottom:20px; animation:fadeIn 1s ease; }}
            img {{ border-radius:16px; box-shadow:0 6px 15px rgba(0,0,0,0.2); transition: transform 0.3s; }}
            img:hover {{ transform: scale(1.02); }}
            .analysis {{ white-space: pre-line; line-height:1.6; color:#333; animation:fadeIn 1.5s ease; }}
            a {{ display:block; text-align:center; margin-top:20px; color:#00C853; text-decoration:none; font-weight:bold; transition: color 0.3s; }}
            a:hover {{ color:#008f5a; }}
            @keyframes fadeIn {{ from {{opacity:0; transform:translateY(20px);}} to {{opacity:1; transform:translateY(0);}} }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Итоговый анализ автомобиля</h2>
            <div class="image-block">
                <img src="data:image/jpeg;base64,{img_base64}" style="max-width:100%;">
            </div>
            <div class="analysis">{gemini_text}</div>
            <a href="/">Загрузить другие фото</a>
        </div>
    </body>
    </html>
    """

# ===== Запуск =====
if __name__ == "__main__":
    uvicorn.run("case1:app", host="127.0.0.1", port=8000, reload=True)
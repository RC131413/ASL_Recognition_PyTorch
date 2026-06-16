from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import json

app = FastAPI(title="ASL Alphabet Classifier API")

templates = Jinja2Templates(directory="templates")

print("Ładowanie modelu ONNX i mapowania klas...")
ort_session = ort.InferenceSession("asl_classifier.onnx", providers=['CPUExecutionProvider'])

# Plik z tłumaczeniem indeksów na litery
with open("mapping.json", "r") as f:
    idx_to_class = json.load(f)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Wycinanie idealnego kwadratu ze środka
    width, height = img.size
    new_size = min(width, height)

    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2

    img = img.crop((left, top, right, bottom))
    img = img.resize((224, 224))

    # Zamiana na tablicę NumPy i normalizacja kolorów
    img_data = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_data = ((img_data - mean) / std).astype(np.float32)

    img_data = np.transpose(img_data, (2, 0, 1))
    img_data = np.expand_dims(img_data, axis=0)

    return img_data

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict", response_class=HTMLResponse)
async def predict_sign(request: Request, file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()

        input_data = preprocess_image(image_bytes)
        outputs = ort_session.run(None, {"input": input_data})

        predicted_idx = np.argmax(outputs[0])
        predicted_letter = idx_to_class[str(predicted_idx)]

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"predicted_letter": predicted_letter}
        )
    except Exception as e:
         return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"error_message": str(e)}
         )
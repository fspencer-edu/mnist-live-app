from __future__ import annotations

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import numpy as np

from model_loader import get_model
from preprocess import preprocess_image_bytes
from schemas import PredictionResponse

app = FastAPI(title="MNIST Live Probability API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    x = preprocess_image_bytes(image_bytes)

    model = get_model()
    probs = model.predict(x, verbose=0)[0]

    predicted_digit = int(np.argmax(probs))
    confidence = float(np.max(probs))

    return PredictionResponse(
        predicted_digit=predicted_digit,
        confidence=confidence,
        probabilities={str(i): float(probs[i]) for i in range(10)},
    )
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, conint, confloat
from typing import List
import pandas as pd
import joblib
from pathlib import Path
import io
import subprocess

app = FastAPI(title="Heart Disease Classifier API")

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models/heart_model.pkl"

# Загружаем модель
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        raise RuntimeError("Модель ещё не обучена. Используйте /train для обучения.")

model = load_model()

# Pydantic схема
class HeartData(BaseModel):
    age: conint(ge=0, le=120)
    sex: conint(ge=0, le=1)
    cp: conint(ge=0, le=3)
    trestbps: conint(ge=50, le=250)
    chol: conint(ge=50, le=600)
    fbs: conint(ge=0, le=1)
    restecg: conint(ge=0, le=2)
    thalach: conint(ge=60, le=250)
    exang: conint(ge=0, le=1)
    oldpeak: confloat(ge=0, le=10)
    slope: conint(ge=0, le=2)
    ca: conint(ge=0, le=4)
    thal: conint(ge=0, le=3)

# ----------------------
# Эндпоинты
# ----------------------

@app.get("/")
def root():
    return {"status": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: HeartData):
    df = pd.DataFrame([data.dict()])
    try:
        pred = model.predict(df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"prediction": int(pred)}

@app.post("/predict_batch")
def predict_batch(data: List[HeartData]):
    df = pd.DataFrame([item.dict() for item in data])
    try:
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predictions": preds.tolist()}

@app.post("/predict_file")
def predict_file(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(io.BytesIO(file.file.read()))
        preds = model.predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"predictions": preds.tolist()}

@app.post("/train")
def train():
    """
    Переобучение модели внутри Docker. Должен быть CSV в папке data/heart.csv
    """
    try:
        subprocess.run(["python", str(BASE_DIR / "src/train_model.py")], check=True)
        global model
        model = load_model()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обучения: {e}")
    return {"status": "Модель успешно обучена"}

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "heart_model.pkl"
DATA_PATH = BASE_DIR / "data/heart.csv"

# Загружаем датасет
df = pd.read_csv(DATA_PATH)
X = df.drop("target", axis=1)
y = df["target"]

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Сохраняем
joblib.dump(model, MODEL_PATH)
print("Модель успешно обучена и сохранена в", MODEL_PATH)
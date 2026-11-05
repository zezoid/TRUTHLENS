from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, List, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib


def load_model(model_path: str = "models/best_model.joblib"):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}.")
    return joblib.load(model_path)


def predict_article(article_text: str, model_path: str = "models/best_model.joblib") -> str:
    model = load_model(model_path)
    prediction = model.predict([article_text])[0]
    return "FAKE NEWS" if prediction == 0 else "REAL NEWS"


def predict_many(article_texts: Iterable[str], model_path: str = "models/best_model.joblib") -> List[str]:
    model = load_model(model_path)
    predictions = model.predict(list(article_texts))
    return ["FAKE NEWS" if pred == 0 else "REAL NEWS" for pred in predictions]


if __name__ == "__main__":
    sample_text = "Government officials announced a new plan to improve public services across the country."
    print(predict_article(sample_text))

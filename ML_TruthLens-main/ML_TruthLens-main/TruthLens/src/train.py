from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.evaluate import calculate_metrics, plot_confusion_matrix, plot_model_comparison, save_classification_report
from src.feature_extraction import build_tfidf_vectorizer
from src.preprocessing import TextPreprocessor, prepare_combined_text


MODEL_SPECS = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=None),
    "KNN": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(random_state=42, class_weight="balanced"),
}


def load_and_merge_datasets(raw_dir: str = "datasets/raw") -> pd.DataFrame:
    raw_path = Path(raw_dir)
    fake_path = raw_path / "Fake.csv"
    true_path = raw_path / "True.csv"

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Dataset files were not found. Download Fake.csv and True.csv into datasets/raw/ before training."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    combined = pd.concat([fake_df, true_df], ignore_index=True)
    combined = combined[["title", "text", "subject", "date", "label"]]
    combined = prepare_combined_text(combined)
    combined = combined.drop_duplicates(subset=["combined_text"], keep="first")
    return combined


def make_model_pipeline(model_name: str, vectorizer: TfidfVectorizer):
    if model_name == "KNN":
        return Pipeline(
            [
                ("preprocessor", TextPreprocessor()),
                ("tfidf", vectorizer),
                ("svd", TruncatedSVD(n_components=50, random_state=42)),
                ("model", MODEL_SPECS[model_name]),
            ]
        )
    if model_name == "Linear SVM":
        return Pipeline(
            [
                ("preprocessor", TextPreprocessor()),
                ("tfidf", vectorizer),
                ("model", MODEL_SPECS[model_name]),
            ]
        )
    return Pipeline(
        [
            ("preprocessor", TextPreprocessor()),
            ("tfidf", vectorizer),
            ("model", MODEL_SPECS[model_name]),
        ]
    )


def train_all_models(
    dataset_path: str = "datasets/processed/news_dataset.csv",
    random_state: int = 42,
    test_size: float = 0.2,
    output_dir: str = "outputs",
):
    if Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)
    else:
        df = load_and_merge_datasets()
        processed_dir = Path(dataset_path).parent
        processed_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_path, index=False)

    X = df["combined_text"].fillna("")
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    results = []
    model_artifacts = {}
    vectorizer_config = {
        "max_features": 20000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "stop_words": "english",
    }

    for model_name in MODEL_SPECS:
        vectorizer = build_tfidf_vectorizer(**vectorizer_config)
        pipeline = make_model_pipeline(model_name, vectorizer)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = None
        if hasattr(pipeline, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X_test)[:, 1]
            except Exception:
                y_prob = None
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        metrics["Model"] = model_name
        metrics["Fake_Recall"] = recall_score(y_test, y_pred, pos_label=0, zero_division=0)
        metrics["Macro_F1"] = f1_score(y_test, y_pred, average="macro", zero_division=0)
        results.append(metrics)

        plot_confusion_matrix(y_test, y_pred, output_path=f"{output_dir}/confusion_matrices/{model_name.lower().replace(' ', '_')}_cm.png")
        save_classification_report(model_name, y_test, y_pred, output_dir=f"{output_dir}/reports")
        model_artifacts[model_name] = pipeline

    results_df = pd.DataFrame(results)
    results_df = results_df[["Model", "accuracy", "precision", "recall", "f1", "Fake_Recall", "Macro_F1", "roc_auc"]]

    results_df.to_csv(f"{output_dir}/reports/model_comparison.csv", index=False)
    plot_model_comparison(results_df.rename(columns={"f1": "f1", "accuracy": "accuracy"}).to_dict("records"), metric="f1", output_path=f"{output_dir}/graphs/model_comparison.png")

    best_model_name = results_df.sort_values(["Macro_F1", "Fake_Recall", "accuracy"], ascending=False).iloc[0]["Model"]
    best_model = model_artifacts[best_model_name]
    joblib.dump(best_model, "models/best_model.joblib")

    return {
        "best_model_name": best_model_name,
        "results": results_df,
        "best_model": best_model,
    }


def load_saved_model(path: str = "models/best_model.joblib"):
    return joblib.load(path)


if __name__ == "__main__":
    train_all_models()

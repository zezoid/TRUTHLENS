from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score


def calculate_metrics(y_true, y_pred, y_proba: Optional[List[float]] = None) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": float("nan"),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics


def save_classification_summary(report: Dict[str, float], output_path: str) -> None:
    df = pd.DataFrame([report])
    df.to_csv(output_path, index=False)


def plot_confusion_matrix(y_true, y_pred, labels=None, output_path: str = "outputs/confusion_matrices/confusion_matrix.png") -> None:
    labels = labels or [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def plot_model_comparison(results: List[dict], metric: str = "f1", output_path: str = "outputs/graphs/model_comparison.png") -> None:
    if not results:
        return
    df = pd.DataFrame(results)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Model", y=metric, palette="viridis")
    plt.xticks(rotation=25, ha="right")
    plt.ylabel(metric.capitalize())
    plt.title(f"Model Comparison by {metric.capitalize()}")
    plt.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path)
    plt.close()


def save_classification_report(clf_name: str, y_true, y_pred, output_dir: str = "outputs/reports") -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_true, y_pred, target_names=["Fake", "Real"], output_dict=True)
    report_df = pd.DataFrame(report).T
    final_path = path / f"{clf_name.lower().replace(' ', '_')}_report.csv"
    report_df.to_csv(final_path)
    return str(final_path)


from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def main():
    # Load validation predictions
    data = np.load(OUTPUT_DIR / "evaluation_data.npz")

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    image_paths = data["image_paths"]

    # Calculate performance metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print("\nBaseline Performance")
    print("-------------------------")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")

    # Save metrics to CSV
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Score": [accuracy, precision, recall, f1]
    })

    metrics.to_csv(
        OUTPUT_DIR / "baseline_metrics.csv",
        index=False
    )

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\nConfusion Matrix:")
    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Authentic", "Counterfeit"]
    )

    display.plot(cmap="Blues", values_format="d")

    plt.title("Baseline Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Generate performance bar chart
    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        metrics["Metric"],
        metrics["Score"] * 100
    )

    plt.bar_label(bars, fmt="%.1f%%", padding=3)
    plt.ylim(0, 110)
    plt.ylabel("Percentage (%)")
    plt.title("Baseline Classification Performance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_performance.png", dpi=300)
    plt.close()

    # Identify incorrectly classified images
    results = pd.DataFrame({
        "image_identifier": image_paths,
        "true_label": y_true,
        "predicted_label": y_pred
    })

    incorrect = results[
        results["true_label"] != results["predicted_label"]
    ].copy()

    incorrect["error_type"] = np.where(
        incorrect["true_label"] == 1,
        "False Negative",
        "False Positive"
    )

    incorrect.to_csv(
        OUTPUT_DIR / "misclassified_images.csv",
        index=False
    )

    print("\nMisclassified Images:")
    print(incorrect.to_string(index=False))

    print("\nEvaluation complete!")
    print("Results saved to outputs/")


if __name__ == "__main__":
    main()
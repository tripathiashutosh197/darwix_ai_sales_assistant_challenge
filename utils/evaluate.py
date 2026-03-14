"""
utils/evaluate.py
------------------
Evaluates the emotion detector on a labeled CSV test set and
prints accuracy, F1 per class, and a confusion matrix.

Usage:
    python utils/evaluate.py --csv data/val.csv --model transformer
    python utils/evaluate.py --csv data/val.csv --model vader
"""

import argparse
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def evaluate(csv_path: str, model_type: str = "transformer"):
    from emotion import get_detector

    detector = get_detector(model_type)
    labels_true = []
    labels_pred = []

    print(f"Loading test data from {csv_path}...")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [(r[0].strip(), r[1].strip().lower()) for r in reader if len(r) >= 2]

    print(f"Running {model_type} detector on {len(rows)} samples...")
    for i, (text, true_label) in enumerate(rows):
        result = detector.detect(text)
        labels_true.append(true_label)
        labels_pred.append(result.label)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(rows)}...")

    # Metrics
    try:
        from sklearn.metrics import (
            accuracy_score, classification_report, confusion_matrix
        )
    except ImportError:
        print("Install scikit-learn: pip install scikit-learn")
        return

    acc = accuracy_score(labels_true, labels_pred)
    print(f"\n{'='*50}")
    print(f"  Model:    {model_type}")
    print(f"  Samples:  {len(rows)}")
    print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print(f"{'='*50}\n")
    print(classification_report(labels_true, labels_pred, zero_division=0))

    # Confusion matrix
    classes = sorted(set(labels_true + labels_pred))
    cm = confusion_matrix(labels_true, labels_pred, labels=classes)
    col_w = 10
    print("Confusion matrix (rows=true, cols=predicted):")
    print(" " * col_w + "".join(f"{c[:col_w]:>{col_w}}" for c in classes))
    for i, row_label in enumerate(classes):
        print(f"{row_label[:col_w]:<{col_w}}" + "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(classes))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",   required=True, help="Path to labeled test CSV")
    parser.add_argument("--model", default="transformer", choices=["vader", "transformer"])
    args = parser.parse_args()
    evaluate(args.csv, args.model)

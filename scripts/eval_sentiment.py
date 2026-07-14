"""
Script di valutazione del modulo Sentiment sul dataset sintetico.

Uso:
    python3 scripts/eval_sentiment.py

Confronta le predizioni di `sentiment_module.compute_severity_score` con la
ground truth `expected_severity` del CSV, e produce:
- Overall accuracy
- Matrice di confusione formattata
- Per-class precision/recall/f1
- Lista degli errori (per debug/tuning)
"""

from __future__ import annotations
import sys
import os
import csv

from collections import defaultdict
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.nlp.sentiment_module import compute_severity_score, severity_to_category


DATASET_PATH = Path("data/nlp/synthetic_complaints.csv")
CATEGORIES = ["low", "medium", "high"]


def evaluate() -> None:
    """Esegue la valutazione end-to-end sul dataset sintetico."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset non trovato: {DATASET_PATH}")

    predictions: list[tuple[str, str, str, float]] = []
    # (row_id, expected, predicted, score)

    print(f"Valuto sentiment_module su {DATASET_PATH}...")
    with DATASET_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, 1):
            score = compute_severity_score(row["text"])
            pred = severity_to_category(score)
            predictions.append((row["id"], row["expected_severity"], pred, score))
            if i % 10 == 0:
                print(f"  processato {i}...")

    print()
    # ------------------------------------------------------------
    # Overall accuracy
    # ------------------------------------------------------------
    correct = sum(1 for _, exp, pred, _ in predictions if exp == pred)
    total = len(predictions)
    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} ({100*accuracy:.1f}%)")

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for _, exp, pred, _ in predictions:
        confusion[(exp, pred)] += 1

    print("\nMatrice di confusione (righe = expected, colonne = predicted):")
    header = f"  {'':10}" + "".join(f" {c:>8}" for c in CATEGORIES)
    print(header)
    for exp in CATEGORIES:
        row_str = f"  {exp:10}"
        for pred in CATEGORIES:
            row_str += f" {confusion[(exp, pred)]:>8}"
        print(row_str)

    # ------------------------------------------------------------
    # Per-class precision/recall/f1
    # ------------------------------------------------------------
    print("\nMetriche per classe:")
    print(f"  {'category':10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for cat in CATEGORIES:
        tp = confusion[(cat, cat)]
        fp = sum(confusion[(exp, cat)] for exp in CATEGORIES if exp != cat)
        fn = sum(confusion[(cat, pred)] for pred in CATEGORIES if pred != cat)
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"  {cat:10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {support:>10}")

    # ------------------------------------------------------------
    # Errori (mostrali per debug)
    # ------------------------------------------------------------
    errors = [(rid, exp, pred, score) for rid, exp, pred, score in predictions if exp != pred]
    if errors:
        print(f"\nErrori ({len(errors)}):")
        for rid, exp, pred, score in errors:
            print(f"  [{rid}] expected={exp}, got={pred} (score={score:.3f})")


if __name__ == "__main__":
    evaluate()
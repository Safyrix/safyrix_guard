from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "training_v2.jsonl"

MODELS_DIR = BASE_DIR.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "guardian_text_pipeline_v2.pkl"


def load_dataset(path: Path) -> Tuple[List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
            labels.append(obj["label"])
    return texts, labels


def build_model() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=8000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    print("📌 Loading dataset v2...")
    X, y = load_dataset(DATASET_PATH)

    print("📌 Building model v2...")
    model = build_model()

    print("📌 Training v2...")
    model.fit(X, y)

    print("📌 Saving v2 model...")
    joblib.dump(model, MODEL_PATH)

    print(f"✅ v2 model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

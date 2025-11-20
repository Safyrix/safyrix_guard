import json
import pickle
from pathlib import Path
from typing import Dict, List

MODELS_DIR = Path("models")
DATA_FILE = Path("data") / "training_patterns.json"
MODEL_FILE = MODELS_DIR / "guardian_text_model.pkl"


def load_training_data() -> Dict[str, List[str]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Traning fajl ne postoji: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Ocekivana struktura training fajla je dict {category: [fraze]}")

    # Normalizuj sve fraze na lowercase radi doslednosti
    normalized = {
        category: [phrase.lower() for phrase in phrases]
        for category, phrases in data.items()
    }

    return normalized


def save_model(patterns: Dict[str, List[str]]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(patterns, f)

    print(f"[TRAIN] Model sacuvan u: {MODEL_FILE}")


def main() -> None:
    print("[TRAIN] Ucitavam trening podatke...")
    patterns = load_training_data()

    print("[TRAIN] Broj kategorija:", len(patterns))
    for cat, phrases in patterns.items():
        print(f"  - {cat}: {len(phrases)} fraza")

    save_model(patterns)
    print("[TRAIN] Gotovo.")


if __name__ == "__main__":
    main()

from pathlib import Path
from typing import Dict, List

import joblib


class GuardianTextAnalyzer:
    def __init__(self):
        model_path = Path("models") / "guardian_text_model.pkl"
        if not model_path.exists():
            raise RuntimeError(
                f"Model nije pronadjen na {model_path}. Pokreni prvo 'python ml\\train_model.py'."
            )
        self.pipeline = joblib.load(model_path)

    def analyze(self, message: str) -> Dict:
        proba = self.pipeline.predict_proba([message])[0]
        classes: List[str] = list(self.pipeline.classes_)
        # max klasa
        best_idx = proba.argmax()
        risk_level = classes[best_idx]
        confidence = float(proba[best_idx] * 100)

        # Jednostavno objasnjenje na osnovu klase
        if risk_level == "high":
            explanation = (
                "Model je prepoznao reci i obrasce tipicne za ucenjivanje, pretnje, "
                "tajnost ili trazenje licnih podataka."
            )
        elif risk_level == "medium":
            explanation = (
                "Model vidi pojmove koji lice na emocionalni pritisak ili manipulaciju, "
                "ali bez jasnih pretnji."
            )
        else:
            explanation = (
                "Model nije prepoznao tipicne obrasce manipulacije, pretnji ili ucenjivanja."
            )

        return {
            "risk_level": risk_level,
            "confidence": round(confidence, 2),
            "ml_proba": {
                "classes": classes,
                "proba": [float(x) for x in proba],
            },
            "explanation_for_user": explanation,
        }


# Singleton instanca koju koristi API
guardian_analyzer = GuardianTextAnalyzer()

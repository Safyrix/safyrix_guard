# app/guardian_model.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import joblib
import numpy as np

from .policy_engine import detect_pii, decide_policy


# -------------------------------------------------------------------------
# Putevi do modela
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Novi pipeline model (TF-IDF + klasifikator u jednom)
PIPELINE_PATH = MODELS_DIR / "guardian_text_pipeline.pkl"

# Stari: odvojeni model + vectorizer
MODEL_PATH = MODELS_DIR / "guardian_text_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "guardian_vectorizer.pkl"


class GuardianAgent:
    """
    Glavni wrapper oko:
    - ML modela (tekst -> low/medium/high)
    - PII detekcije (regex)
    - policy engine-a (Guardian score + pravila)

    FastAPI poziva samo metodu `analyze()`.

    Podrzava dva rezima rada:
    1) Novi nacin: guardian_text_pipeline.pkl (pipeline sa TF-IDF + modelom)
    2) Stari nacin: guardian_text_model.pkl + guardian_vectorizer.pkl
    """

    def __init__(self) -> None:
        # --- Pipeline model (novi) ---
        self.pipeline: Optional[Any] = None
        if PIPELINE_PATH.exists():
            self.pipeline = joblib.load(PIPELINE_PATH)

        # --- Legacy model + vectorizer (stari) ---
        self.model: Optional[Any] = None
        self.vectorizer: Optional[Any] = None

        if MODEL_PATH.exists():
            self.model = joblib.load(MODEL_PATH)

        if VECTORIZER_PATH.exists():
            self.vectorizer = joblib.load(VECTORIZER_PATH)

    # ---------------------------------------------------------------------
    #                          HELPER: da li imamo ML?
    # ---------------------------------------------------------------------
    @property
    def has_ml(self) -> bool:
        """
        Vraca True ako imamo bar jedan validan ML setup:
        - pipeline (preporuceno)
        - ili legacy model + vectorizer
        """
        if self.pipeline is not None:
            return True
        return self.model is not None and self.vectorizer is not None

    # ---------------------------------------------------------------------
    #                           ML PREDIKCIJA
    # ---------------------------------------------------------------------
    def _predict_ml(self, message: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Vrsi klasifikaciju rizika pomocu ML modela.

        Vraca:
        - ml_label:  "low" / "medium" / "high"
        - ml_conf:   verovatnoca za izabranu klasu (0–1)
        - ml_details: dict sa klasama i verovatnocama (za debug / frontend)

        Ako model ne postoji, vraca fallback: ("low", 0.0, {...})
        """
        # Fallback rezim bez ML-a
        if not self.has_ml:
            return "low", 0.0, {
                "classes": ["low", "medium", "high"],
                "proba": [0.0, 0.0, 0.0],
                "risk_level": "low",
            }

        try:
            # ----------------- 1) Novi rezim: pipeline -----------------
            if self.pipeline is not None:
                X = [message]
                probs = self.pipeline.predict_proba(X)[0]  # shape (n_classes,)
                classes = self.pipeline.classes_

            # ----------------- 2) Legacy rezim: vectorizer + model -----
            else:
                X = self.vectorizer.transform([message])
                probs = self.model.predict_proba(X)[0]
                classes = self.model.classes_

            best_idx = int(np.argmax(probs))
            ml_label = str(classes[best_idx]).lower()
            ml_conf = float(probs[best_idx])

            ml_details: Dict[str, Any] = {
                "classes": [str(c) for c in classes],
                "proba": probs.tolist(),
                "risk_level": ml_label,
            }
            return ml_label, ml_conf, ml_details

        except Exception:
            # U produkciji bi ovde dodali logging.warn(...)
            return "low", 0.0, {
                "classes": ["low", "medium", "high"],
                "proba": [0.0, 0.0, 0.0],
                "risk_level": "low",
            }

    # ---------------------------------------------------------------------
    #                           GLAVNA FUNKCIJA
    # ---------------------------------------------------------------------
    def analyze(self, message: str) -> Dict[str, Any]:
        """
        Jedina funkcija koju poziva FastAPI.

        Koraci:
        1) PII detekcija (regex / heuristika)
        2) ML klasifikacija rizika
        3) Policy engine (Guardian score + pravila)
        4) Formatiranje rezultata
        """

        # ------------------- 1. PII DETEKCIJA -------------------
        pii_data = detect_pii(message)
        pii_flags: List[str] = pii_data.get("flags", [])
        pii_matches: List[str] = pii_data.get("matches", [])

        # ------------------- 2. ML KLASIFIKACIJA ----------------
        ml_label, ml_confidence, ml_details = self._predict_ml(message)

        # ------------------- 3. POLICY ENGINE -------------------
        final_risk, policy_action, policy_flags = decide_policy(
            message=message,
            pii_flags=pii_flags,
            ml_label=ml_label,
            ml_confidence=ml_confidence,
        )

        # ------------------- 4. FORMATIRANJE ODGOVORA ----------
        result: Dict[str, Any] = {
            "risk_level": final_risk,
            "categories": policy_flags.get("categories", []),
            "confidence": ml_confidence,
            "red_flags": policy_flags.get("red_flags", []) + pii_flags,
            "explanation_for_user": policy_flags.get("explanation_for_user", ""),
            "recommended_actions": policy_flags.get("recommended_actions", []),
            "ml_proba": ml_details,
            "pii_matches": pii_matches,
            "policy_action": policy_action,
            "guardian_score": policy_flags.get("guardian_score"),
            "ml_meta": policy_flags.get("ml_meta", {}),
        }

        return result

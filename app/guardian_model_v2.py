# app/guardian_model_v2.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import joblib
import numpy as np

from .policy_engine import decide_policy, detect_pii

# Definicija putanja
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

PIPELINE_V2_PATH = MODELS_DIR / "guardian_text_pipeline_v2.pkl"
PIPELINE_V1_PATH = MODELS_DIR / "guardian_text_pipeline.pkl"


class GuardianAgentV2:
    """
    V2 agent: koristi guardian_text_pipeline_v2.pkl ako postoji,
    u suprotnom se vraca na v1 pipeline.

    Ovo NE dira postojeci GuardianAgent (v1).
    """

    def __init__(self) -> None:
        self.pipeline: Optional[Any] = None
        self.version: str = "none"

        if PIPELINE_V2_PATH.exists():
            try:
                self.pipeline = joblib.load(PIPELINE_V2_PATH)
                self.version = "v2"
            except Exception:
                # Nastavlja dalje da proveri v1 ako v2 ne uspe
                pass

        if self.pipeline is None and PIPELINE_V1_PATH.exists():
            try:
                self.pipeline = joblib.load(PIPELINE_V1_PATH)
                self.version = "v1"
            except Exception:
                # Ostaje 'none'
                pass

    @property
    def has_ml(self) -> bool:
        return self.pipeline is not None

    def _predict_ml(self, message: str) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predviđa nivo rizika koristeći učitani ML model.
        Vraća podrazumevane 'low' vrednosti ako model nije dostupan.
        """
        # Podrazumevani/Sigurnosni odgovor
        default_result: Dict[str, Any] = {
            "classes": ["low", "medium", "high"],
            "proba": [0.0, 0.0, 0.0],
            "risk_level": "low",
            "version": self.version,
        }

        if not self.has_ml:
            return "low", 0.0, default_result

        try:
            X = [message]
            probs = self.pipeline.predict_proba(X)[0]
            classes = self.pipeline.classes_

            best_idx = int(np.argmax(probs))
            ml_label = str(classes[best_idx]).lower()
            ml_conf = float(probs[best_idx])

            ml_details: Dict[str, Any] = {
                "classes": [str(c) for c in classes],
                "proba": probs.tolist(),
                "risk_level": ml_label,
                "version": self.version,
            }
            return ml_label, ml_conf, ml_details
        except Exception:
            # Vraća podrazumevanu vrednost u slučaju greške pri predikciji
            return "low", 0.0, default_result

    # Definicija putanje za logovanje
    ACTIVE_LEARNING_LOG = Path(__file__).resolve().parent / "data" / "active_learning_candidates.log"

        # Definicija putanje za logovanje (class-level konstanta)
    ACTIVE_LEARNING_LOG = Path(__file__).resolve().parent / "data" / "active_learning_candidates.log"

    def log_for_active_learning(self, message: str, ml_confidence: float) -> None:
        """
        Snima poruke koje su interesantne za dalje treniranje:
        - nizak confidence
        - potencijalno rizicne kljucne reci
        """
        # Kreiraj folder ako ne postoji
        self.ACTIVE_LEARNING_LOG.parent.mkdir(exist_ok=True, parents=True)

        # Zamena novih linija ('\n' i '\r') razmakom
        cleaned_message = message.replace("\n", " ").replace("\r", " ")
        line = f"{ml_confidence:.3f}\t{cleaned_message}\n"

        with self.ACTIVE_LEARNING_LOG.open("a", encoding="utf-8") as f:
            f.write(line)


    def analyze(self, message: str) -> Dict[str, Any]:
        """
        Glavna metoda za analizu poruke kombinujući PII, ML i politiku.
        """
        pii_data = detect_pii(message)
        pii_flags: List[str] = pii_data.get("flags", [])
        pii_matches: List[str] = pii_data.get("matches", [])

        ml_label, ml_conf, ml_details = self._predict_ml(message)

        # Active Learning hook:
        msg_lower = message.lower()
        suspicious_keywords = ["jmbg", "sifra", "password", "lozinka", "prijava", "logovanje", "pristup", "kod"]

        if ml_conf < 0.7 or any(k in msg_lower for k in suspicious_keywords):
            # Poziv funkcije log_for_active_learning (pretpostavljajući da je dostupna/ispravno definisana)
            self.log_for_active_learning(message, ml_conf) # Ispravljen poziv ako je statička metoda

        final_risk, policy_action, policy_flags = decide_policy(
            message=message,
            pii_flags=pii_flags,
            ml_label=ml_label,
            ml_confidence=ml_conf,
        )

        return {
            "risk_level": final_risk,
            "categories": policy_flags.get("categories", []),
            "confidence": ml_conf,
            "red_flags": policy_flags.get("red_flags", []) + pii_flags,
            "explanation_for_user": policy_flags.get(
                "explanation_for_user", ""
            ),
            "recommended_actions": policy_flags.get(
                "recommended_actions", []
            ),
            "ml_proba": ml_details,
            "pii_matches": pii_matches,
            "policy_action": policy_action,
            "guardian_score": policy_flags.get("guardian_score"),
            "ml_meta": policy_flags.get("ml_meta", {}),
        }
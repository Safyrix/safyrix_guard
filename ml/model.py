import pickle
import re
from pathlib import Path
from typing import Dict, List, Any


MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "guardian_text_model.pkl"


class GuardianModel:
    """
    Jednostavan tekstualni risk engine.

    - Ako postoji models/guardian_text_model.pkl, ucitava paterne odatle
    - Ako ne postoji, koristi podrazumevani skup pravila (hard-coded)
    """

    def __init__(self) -> None:
        self.patterns: Dict[str, List[str]] = self._load_patterns()

    def _load_patterns(self) -> Dict[str, List[str]]:
        """Pokusaj da ucitas paterne iz pickle fajla, inace koristi default."""

        if MODEL_FILE.exists():
            with open(MODEL_FILE, "rb") as f:
                data = pickle.load(f)

            # Ocekivana struktura: {"manipulation": [...], "threat": [...], ...}
            if isinstance(data, dict):
                return data

        # Fallback default paterni – ovo deluje kao "prvi draft" pravila.
        return {
            "manipulation": [
                r"ako me stvarno volis",
                r"nemoj da kazes roditeljima",
                r"posalji mi svoju lokaciju",
                r"zabranjujem ti",
                r"ako me volis dokazaces",
            ],
            "threat": [
                r"ako ne uradis",
                r"videces ti",
                r"naci cu te",
            ],
            "grooming": [
                r"nemoj da ukljucis odrasle",
                r"ovo je nasa tajna",
                r"nemoj nikome da kazes",
            ],
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analiziraj tekst i vrati strukturisan odgovor za API sloj.
        """

        text_low = text.lower()
        matched_categories: List[str] = []
        red_flags: List[str] = []

        for category, rules in self.patterns.items():
            for pattern in rules:
                if re.search(pattern, text_low):
                    if category not in matched_categories:
                        matched_categories.append(category)
                    red_flags.append(pattern)

        if not matched_categories:
            return {
                "risk_level": "low",
                "categories": [],
                "confidence": 30,
                "red_flags": [],
                "explanation_for_user": (
                    "U ovoj poruci nisu prepoznate tipicne fraze manipulacije, "
                    "pretnje ili neprimerenog ponasanja."
                ),
                "recommended_actions": [
                    "Ako se osecas prijatno u komunikaciji i nema cudnih zahteva, rizik je nizak."
                ],
            }

        # Sto vise kategorija, to jaci rizik
        if len(matched_categories) >= 2:
            risk_level = "high"
            confidence = 80
        else:
            risk_level = "medium"
            confidence = 60

        explanation = (
            "Guardian je prepoznao sumnjive fraze koje lice na manipulaciju, "
            "ucenu ili neprimereno vezivanje. "
            "Proveri da li neko od tebe trazi nesto sto ti nije prijatno."
        )

        recommended_actions = [
            "Postavi jasne granice u komunikaciji.",
            "Ako te neko tera da uradis nesto sto ne zelis – imas pravo da kazes NE.",
            "Ako se osecas ugrozeno ili zbunjeno, obrati se odrasloj osobi kojoj verujes.",
        ]

        return {
            "risk_level": risk_level,
            "categories": matched_categories,
            "confidence": confidence,
            "red_flags": red_flags,
            "explanation_for_user": explanation,
            "recommended_actions": recommended_actions,
        }

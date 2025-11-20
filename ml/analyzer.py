from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List

from ml.patterns import detect_categories, heuristic_risk_level


@dataclass
class AnalysisResult:
    risk_level: str
    categories: List[str]
    confidence: int
    red_flags: List[str]
    explanation_for_user: str
    recommended_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GuardianTextAnalyzer:
    """
    Jednostavan rule-based analizator.
    Kasnije mozes da ubacis i ML pa da ovde kombinujes.
    """

    def analyze(self, message: str, user_id: str) -> Dict[str, Any]:
        categories, red_flags = detect_categories(message)
        risk_level = heuristic_risk_level(categories)

        # confidence na grubo, samo da imas nesto
        base_confidence = {
            "low": 30,
            "medium": 60,
            "high": 80,
        }[risk_level]

        explanation = self._build_explanation(risk_level, categories)
        actions = self._build_actions(risk_level)

        result = AnalysisResult(
            risk_level=risk_level,
            categories=categories,
            confidence=base_confidence,
            red_flags=red_flags,
            explanation_for_user=explanation,
            recommended_actions=actions,
        )
        return result.to_dict()

    @staticmethod
    def _build_explanation(risk_level: str, categories: List[str]) -> str:
        if risk_level == "low":
            return (
                "Guardian u ovoj poruci nije prepoznao tipicne fraze manipulacije, "
                "pretnje ili neprimerenog ponasanja. Ipak, uvek slusaj svoj osecaj - "
                "ako se ne osecas prijatno, imas pravo da prekinas komunikaciju."
            )

        if risk_level == "medium":
            return (
                "Guardian je prepoznao potencijalno rizicne signale u poruci "
                f"(kategorije: {', '.join(categories)}). Preporucuje se oprez."
            )

        # high
        return (
            "Guardian je prepoznao opasne fraze koje lice na ucenjivanje, skrivanje "
            "od roditelja, trazenje lokacije ili druge oblike manipulacije. "
            "Ovu komunikaciju treba shvatiti ozbiljno."
        )

    @staticmethod
    def _build_actions(risk_level: str) -> List[str]:
        if risk_level == "low":
            return [
                "Ako se osecas prijatno u komunikaciji i nema cudnih zahteva, rizik je nizak."
            ]

        if risk_level == "medium":
            return [
                "Ne donosi odluke na brzinu.",
                "Ako si maloletan, popricaj sa roditeljem ili odraslom osobom od poverenja.",
            ]

        # high
        return [
            "Ne salji licne podatke, adresu, slike ili lokaciju.",
            "Napusti razgovor ako se osecas ugrozeno.",
            "Ako si maloletan, odmah obavesti roditelje ili staratelja.",
        ]

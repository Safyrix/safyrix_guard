# app/policy_engine.py

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------
# Konstante
# ---------------------------------------------------------

# Ako je sigurnost ML modela ispod ovoga → preporuka za REVIEW
LOW_CONF_THRESHOLD: float = 0.60  # 60%

# Kriticni PII tipovi (i broj i namera za njih su HIGH risk)
CRITICAL_PII_FLAGS = {"jmbg", "credit_card", "iban", "passport", "address"}


# ---------------------------------------------------------
# 1. PII DETEKCIJA (JMBG, kartica, mejl, telefon, ...)
# ---------------------------------------------------------

PII_PATTERNS: Dict[str, re.Pattern] = {
    "jmbg": re.compile(r"\b\d{13}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "phone": re.compile(r"\b\+?\d{9,14}\b"),
    # po potrebi prosirujemo:
    # "address": re.compile(r"..."),
    # "iban": re.compile(r"..."),
    # "passport": re.compile(r"..."),
}


def map_confidence_to_risk(ml_conf: float) -> str:
    """
    Mapiranje samo na osnovu confidence-a.
    Koristicemo ga za debug / buducu logiku, ali final_risk dolazi iz jednog mesta.
    """
    if ml_conf >= 0.80:
        return "high"
    elif ml_conf >= 0.50:
        return "medium"
    else:
        return "low"


def detect_pii(message: str) -> Dict[str, List[str]]:
    """
    Detekcija prostih PII paterna u tekstu.

    Vraca dict kompatibilan sa AnalyzeResponse:

    {
        "flags":   ["jmbg", "email"],
        "matches": ["1234567890123", "neko@example.com"]
    }
    """
    flags: List[str] = []
    matches: List[str] = []

    for pii_type, pattern in PII_PATTERNS.items():
        for m in pattern.finditer(message):
            flags.append(pii_type)
            matches.append(m.group(0))

    flags = sorted(set(flags))
    return {"flags": flags, "matches": matches}


# ---------------------------------------------------------
# 1b. Detekcija NAMERE za PII (implicitni rizik)
# ---------------------------------------------------------

def detect_pii_intent(message: str) -> List[str]:
    """
    Detektuje NAMERU da se traze osetljivi podaci (bez brojeva):
    - "daj mi jmbg"
    - "posalji mi lozinku"
    - "koji je tvoj broj kartice"
    - "daj mi verifikacioni kod" itd.

    Vraca listu "intent" flagova, npr:
    ["intent_jmbg", "intent_password"]
    """
    msg = message.lower()
    intent_flags: List[str] = []

    # JMBG namera
    if re.search(r"(daj mi|posalji mi|treba mi|koji je|unesi)\s+.*jmbg", msg):
        intent_flags.append("intent_jmbg")

    # password / sifra / lozinka
    if re.search(r"(daj mi|posalji mi|treba mi|unesi)\s+.*(sifra|lozinka|password|pin)", msg):
        intent_flags.append("intent_password")

    # broj kartice
    if re.search(r"(daj mi|posalji mi|treba mi|koji je)\s+.*(kartic|card)", msg):
        intent_flags.append("intent_credit_card")

    # verifikacioni kod / kod sa telefona / token
    if re.search(r"(daj mi|posalji mi|treba mi)\s+.*(kod|code|token)", msg):
        intent_flags.append("intent_token")

    return intent_flags


# ---------------------------------------------------------
# 2. Small-talk heuristika (rasterecuje ML)
# ---------------------------------------------------------

SMALLTALK_PATTERNS: List[str] = [
    "kako si",
    "sta radis",
    "kako se zoves",
    "lepo je vreme",
    "sta ima",
    "vidimo se",
    "sta planiras danas",
]


def is_smalltalk(message: str) -> bool:
    """
    Vraca True ako poruka izgleda kao obican small-talk:
    - kratka je
    - nema brojeve
    - sadrzi neku od tipicnih small-talk fraza
    """
    msg = message.lower()

    if any(ch.isdigit() for ch in msg):
        return False
    if len(msg) > 80:
        return False

    return any(pat in msg for pat in SMALLTALK_PATTERNS)


# ---------------------------------------------------------
# 3. Guardian scoring 0–100 (ML + PII)
# ---------------------------------------------------------

def compute_guardian_score(
    ml_label: str,
    ml_confidence: float,
    pii_flags: List[str],
) -> int:
    """
    Racuna jedan broj 0–100 koji kombinuje:
    - ML label (low / medium / high)
    - sigurnost modela (confidence)
    - prisustvo PII flagova
    """
    score = 0

    label = ml_label.lower()
    if label == "high":
        score += 60
    elif label == "medium":
        score += 35
    else:
        score += 10

    score += int(max(0.0, min(ml_confidence, 1.0)) * 20)

    if pii_flags:
        score += 15

        critical_flags = {"jmbg", "credit_card", "iban", "passport", "address"}
        if any(f in critical_flags for f in pii_flags):
            score += 15

    score = max(0, min(score, 100))
    return score


def map_score_to_risk(score: int) -> str:
    """
    JEDINO mesto gde se score mapira na nivo rizika.
    Ovo drzi sve pragove na jednom mestu.
    """
    if score >= 75:
        return "high"
    elif score >= 45:
        return "medium"
    else:
        return "low"


# ---------------------------------------------------------
# 4. Glavni policy sloj
# ---------------------------------------------------------

def decide_policy(
    message: str,
    pii_flags: List[str],
    ml_label: str,
    ml_confidence: float,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Glavna odluka policy layer-a.

    Ulaz:
    - message: originalna poruka
    - pii_flags: lista PII tipova detektovanih u poruci
    - ml_label: "low" / "medium" / "high" iz ML modela
    - ml_confidence: sigurnost ML modela (0–1)

    Izlaz:
    - final_risk: "low" / "medium" / "high"
    - policy_action: "ALLOW" / "ALLOW_BUT_LOG" / "BLOCK" / "REVIEW"
    - policy_flags: dodatne informacije za frontend / log
    """

    red_flags: List[str] = []
    categories: List[str] = []
    recommended_actions: List[str] = []

    # 1) Guardian score (ML + PII)
    guardian_score = compute_guardian_score(
        ml_label=ml_label,
        ml_confidence=ml_confidence,
        pii_flags=pii_flags,
    )

    # 1b) Detekcija NAMERE za PII (implicitni rizik)
    pii_intent_flags = detect_pii_intent(message)

    # 1c) Kriticni PII (broj) ili namera -> pojacavamo score na HIGH zonu
    if any(f in CRITICAL_PII_FLAGS for f in pii_flags) or pii_intent_flags:
        # ako je score nizak, digni ga da reflektuje HIGH rizik
        if guardian_score < 90:
            guardian_score = 90

    # 2) Bazni rizik samo iz score-a – JEDNO mesto
    final_risk = map_score_to_risk(guardian_score)

    # 3) Bazna akcija iz final_riska
    if final_risk == "high":
        policy_action = "BLOCK"
    elif final_risk == "medium":
        policy_action = "ALLOW_BUT_LOG"
    else:
        policy_action = "ALLOW"

    # 4) PII utice na kategorije i red_flags (ali ne razbija konzistentnost)
    if pii_flags:
        categories.append("pii")
        red_flags.append("pii_detected")

    # 4b) PII namera (bez brojeva) – oznacavamo posebno
    if pii_intent_flags:
        categories.append("pii_intent")
        red_flags.append("pii_intent_detected")

    # 5) Small-talk override – ako nema PII i poruka je jako benignа,
    # forsiramo LOW + ALLOW (i ovo je i dalje centralizovano).
    if not pii_flags and not pii_intent_flags and is_smalltalk(message):
        final_risk = "low"
        policy_action = "ALLOW"
        red_flags.append("smalltalk_override")

    # 6) Niska sigurnost -> ne menjamo final_risk,
    # ali policy_action ide u REVIEW i dodajemo flag.
    low_confidence_note = ""
    if ml_confidence < LOW_CONF_THRESHOLD:
        policy_action = "REVIEW"
        red_flags.append("low_confidence_ml")
        low_confidence_note = (
            f"Model nije dovoljno siguran (confidence={ml_confidence:.2f}). "
            "Preporucena rucna provera."
        )

    # 7) Tekst za korisnika – koristi iskljucivo FINAL_RISK kao istinu
    explanation_parts: List[str] = [
        f"Finalni rizik je klasifikovan kao **{final_risk.upper()}**.",
        (
            "Interno, ML model predlaze klasu "
            f"**{ml_label.upper()}** sa verovatnocom {ml_confidence:.2f} (0–1 skala), "
            f"a Guardian score (kombinacija ML, PII i heuristika) iznosi {guardian_score}/100."
        ),
    ]

    if pii_flags:
        explanation_parts.append(
            f"Detektovani su osetljivi podaci (PII): {', '.join(pii_flags)}."
        )

    if pii_intent_flags:
        explanation_parts.append(
            "Detektovana je namera da se traze osetljivi podaci "
            f"({', '.join(pii_intent_flags)}), sto povecava rizik."
        )

    if "smalltalk_override" in red_flags:
        explanation_parts.append(
            "Poruka je prepoznata kao obican razgovor (small-talk), "
            "bez licnih podataka i bez znakova prevare."
        )

    if low_confidence_note:
        explanation_parts.append(low_confidence_note)

    explanation_for_user = " ".join(explanation_parts).strip()

    # 8) Preporucene akcije – opet na osnovu FINAL_RISK
    if final_risk == "high":
        recommended_actions = [
            "Ne salji ovu poruku direktno u javni AI alat.",
            "Ukloni licne i poslovno osetljive podatke pre slanja.",
            "Kontaktiraj IT / bezbednosni tim ako nisi siguran.",
        ]
    elif final_risk == "medium":
        recommended_actions = [
            "Proveri da li poruka sadrzi licne ili poslovno osetljive podatke.",
            "Po mogucnosti, anonimuzuj informacije pre slanja.",
        ]
    else:  # low
        recommended_actions = [
            "Poruka izgleda niskog rizika, ali i dalje postuj interne politike.",
        ]

    policy_flags: Dict[str, Any] = {
        "categories": categories,
        "red_flags": red_flags,
        "explanation_for_user": explanation_for_user,
        "recommended_actions": recommended_actions,
        "guardian_score": guardian_score,
        "ml_meta": {
            "ml_label": ml_label,
            "ml_confidence": ml_confidence,
            # za debug/analitiku mozes da gledas i ovo:
            "pii_intent_flags": pii_intent_flags,
        },
    }

    return final_risk, policy_action, policy_flags

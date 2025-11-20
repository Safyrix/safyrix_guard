from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Pattern, Tuple
import re


# ---- 1. Osnovne kategorije i fraze ----

BASE_PATTERNS: Dict[str, List[str]] = {
    "secrecy": [
        "nemoj da kazes roditeljima",
        "nemoj roditeljima da kazes",
        "nemoj da ukljucis odrasle",
        "ovo je nasa tajna",
        "nemoj nikome da kazes",
        "nemoj nikom reci",
        "nemoj nikom da pricas",
    ],
    "location_request": [
        "posalji mi svoju adresu",
        "daj mi svoju adresu",
        "posalji mi adresu",
        "posalji mi svoju lokaciju",
        "posalji mi lokaciju",
        "share location",
    ],
    "meeting_request": [
        "hajde da se nadjemo",
        "dodji kod mene",
        "reci mi kada su ti roditelji odsutni",
        "reci mi kada si sam kuci",
        "mozemo da se vidimo sami",
    ],
    "money_pressure": [
        "posudi mi novac",
        "daj mi pare",
        "da li imas karticu",
        "posalji mi sa kartice",
        "posalji mi novac",
        "daj mi podatke sa kartice",
    ],
    "threat": [
        "ako ne uradis ovo",
        "videces ti",
        "nemoj da se igras sa mnom",
        "imam tvoje slike",
        "objavicu svima",
        "zapamti sa kim pricas",
    ],
    "flattery_trust": [
        "samo ti mene razumes",
        "ne moras nikom drugom da pricas",
        "ti si poseban",
        "ti si posebna",
        "mozes da verujes samo meni",
        "nikom drugom ne veruj",
    ],
    "sexual_content": [
        "posalji mi sliku u vesu",
        "posalji mi sliku bez majice",
        "nemoj nikom da pokazes nase slike",
        "da li spavas sam",
        "da li spavas sama",
        "sta imas obuceno",
    ],
    "age_role_gap": [
        "ja sam dosta stariji",
        "ja sam mnogo stariji",
        "znam bolje od tvojih roditelja",
        "ne moras da slusas roditelje",
        "oni ne razumeju",
    ],
}


# ---- 2. Konfiguracija regexa ----

@dataclass(frozen=True)
class PatternConfig:
    max_gap: int = 40  # koliko teksta smemo između reči

    def phrase_to_regex(self, phrase: str) -> str:
        cleaned = phrase.lower()
        parts = [re.escape(p) for p in cleaned.split() if p]
        if not parts:
            return ""
        gap = rf"[\W_]{{0,{self.max_gap}}}"
        return gap.join(parts)


CONFIG = PatternConfig()


def compile_patterns(base: Dict[str, List[str]] = BASE_PATTERNS) -> Dict[str, List[Pattern[str]]]:
    compiled: Dict[str, List[Pattern[str]]] = {}

    for category, phrases in base.items():
        bucket: List[Pattern[str]] = []
        for phrase in phrases:
            body = CONFIG.phrase_to_regex(phrase)
            if not body:
                continue
            pat = re.compile(body, flags=re.IGNORECASE)
            bucket.append(pat)
        compiled[category] = bucket

    return compiled


COMPILED_PATTERNS: Dict[str, List[Pattern[str]]] = compile_patterns()


# ---- 3. Helper funkcije ----

def detect_categories(message: str) -> Tuple[List[str], List[str]]:
    hits: List[str] = []
    red_flags: List[str] = []

    text = message.lower()

    for category, patterns in COMPILED_PATTERNS.items():
        for pat in patterns:
            m = pat.search(text)
            if m:
                hits.append(category)
                snippet = text[max(0, m.start() - 10): m.end() + 10]
                red_flags.append(f"{category}: ...{snippet}...")
                break

    hits = sorted(set(hits))
    return hits, red_flags


def heuristic_risk_level(categories: List[str]) -> str:
    if not categories:
        return "low"

    high_risk = {"secrecy", "location_request", "meeting_request", "threat", "sexual_content"}
    medium_risk = {"money_pressure", "flattery_trust", "age_role_gap"}

    if any(c in high_risk for c in categories):
        return "high"
    if any(c in medium_risk for c in categories):
        return "medium"
    return "low"

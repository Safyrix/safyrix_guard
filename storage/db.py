import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


class FeedbackStorage:
    """
    Prosta JSON 'baza' za cuvanje istorije analiza.
    Nije za produkciju, ali je super za demo i debug.
    """

    FILE = Path("storage") / "data.json"

    def __init__(self) -> None:
        self.FILE.parent.mkdir(parents=True, exist_ok=True)
        self.db: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if not self.FILE.exists():
            return []
        try:
            with open(self.FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def add_entry(self, user_id: str, result: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "risk_level": result,
        }
        self.db.append(entry)
        self._save()

    def _save(self) -> None:
        with open(self.FILE, "w", encoding="utf-8") as f:
            json.dump(self.db, f, indent=2, ensure_ascii=False)

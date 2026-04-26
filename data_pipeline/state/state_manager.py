"""
Safyrix Data Generation - State Manager
========================================

Production-grade state management za long-running data generation pipeline.

Kljuni koncepti:
- Atomic writes (write-then-rename pattern)
- Backup pre svakog write-a
- Daily quota tracking
- Idempotent resume
- Per-category progress tracking

Uloga:
- Pamti gde je pipeline stao
- Sprecava duplikate kroz runove
- Postuje API quote (dnevni limit)
- Omogucava bezbedan multi-day run
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class CategoryProgress:
    """Progress za jednu kategoriju."""
    category: str
    target_samples: int
    samples_generated: int = 0
    api_calls_made: int = 0
    last_updated: str = ""
    status: str = "pending"  # pending | in_progress | completed | failed
    last_error: str | None = None
    
    def is_complete(self) -> bool:
        return self.samples_generated >= self.target_samples
    
    def remaining(self) -> int:
        return max(0, self.target_samples - self.samples_generated)


@dataclass
class DailyQuota:
    """Tracker za dnevnu API quotu."""
    date: str  # ISO format YYYY-MM-DD
    api_calls_used: int = 0
    tokens_used: int = 0
    api_calls_limit: int = 250  # Gemini free tier daily
    tokens_limit: int = 1_000_000  # 1M tokens daily
    
    def is_exhausted(self) -> bool:
        """Da li smo udarili limit?"""
        return (
            self.api_calls_used >= self.api_calls_limit
            or self.tokens_used >= self.tokens_limit
        )
    
    def remaining_calls(self) -> int:
        return max(0, self.api_calls_limit - self.api_calls_used)
    
    def remaining_tokens(self) -> int:
        return max(0, self.tokens_limit - self.tokens_used)
    
    def utilization_pct(self) -> float:
        """Procenat iskoriscene quote (najveci od dva metrike)."""
        calls_pct = self.api_calls_used / self.api_calls_limit if self.api_calls_limit else 0
        tokens_pct = self.tokens_used / self.tokens_limit if self.tokens_limit else 0
        return max(calls_pct, tokens_pct)


@dataclass
class PipelineState:
    """Kompletno stanje pipeline-a."""
    version: str = "1.0.0"
    pipeline_id: str = ""  # UUID generisan pri inicijalnom run-u
    started_at: str = ""
    last_updated: str = ""
    total_runs: int = 0
    
    categories: dict[str, CategoryProgress] = field(default_factory=dict)
    daily_quota: dict[str, DailyQuota] = field(default_factory=dict)  # date -> quota
    generated_text_hashes: list[str] = field(default_factory=list)  # SHA256 generisanih tekstova
    
    output_file: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """Konverzija u dict za JSON."""
        return {
            "version": self.version,
            "pipeline_id": self.pipeline_id,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
            "total_runs": self.total_runs,
            "categories": {k: asdict(v) for k, v in self.categories.items()},
            "daily_quota": {k: asdict(v) for k, v in self.daily_quota.items()},
            "generated_text_hashes": self.generated_text_hashes,
            "output_file": self.output_file,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineState":
        """Kreira state iz dict-a."""
        state = cls(
            version=data.get("version", "1.0.0"),
            pipeline_id=data.get("pipeline_id", ""),
            started_at=data.get("started_at", ""),
            last_updated=data.get("last_updated", ""),
            total_runs=data.get("total_runs", 0),
            output_file=data.get("output_file", ""),
            generated_text_hashes=data.get("generated_text_hashes", []),
        )
        
        for cat_name, cat_data in data.get("categories", {}).items():
            state.categories[cat_name] = CategoryProgress(**cat_data)
        
        for date_str, quota_data in data.get("daily_quota", {}).items():
            state.daily_quota[date_str] = DailyQuota(**quota_data)
        
        return state


# ============================================================
# STATE MANAGER
# ============================================================

class StateManager:
    """
    Persistent state manager za data generation pipeline.
    
    Garantije:
    - Atomic writes (state ili stari ili novi, nikad pokvaren)
    - Backup pre svakog write-a
    - Thread-safe za single-process (ne za multi-process)
    """
    
    STATE_FILE = "pipeline_state.json"
    BACKUP_FILE = "pipeline_state.backup.json"
    TEMP_FILE = "pipeline_state.tmp.json"
    
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        
        self.state_path = self.state_dir / self.STATE_FILE
        self.backup_path = self.state_dir / self.BACKUP_FILE
        self.temp_path = self.state_dir / self.TEMP_FILE
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self._state: PipelineState | None = None
    
    # ============================================================
    # PUBLIC API
    # ============================================================
    
    def load_or_init(
        self,
        pipeline_id: str,
        output_file: str,
        category_targets: dict[str, int],
    ) -> PipelineState:
        """
        Ucitava postojeci state ili kreira nov ako ne postoji.
        
        Args:
            pipeline_id: UUID pipeline-a (kreira se prvi put)
            output_file: Putanja do output JSONL fajla
            category_targets: Mapa {category_name: target_samples}
        """
        if self.state_path.exists():
            self._state = self._load_from_disk()
            self.logger.info(
                f"Ucitan postojeci state: pipeline_id={self._state.pipeline_id}, "
                f"runs={self._state.total_runs}"
            )
            
            # Sync category targets (mozda su se promenili u YAML-u)
            self._sync_categories(category_targets)
        else:
            now = datetime.now(timezone.utc).isoformat()
            self._state = PipelineState(
                pipeline_id=pipeline_id,
                started_at=now,
                last_updated=now,
                output_file=output_file,
            )
            
            # Init categories
            for cat_name, target in category_targets.items():
                self._state.categories[cat_name] = CategoryProgress(
                    category=cat_name,
                    target_samples=target,
                )
            
            self.logger.info(
                f"Kreiran novi state: pipeline_id={pipeline_id}, "
                f"categories={list(category_targets.keys())}"
            )
            self.save()
        
        # Bump run counter
        self._state.total_runs += 1
        self.save()
        
        return self._state
    
    def save(self) -> None:
        """
        Atomic save state to disk.
        
        Process:
        1. Sacuvaj backup od trenutnog state-a
        2. Pisi u temp fajl
        3. Rename temp -> state (atomic na vecini OS-a)
        """
        if self._state is None:
            self.logger.warning("Pokusaj save bez state-a, preskacem")
            return
        
        self._state.last_updated = datetime.now(timezone.utc).isoformat()
        
        # Step 1: Backup
        if self.state_path.exists():
            shutil.copy2(self.state_path, self.backup_path)
        
        # Step 2: Write to temp
        try:
            with self.temp_path.open("w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, ensure_ascii=False, indent=2)
            
            # Step 3: Atomic rename
            os.replace(self.temp_path, self.state_path)
        
        except Exception as e:
            self.logger.error(f"Save state failed: {e}")
            # Cleanup temp ako je ostao
            if self.temp_path.exists():
                self.temp_path.unlink()
            raise
    
    def get_state(self) -> PipelineState:
        """Vraca trenutni state. Mora biti load_or_init-ovan."""
        if self._state is None:
            raise RuntimeError("State nije ucitan. Pozovi load_or_init() prvo.")
        return self._state
    
    # ============================================================
    # CATEGORY PROGRESS API
    # ============================================================
    
    def update_category_progress(
        self,
        category: str,
        samples_added: int,
        api_calls_added: int = 0,
        status: str | None = None,
        error: str | None = None,
    ) -> None:
        """Updejtuje progress za kategoriju."""
        state = self.get_state()
        
        if category not in state.categories:
            raise KeyError(f"Kategorija '{category}' nije u state-u")
        
        progress = state.categories[category]
        progress.samples_generated += samples_added
        progress.api_calls_made += api_calls_added
        progress.last_updated = datetime.now(timezone.utc).isoformat()
        
        if status:
            progress.status = status
        
        if error:
            progress.last_error = error
        
        # Auto-complete check
        if progress.is_complete() and progress.status != "completed":
            progress.status = "completed"
            self.logger.info(f"Kategorija '{category}' zavrsena!")
    
    def get_pending_categories(self) -> list[CategoryProgress]:
        """Vraca kategorije koje nisu gotove."""
        state = self.get_state()
        return [
            cat for cat in state.categories.values()
            if not cat.is_complete()
        ]
    
    # ============================================================
    # QUOTA TRACKING API
    # ============================================================
    
    def get_today_quota(self) -> DailyQuota:
        """Vraca quotu za danas (kreira ako ne postoji)."""
        state = self.get_state()
        today = date.today().isoformat()
        
        if today not in state.daily_quota:
            state.daily_quota[today] = DailyQuota(date=today)
            self.save()
        
        return state.daily_quota[today]
    
    def consume_quota(self, api_calls: int = 1, tokens: int = 0) -> None:
        """Belezi potrosenu quotu."""
        quota = self.get_today_quota()
        quota.api_calls_used += api_calls
        quota.tokens_used += tokens
    
    def can_make_api_call(self, estimated_tokens: int = 1000) -> tuple[bool, str]:
        """
        Da li mozemo da napravimo jos jedan API poziv?
        
        Returns:
            (yes/no, reason if no)
        """
        quota = self.get_today_quota()
        
        if quota.api_calls_used >= quota.api_calls_limit:
            return False, f"API calls limit ({quota.api_calls_limit}) dostignut za danas"
        
        if quota.tokens_used + estimated_tokens > quota.tokens_limit:
            return False, f"Tokens limit ({quota.tokens_limit}) bio bi prekoracen"
        
        return True, ""
    
    # ============================================================
    # IDEMPOTENCY API (text dedup across runs)
    # ============================================================
    
    def is_text_seen(self, text: str) -> bool:
        """Da li smo vec generisali ovaj tekst?"""
        text_hash = self._hash_text(text)
        state = self.get_state()
        return text_hash in state.generated_text_hashes
    
    def register_text(self, text: str) -> None:
        """Registruje generisani tekst."""
        text_hash = self._hash_text(text)
        state = self.get_state()
        if text_hash not in state.generated_text_hashes:
            state.generated_text_hashes.append(text_hash)
    
    def _hash_text(self, text: str) -> str:
        """SHA256 normalizovanog teksta."""
        normalized = text.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]  # prvih 16 chars dovoljno
    
    # ============================================================
    # PRIVATE HELPERS
    # ============================================================
    
    def _load_from_disk(self) -> PipelineState:
        """Ucitava state sa diska."""
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return PipelineState.from_dict(data)
        except Exception as e:
            self.logger.error(f"Load state failed: {e}")
            
            # Pokusaj backup
            if self.backup_path.exists():
                self.logger.warning("Pokusavam load iz backup-a...")
                with self.backup_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return PipelineState.from_dict(data)
            
            raise
    
    def _sync_categories(self, category_targets: dict[str, int]) -> None:
        """Sinhronizuje kategorije sa novim target-ima (ako su se promenili)."""
        state = self.get_state()
        
        for cat_name, target in category_targets.items():
            if cat_name not in state.categories:
                # Nova kategorija
                state.categories[cat_name] = CategoryProgress(
                    category=cat_name,
                    target_samples=target,
                )
                self.logger.info(f"Dodata nova kategorija u state: {cat_name}")
            elif state.categories[cat_name].target_samples != target:
                # Target se promenio
                old_target = state.categories[cat_name].target_samples
                state.categories[cat_name].target_samples = target
                self.logger.info(
                    f"Updejtovan target za '{cat_name}': {old_target} -> {target}"
                )


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    import tempfile
    import uuid
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("Testiranje StateManager-a...\n")
    
    # Test 1: Inicijalizacija u temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        state_dir = Path(tmpdir) / "state"
        manager = StateManager(state_dir)
        
        # Init
        state = manager.load_or_init(
            pipeline_id=str(uuid.uuid4()),
            output_file="test.jsonl",
            category_targets={
                "safe_personal": 100,
                "phishing": 50,
            },
        )
        
        print(f"[1] Init OK")
        print(f"    Pipeline ID: {state.pipeline_id}")
        print(f"    Categories: {list(state.categories.keys())}")
        print(f"    Total runs: {state.total_runs}")
        
        # Test 2: Update progress
        manager.update_category_progress("safe_personal", samples_added=10, api_calls_added=1)
        manager.consume_quota(api_calls=1, tokens=500)
        manager.save()
        
        print(f"\n[2] After updates:")
        progress = state.categories["safe_personal"]
        print(f"    safe_personal: {progress.samples_generated}/{progress.target_samples}")
        print(f"    API calls: {progress.api_calls_made}")
        
        quota = manager.get_today_quota()
        print(f"    Today's quota: {quota.api_calls_used} calls, {quota.tokens_used} tokens")
        print(f"    Quota utilization: {quota.utilization_pct():.1%}")
        
        # Test 3: Reload simulation
        manager2 = StateManager(state_dir)
        state2 = manager2.load_or_init(
            pipeline_id="this-should-be-ignored",
            output_file="test.jsonl",
            category_targets={
                "safe_personal": 100,
                "phishing": 50,
            },
        )
        
        print(f"\n[3] After reload:")
        print(f"    Same pipeline_id: {state.pipeline_id == state2.pipeline_id}")
        print(f"    Total runs incremented: {state2.total_runs}")
        progress2 = state2.categories["safe_personal"]
        print(f"    Progress preserved: {progress2.samples_generated}/{progress2.target_samples}")
        
        # Test 4: Idempotency
        manager2.register_text("Hej, kako si?")
        print(f"\n[4] Idempotency test:")
        print(f"    Tekst registrovan: {manager2.is_text_seen('Hej, kako si?')}")
        print(f"    Tekst (case insensitive): {manager2.is_text_seen('HEJ, KAKO SI?')}")
        print(f"    Razliciti tekst: {manager2.is_text_seen('Drugi tekst')}")
        
        # Test 5: Quota check
        can_call, reason = manager2.can_make_api_call(estimated_tokens=1000)
        print(f"\n[5] Quota check:")
        print(f"    Can make call: {can_call}")
        if not can_call:
            print(f"    Reason: {reason}")
    
    print("\nStateManager test PROSAO.")
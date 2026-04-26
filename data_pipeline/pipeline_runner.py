"""
Safyrix Data Generation - Pipeline Runner v2
==============================================

Production-grade orchestrator sa state management-om.

Razlike u odnosu na run_generation.py (v1):
- Persistent state (resume capability)
- Daily quota tracking (postuje Gemini limite)
- Idempotent (ne pravi duplikate kroz runove)
- Centralni master output fajl (umesto novih po runu)
- Graceful shutdown (Ctrl+C ne pokvari state)

Pokretanje:
    python -m data_pipeline.pipeline_runner                    # nastavi gde si stao
    python -m data_pipeline.pipeline_runner --category NAME    # samo jedna kategorija
    python -m data_pipeline.pipeline_runner --reset            # pocni iznova (PAZI!)
    python -m data_pipeline.pipeline_runner --status           # samo prikazi status
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

from data_pipeline.config.categories_loader import (
    DatasetConfig,
    CategoryConfig,
    load_categories,
)
from data_pipeline.generators.base_generator import GenerationRequest
from data_pipeline.generators.gemini_generator import GeminiGenerator
from data_pipeline.generators.prompt_templates import (
    get_template,
    list_available_categories,
)
from data_pipeline.state.state_manager import StateManager, PipelineState
from data_pipeline.validators.duplicate_checker import DuplicateChecker
from data_pipeline.validators.format_validator import FormatValidator


# ============================================================
# CONSTANTS
# ============================================================

MASTER_OUTPUT_FILENAME = "training_v3_master.jsonl"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SLEEP_BETWEEN_CALLS = 4.0  # 15 RPM = 4s pauza
MAX_CONSECUTIVE_FAILURES = 5
ESTIMATED_TOKENS_PER_BATCH = 1500


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(log_dir: Path, level: int = logging.INFO) -> Path:
    """Postavlja file + console logging. Vraca putanju do log fajla."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    
    # Cleanup postojecih handlera (vazno za multiple runs)
    for h in list(root.handlers):
        root.removeHandler(h)
    
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    
    return log_file


# ============================================================
# GRACEFUL SHUTDOWN
# ============================================================

class ShutdownHandler:
    """Catch Ctrl+C i SIGTERM da pipeline moze da se zaustavi cisto."""
    
    def __init__(self) -> None:
        self.should_stop = False
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
    
    def _handle_signal(self, signum: int, frame: Any) -> None:
        if self.should_stop:
            # Drugi Ctrl+C - force quit
            print("\nForce quit!")
            sys.exit(1)
        
        print("\n\nShutdown signal primljen. Cuvam state i zaustavljam...")
        print("(Pritisni Ctrl+C jos jednom za force quit)")
        self.should_stop = True


# ============================================================
# PIPELINE RUNNER
# ============================================================

class PipelineRunner:
    """
    Production-grade pipeline runner sa state management-om.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        api_key: str,
        output_dir: Path,
        state_dir: Path,
        batch_size: int = DEFAULT_BATCH_SIZE,
        sleep_between_calls: float = DEFAULT_SLEEP_BETWEEN_CALLS,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.sleep_between_calls = sleep_between_calls
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.shutdown = ShutdownHandler()
        
        # Master output
        self.output_file = self.output_dir / MASTER_OUTPUT_FILENAME
        
        # State
        self.state_manager = StateManager(state_dir)
        
        # Komponente
        self.generator: GeminiGenerator | None = None  # lazy init
        self.format_validator = FormatValidator(
            min_text_length=15,
            max_text_length=400,
        )
        self.duplicate_checker = DuplicateChecker(
            near_duplicate_threshold=0.85,
        )
        
        # Run-time stats
        self.run_start = time.time()
        self.run_attempted = 0
        self.run_saved = 0
    
    # ============================================================
    # PUBLIC API
    # ============================================================
    
    def run(self, only_category: str | None = None) -> None:
        """Glavni run."""
        self.logger.info("=" * 70)
        self.logger.info("Safyrix Pipeline Runner v2 - START")
        self.logger.info("=" * 70)
        
        # Init/load state
        eligible_categories = self._get_eligible_categories()
        category_targets = {
            name: cat.target_samples
            for name, cat in eligible_categories.items()
        }
        
        state = self.state_manager.load_or_init(
            pipeline_id=str(uuid.uuid4()),
            output_file=str(self.output_file),
            category_targets=category_targets,
        )
        
        self.logger.info(f"Pipeline ID: {state.pipeline_id}")
        self.logger.info(f"Run #: {state.total_runs}")
        self.logger.info(f"Output: {self.output_file}")
        
        # Init generator (lazy - tek kad imamo state)
        self.generator = GeminiGenerator(api_key=self.api_key)
        
        # Restore duplicate checker state
        self._restore_duplicate_state()
        
        # Print initial status
        self._print_status()
        
        # Filter categories za ovaj run
        if only_category:
            if only_category not in eligible_categories:
                raise ValueError(f"Kategorija '{only_category}' nije eligible")
            run_categories = {only_category: eligible_categories[only_category]}
        else:
            run_categories = eligible_categories
        
        # Glavna petlja
        for cat_name, cat_config in run_categories.items():
            if self.shutdown.should_stop:
                break
            
            progress = state.categories[cat_name]
            
            if progress.is_complete():
                self.logger.info(f"Kategorija '{cat_name}' vec gotova, preskacem")
                continue
            
            # Quota check
            can_call, reason = self.state_manager.can_make_api_call(
                estimated_tokens=ESTIMATED_TOKENS_PER_BATCH
            )
            if not can_call:
                self.logger.warning(f"Zaustavljam - {reason}")
                self.logger.warning("Sutra pokreni pipeline ponovo da nastavi.")
                break
            
            self._generate_category(cat_name, cat_config)
            
            # Save state izmedju kategorija
            self.state_manager.save()
        
        # Final
        self._print_final_report()
    
    def show_status(self) -> None:
        """Prikazuje status bez da menja state."""
        eligible_categories = self._get_eligible_categories()
        category_targets = {
            name: cat.target_samples
            for name, cat in eligible_categories.items()
        }
        
        # Probaj da load-uje state ali ne kreiraj
        if not self.state_manager.state_path.exists():
            print("State ne postoji. Pipeline jos nije pokrenut.")
            return
        
        state = self.state_manager.load_or_init(
            pipeline_id="dummy",  # nece se koristiti jer state postoji
            output_file=str(self.output_file),
            category_targets=category_targets,
        )
        
        # Reverse the run counter bump (jer load_or_init ga uvecava)
        state.total_runs -= 1
        self.state_manager.save()
        
        self._print_status()
    
    def reset(self) -> None:
        """Resetuje state. PAZI - obrisali bi smo sav progres."""
        if self.state_manager.state_path.exists():
            print(f"Brisem state fajl: {self.state_manager.state_path}")
            self.state_manager.state_path.unlink()
        
        if self.state_manager.backup_path.exists():
            print(f"Brisem backup fajl: {self.state_manager.backup_path}")
            self.state_manager.backup_path.unlink()
        
        if self.output_file.exists():
            response = input(
                f"\nMaster output fajl postoji ({self.output_file}). "
                f"Brisem? [y/N]: "
            )
            if response.lower() == "y":
                self.output_file.unlink()
                print("Output fajl obrisan.")
            else:
                print("Output fajl zadrzan (nove generacije ce se appendovati).")
        
        print("\nReset zavrsen. Sledeci run pocinje iznova.")
    
    # ============================================================
    # PRIVATE - GENERATION LOGIC
    # ============================================================
    
    def _generate_category(
        self,
        cat_name: str,
        cat_config: CategoryConfig,
    ) -> None:
        """Generise primere za jednu kategoriju do target-a."""
        state = self.state_manager.get_state()
        progress = state.categories[cat_name]
        
        if progress.status != "in_progress":
            self.state_manager.update_category_progress(cat_name, 0, status="in_progress")
            self.state_manager.save()
        
        template = get_template(cat_name)
        consecutive_failures = 0
        
        self.logger.info(
            f"\n{'─' * 70}\n"
            f"Kategorija: {cat_name}\n"
            f"Progress: {progress.samples_generated}/{progress.target_samples} "
            f"(remaining: {progress.remaining()})\n"
            f"{'─' * 70}"
        )
        
        pbar = tqdm(
            total=progress.target_samples,
            initial=progress.samples_generated,
            desc=f"  {cat_name}",
            unit="primera",
        )
        
        while progress.samples_generated < progress.target_samples:
            # Check shutdown
            if self.shutdown.should_stop:
                self.logger.info("Shutdown trazen, prekidam kategoriju")
                break
            
            # Check consecutive failures
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.logger.error(
                    f"Previse uzastopnih neuspeha ({consecutive_failures}), "
                    f"prekidam '{cat_name}'"
                )
                self.state_manager.update_category_progress(
                    cat_name, 0,
                    status="failed",
                    error=f"{consecutive_failures} consecutive failures",
                )
                break
            
            # Check daily quota
            can_call, reason = self.state_manager.can_make_api_call(
                estimated_tokens=ESTIMATED_TOKENS_PER_BATCH
            )
            if not can_call:
                self.logger.warning(f"Quota iscrpljen tokom rada: {reason}")
                break
            
            # Generate batch
            remaining = progress.remaining()
            batch_size = min(self.batch_size, remaining)
            
            saved_in_batch = self._generate_batch(
                cat_name, template, batch_size
            )
            
            self.run_attempted += batch_size
            
            if saved_in_batch == 0:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
                pbar.update(saved_in_batch)
                self.run_saved += saved_in_batch
            
            # Save state (svaki batch!)
            self.state_manager.save()
            
            # Rate limit
            if not self.shutdown.should_stop:
                time.sleep(self.sleep_between_calls)
        
        pbar.close()
        
        # Final status update
        if progress.is_complete():
            self.state_manager.update_category_progress(
                cat_name, 0, status="completed"
            )
        elif progress.status == "in_progress":
            self.state_manager.update_category_progress(
                cat_name, 0, status="pending"
            )
        
        self.state_manager.save()
    
    def _generate_batch(
        self,
        cat_name: str,
        template: Any,
        batch_size: int,
    ) -> int:
        """Generise jedan batch. Vraca broj sacuvanih primera."""
        try:
            request = GenerationRequest(
                category=cat_name,
                template=template,
                n_samples=batch_size,
            )
            
            result = self.generator.generate(request)
            
            # Quota tracking (priblizno - tacne brojeve znamo iz generatora)
            stats = self.generator.get_stats()
            tokens_this_call = stats.get("total_tokens", 0)
            self.state_manager.consume_quota(
                api_calls=1,
                tokens=ESTIMATED_TOKENS_PER_BATCH,  # konzervativno
            )
            
            if not result.success:
                self.logger.warning(f"Batch greska: {result.error}")
                return 0
            
            # Process samples
            saved = 0
            for sample in result.samples:
                sample_dict = sample.to_dict()
                text = sample_dict["text"]
                
                # Format check
                fmt_result = self.format_validator.validate({"text": text})
                if not fmt_result.is_valid:
                    continue
                
                # Cross-run idempotency check
                if self.state_manager.is_text_seen(text):
                    continue
                
                # Within-run dedup
                dup_result = self.duplicate_checker.check_and_register(text)
                if dup_result.is_duplicate:
                    continue
                
                # All good - save
                self._write_sample(sample_dict)
                self.state_manager.register_text(text)
                saved += 1
            
            self.state_manager.update_category_progress(
                cat_name,
                samples_added=saved,
                api_calls_added=1,
            )
            
            return saved
        
        except Exception as e:
            self.logger.error(f"Batch exception: {e}", exc_info=True)
            return 0
    
    def _write_sample(self, sample_dict: dict[str, Any]) -> None:
        """Append sample to master output JSONL."""
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
    
    def _restore_duplicate_state(self) -> None:
        """Ucitava postojeci output u duplicate checker (cross-run consistency)."""
        if not self.output_file.exists():
            return
        
        loaded = 0
        with self.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    if text:
                        self.duplicate_checker.check_and_register(text)
                        loaded += 1
                except json.JSONDecodeError:
                    continue
        
        if loaded:
            self.logger.info(
                f"Restore: {loaded} postojecih primera ucitano u duplicate checker"
            )
    
    def _get_eligible_categories(self) -> dict[str, CategoryConfig]:
        """Kategorije koje imaju i YAML config i prompt template."""
        available_in_templates = set(list_available_categories())
        return {
            name: cat for name, cat in self.config.categories.items()
            if name in available_in_templates
        }
    
    # ============================================================
    # REPORTING
    # ============================================================
    
    def _print_status(self) -> None:
        """Trenutni status pipeline-a."""
        state = self.state_manager.get_state()
        
        print("\n" + "=" * 70)
        print("PIPELINE STATUS")
        print("=" * 70)
        print(f"Pipeline ID: {state.pipeline_id}")
        print(f"Started: {state.started_at}")
        print(f"Last update: {state.last_updated}")
        print(f"Total runs: {state.total_runs}")
        
        # Today's quota
        quota = self.state_manager.get_today_quota()
        print(f"\nQuota za danas ({quota.date}):")
        print(f"  API calls: {quota.api_calls_used}/{quota.api_calls_limit}")
        print(f"  Tokens: {quota.tokens_used:,}/{quota.tokens_limit:,}")
        print(f"  Utilization: {quota.utilization_pct():.1%}")
        
        # Categories
        print(f"\nKategorije:")
        total_target = 0
        total_done = 0
        for name, cat in state.categories.items():
            pct = (cat.samples_generated / cat.target_samples * 100) if cat.target_samples else 0
            status_icon = {
                "completed": "[OK]",
                "in_progress": "[..]",
                "pending": "[--]",
                "failed": "[XX]",
            }.get(cat.status, "[??]")
            print(
                f"  {status_icon} {name}: "
                f"{cat.samples_generated}/{cat.target_samples} ({pct:.0f}%)"
            )
            total_target += cat.target_samples
            total_done += cat.samples_generated
        
        overall_pct = (total_done / total_target * 100) if total_target else 0
        print(f"\nOverall: {total_done}/{total_target} ({overall_pct:.1f}%)")
        print(f"Unique texts seen (cross-run): {len(state.generated_text_hashes)}")
        print("=" * 70)
    
    def _print_final_report(self) -> None:
        duration = time.time() - self.run_start
        
        print("\n" + "=" * 70)
        print("RUN ZAVRSEN")
        print("=" * 70)
        print(f"Trajanje: {duration:.1f}s ({duration/60:.1f}min)")
        print(f"Pokusano u ovom run-u: {self.run_attempted}")
        print(f"Sacuvano u ovom run-u: {self.run_saved}")
        if self.run_attempted > 0:
            print(f"Stopa uspeha: {self.run_saved/self.run_attempted:.1%}")
        
        self._print_status()
        
        if self.shutdown.should_stop:
            print("\nShutdown clean. Sledeci run nastavlja gde je stao.")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Safyrix Pipeline Runner v2")
    parser.add_argument("--category", type=str, default=None,
                        help="Generisi samo ovu kategoriju")
    parser.add_argument("--status", action="store_true",
                        help="Samo prikazi status, ne pokreci")
    parser.add_argument("--reset", action="store_true",
                        help="Resetuj state (PAZI - brise progres)")
    args = parser.parse_args()
    
    # Putanje
    root_dir = Path(__file__).resolve().parent.parent
    yaml_path = root_dir / "data_pipeline" / "config" / "categories.yaml"
    output_dir = root_dir / "data_pipeline" / "output"
    log_dir = root_dir / "data_pipeline" / "logs"
    state_dir = root_dir / "data_pipeline" / "state"
    
    log_file = setup_logging(log_dir)
    
    load_dotenv(root_dir / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY nije postavljen u .env")
        sys.exit(1)
    
    config = load_categories(yaml_path)
    
    runner = PipelineRunner(
        config=config,
        api_key=api_key,
        output_dir=output_dir,
        state_dir=state_dir,
    )
    
    if args.reset:
        runner.reset()
        return
    
    if args.status:
        runner.show_status()
        return
    
    print(f"Log fajl: {log_file}")
    runner.run(only_category=args.category)


if __name__ == "__main__":
    main()
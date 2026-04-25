"""
Safyrix Data Generation - Orchestrator
=======================================

Glavni entry point za generisanje trening podataka.
Spaja sve module u koherentan pipeline.

Pokrece se:
    python -m data_pipeline.run_generation [--category NAME] [--samples N]

Bez argumenata - generise sve kategorije po YAML konfiguraciji.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
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
    TEMPLATES,
    get_template,
    list_available_categories,
)
from data_pipeline.validators.duplicate_checker import DuplicateChecker
from data_pipeline.validators.format_validator import FormatValidator


# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging(log_dir: Path, level: int = logging.INFO) -> None:
    """Postavlja file + console logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"generation_{timestamp}.log"
    
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # File handler - sve
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler - samo INFO i vise
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)
    
    logging.info(f"Logging postavljen. Log fajl: {log_file}")


# ============================================================
# ORCHESTRATOR
# ============================================================

class GenerationOrchestrator:
    """
    Glavni orchestrator koji spaja sve komponente pipeline-a.
    """
    
    BATCH_SIZE = 10  # koliko primera trazimo od LLM-a po pozivu
    MAX_RETRIES_PER_BATCH = 3
    SLEEP_BETWEEN_CALLS = 4.0  # sekunde - postujemo rate limit (15 RPM)
    
    def __init__(
        self,
        config: DatasetConfig,
        api_key: str,
        output_dir: Path,
        target_samples_override: int | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_samples_override = target_samples_override
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Inicijalizuj komponente
        self.generator = GeminiGenerator(api_key=api_key)
        self.format_validator = FormatValidator(
            min_text_length=15,
            max_text_length=400,
        )
        self.duplicate_checker = DuplicateChecker(
            near_duplicate_threshold=0.85,
        )
        
        # Output fajl
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = self.output_dir / f"training_v3_{timestamp}.jsonl"
        
        # Statistika
        self.start_time = time.time()
        self.total_attempted = 0
        self.total_saved = 0
        self.per_category_saved: dict[str, int] = {}
    
    def run(self, only_category: str | None = None) -> None:
        """
        Pokrece generisanje.
        
        Args:
            only_category: Ako je dat, generise samo tu kategoriju.
                          Inace generise sve iz YAML-a.
        """
        self.logger.info("=" * 70)
        self.logger.info("Safyrix Data Generation - START")
        self.logger.info("=" * 70)
        self.logger.info(f"Output fajl: {self.output_file}")
        
        # Odaberi kategorije
        if only_category:
            if only_category not in self.config.categories:
                raise ValueError(
                    f"Kategorija '{only_category}' ne postoji u config-u. "
                    f"Dostupne: {list(self.config.categories.keys())}"
                )
            categories_to_generate = {only_category: self.config.categories[only_category]}
        else:
            # Samo kategorije za koje imamo prompt template
            available_in_templates = set(list_available_categories())
            categories_to_generate = {
                name: cat for name, cat in self.config.categories.items()
                if name in available_in_templates
            }
            
            missing = set(self.config.categories.keys()) - available_in_templates
            if missing:
                self.logger.warning(
                    f"Sledece kategorije nemaju prompt template (preskacem): {missing}"
                )
        
        self.logger.info(f"Kategorije za generisanje: {list(categories_to_generate.keys())}")
        
        # Generisi po kategoriji
        for cat_name, cat_config in categories_to_generate.items():
            self.logger.info(f"\n{'─' * 70}")
            self.logger.info(f"Kategorija: {cat_name} (target: {cat_config.target_samples})")
            self.logger.info(f"{'─' * 70}")
            
            target = self.target_samples_override or cat_config.target_samples
            self._generate_category(cat_name, cat_config, target)
        
        # Final report
        self._print_final_report()
    
    def _generate_category(
        self,
        cat_name: str,
        cat_config: CategoryConfig,
        target_samples: int,
    ) -> None:
        """Generise primere za jednu kategoriju dok ne dostigne target."""
        template = get_template(cat_name)
        saved_count = 0
        consecutive_failures = 0
        max_consecutive_failures = 5
        
        progress = tqdm(
            total=target_samples,
            desc=f"  {cat_name}",
            unit="primera",
        )
        
        while saved_count < target_samples:
            if consecutive_failures >= max_consecutive_failures:
                self.logger.error(
                    f"Previse uzastopnih neuspeha za {cat_name}, prekidam."
                )
                break
            
            # Koliko trazimo u ovom batch-u
            remaining = target_samples - saved_count
            batch_size = min(self.BATCH_SIZE, remaining)
            
            try:
                request = GenerationRequest(
                    category=cat_name,
                    template=template,
                    n_samples=batch_size,
                )
                
                self.total_attempted += batch_size
                result = self.generator.generate(request)
                
                if not result.success:
                    consecutive_failures += 1
                    self.logger.warning(
                        f"Batch neuspeh ({consecutive_failures}/{max_consecutive_failures}): "
                        f"{result.error}"
                    )
                    time.sleep(self.SLEEP_BETWEEN_CALLS * 2)
                    continue
                
                # Validiraj i snimi
                new_saved = self._process_samples(result.samples, cat_name)
                saved_count += new_saved
                progress.update(new_saved)
                
                if new_saved == 0:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                
                # Rate limiting
                time.sleep(self.SLEEP_BETWEEN_CALLS)
            
            except Exception as e:
                consecutive_failures += 1
                self.logger.error(f"Greska u batch-u: {e}", exc_info=True)
                time.sleep(self.SLEEP_BETWEEN_CALLS * 2)
        
        progress.close()
        self.per_category_saved[cat_name] = saved_count
        self.logger.info(f"  Kategorija '{cat_name}' zavrsena: {saved_count}/{target_samples}")
    
    def _process_samples(self, samples: list, cat_name: str) -> int:
        """Validira primere i snima validne. Vraca broj sacuvanih."""
        saved = 0
        
        for sample in samples:
            # Format validacija
            sample_dict = sample.to_dict()
            sample_for_validation = {"text": sample_dict["text"]}
            
            format_result = self.format_validator.validate(sample_for_validation)
            if not format_result.is_valid:
                continue
            
            # Duplikat provera
            dup_result = self.duplicate_checker.check_and_register(sample_dict["text"])
            if dup_result.is_duplicate:
                continue
            
            # Snimi u JSONL
            self._write_sample(sample_dict)
            saved += 1
            self.total_saved += 1
        
        return saved
    
    def _write_sample(self, sample_dict: dict[str, Any]) -> None:
        """Dopisuje jedan primer u output JSONL fajl."""
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sample_dict, ensure_ascii=False) + "\n")
    
    def _print_final_report(self) -> None:
        """Stampa zavrsni izvestaj."""
        duration = time.time() - self.start_time
        
        self.logger.info("\n" + "=" * 70)
        self.logger.info("ZAVRSNI IZVESTAJ")
        self.logger.info("=" * 70)
        self.logger.info(f"Output fajl: {self.output_file}")
        self.logger.info(f"Trajanje: {duration:.1f}s ({duration/60:.1f}min)")
        self.logger.info(f"Pokusano: {self.total_attempted}")
        self.logger.info(f"Sacuvano: {self.total_saved}")
        if self.total_attempted > 0:
            self.logger.info(
                f"Stopa uspeha: {self.total_saved/self.total_attempted:.1%}"
            )
        
        self.logger.info(f"\nPo kategoriji:")
        for cat_name, count in self.per_category_saved.items():
            self.logger.info(f"  {cat_name}: {count}")
        
        # Generator stats
        self.logger.info(f"\nGenerator stats:")
        for k, v in self.generator.get_stats().items():
            self.logger.info(f"  {k}: {v}")
        
        # Validator stats
        self.logger.info(f"\nFormat validator stats:")
        for k, v in self.format_validator.get_stats().items():
            if k != "error_distribution":
                self.logger.info(f"  {k}: {v}")
        
        self.logger.info(f"\nDuplicate checker stats:")
        for k, v in self.duplicate_checker.get_stats().items():
            self.logger.info(f"  {k}: {v}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Safyrix Data Generation Pipeline")
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Generisi samo ovu kategoriju (default: sve)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Override target samples za testiranje (npr. --samples 50)",
    )
    args = parser.parse_args()
    
    # Putanje
    root_dir = Path(__file__).resolve().parent.parent
    yaml_path = root_dir / "data_pipeline" / "config" / "categories.yaml"
    output_dir = root_dir / "data_pipeline" / "output"
    log_dir = root_dir / "data_pipeline" / "logs"
    
    # Logging
    setup_logging(log_dir)
    
    # Env
    load_dotenv(root_dir / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY nije postavljen u .env")
        sys.exit(1)
    
    # Config
    config = load_categories(yaml_path)
    
    # Orchestrator
    orchestrator = GenerationOrchestrator(
        config=config,
        api_key=api_key,
        output_dir=output_dir,
        target_samples_override=args.samples,
    )
    
    orchestrator.run(only_category=args.category)


if __name__ == "__main__":
    main()
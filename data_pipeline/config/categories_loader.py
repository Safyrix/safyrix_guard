"""
Safyrix Data Generation - Categories Loader
============================================

Ucitava categories.yaml i konvertuje ga u Python objekte.

Validira da je konfiguracija konzistentna pre nego sto se koristi.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class CategoryConfig:
    """Konfiguracija jedne kategorije."""
    name: str
    risk_level: str
    target_samples: int
    description: str
    subcategories: list[str] = field(default_factory=list)
    seed_examples: list[str] = field(default_factory=list)
    notes: str | None = None


@dataclass
class RiskLevelConfig:
    """Konfiguracija risk level-a."""
    name: str
    target_ratio: float
    description: str


@dataclass
class DatasetConfig:
    """Kompletna konfiguracija dataseta."""
    version: str
    language: str
    total_target_samples: int
    description: str
    risk_levels: dict[str, RiskLevelConfig]
    categories: dict[str, CategoryConfig]
    
    def get_categories_by_risk_level(self, risk_level: str) -> list[CategoryConfig]:
        """Vraca sve kategorije datog risk level-a."""
        return [
            cat for cat in self.categories.values()
            if cat.risk_level == risk_level
        ]
    
    def get_total_samples_planned(self) -> int:
        """Suma svih target_samples po kategorijama."""
        return sum(cat.target_samples for cat in self.categories.values())
    
    def validate(self) -> list[str]:
        """
        Validira konzistentnost konfiguracije.
        Vraca listu errors (prazna ako je sve OK).
        """
        errors = []
        
        # Risk level ratios moraju da sumiraju 1.0
        total_ratio = sum(rl.target_ratio for rl in self.risk_levels.values())
        if abs(total_ratio - 1.0) > 0.01:
            errors.append(
                f"Risk level ratios moraju da sumiraju 1.0, ali sumiraju {total_ratio:.2f}"
            )
        
        # Sve kategorije moraju imati validan risk_level
        valid_risk_levels = set(self.risk_levels.keys())
        for cat_name, cat in self.categories.items():
            if cat.risk_level not in valid_risk_levels:
                errors.append(
                    f"Kategorija '{cat_name}' ima nevazeci risk_level: '{cat.risk_level}'. "
                    f"Validni: {valid_risk_levels}"
                )
        
        # Total target samples mora da odgovara sumi po kategorijama (ili bude blizu)
        category_total = self.get_total_samples_planned()
        if category_total != self.total_target_samples:
            # Warning, ne error
            errors.append(
                f"WARNING: total_target_samples={self.total_target_samples}, "
                f"ali suma po kategorijama je {category_total}"
            )
        
        return errors


def load_categories(yaml_path: Path | str) -> DatasetConfig:
    """
    Ucitava categories.yaml i vraca DatasetConfig.
    
    Args:
        yaml_path: Putanja do YAML fajla
    
    Returns:
        DatasetConfig sa svim kategorijama
    
    Raises:
        FileNotFoundError: Ako fajl ne postoji
        ValueError: Ako konfiguracija nije validna
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML fajl ne postoji: {yaml_path}")
    
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    
    # Parse metadata
    metadata = data.get("metadata", {})
    
    # Parse risk levels
    risk_levels = {}
    for name, cfg in data.get("risk_levels", {}).items():
        risk_levels[name] = RiskLevelConfig(
            name=name,
            target_ratio=cfg["target_ratio"],
            description=cfg.get("description", ""),
        )
    
    # Parse categories
    categories = {}
    for name, cfg in data.get("categories", {}).items():
        categories[name] = CategoryConfig(
            name=name,
            risk_level=cfg["risk_level"],
            target_samples=cfg["target_samples"],
            description=cfg.get("description", ""),
            subcategories=cfg.get("subcategories", []),
            seed_examples=cfg.get("seed_examples", []),
            notes=cfg.get("notes"),
        )
    
    config = DatasetConfig(
        version=metadata.get("version", "0.0.0"),
        language=metadata.get("language", "sr"),
        total_target_samples=metadata.get("total_target_samples", 0),
        description=metadata.get("description", ""),
        risk_levels=risk_levels,
        categories=categories,
    )
    
    # Validacija
    errors = config.validate()
    real_errors = [e for e in errors if not e.startswith("WARNING")]
    
    if real_errors:
        error_msg = "\n".join(f"  - {e}" for e in real_errors)
        raise ValueError(f"Konfiguracija nije validna:\n{error_msg}")
    
    return config


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    # Pronadji root putanju
    root_dir = Path(__file__).resolve().parent.parent.parent
    yaml_path = root_dir / "data_pipeline" / "config" / "categories.yaml"
    
    print(f"Ucitavam: {yaml_path}\n")
    
    try:
        config = load_categories(yaml_path)
        
        print(f"Verzija: {config.version}")
        print(f"Jezik: {config.language}")
        print(f"Target samples (total): {config.total_target_samples}")
        print(f"Planirano po kategorijama: {config.get_total_samples_planned()}\n")
        
        print(f"Risk levels:")
        for name, rl in config.risk_levels.items():
            print(f"  {name}: ratio={rl.target_ratio:.2f}")
        
        print(f"\nKategorije ({len(config.categories)}):")
        for name, cat in config.categories.items():
            print(f"  {name} [{cat.risk_level}]: {cat.target_samples} primera")
        
        print(f"\nLow risk kategorije:")
        for cat in config.get_categories_by_risk_level("low"):
            print(f"  - {cat.name}: {cat.target_samples} primera")
        
        # Validacija (warnings ce se pojaviti ako postoje)
        warnings = [e for e in config.validate() if e.startswith("WARNING")]
        if warnings:
            print(f"\nUpozorenja:")
            for w in warnings:
                print(f"  {w}")
        
        print("\nKonfiguracija ucitana uspesno.")
    
    except Exception as e:
        print(f"GRESKA: {e}")
        raise
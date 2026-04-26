"""
Safyrix Data Generation - Multi-Provider Generator
====================================================

Enterprise-grade generator koji koristi vise LLM provajdera
sa pametnim routing-om i automatskim fallback-om.

Patterns:
- Composite (sastoji se od vise BaseGenerator-a)
- Strategy (razlicite routing strategije)
- Circuit Breaker (privremeno iskljuci nezdrav provider)

Prednosti:
- Vise dnevne kvote (zbir svih providera)
- Robusnost (jedan padne, drugi nastavlja)
- Diverzitet podataka (razliciti modeli, razliciti stilovi)
- Performance (Groq brz, Gemini pouzdan)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base_generator import (
    BaseGenerator,
    GenerationRequest,
    GenerationResult,
)


# ============================================================
# ROUTING STRATEGIES
# ============================================================

class RoutingStrategy(Enum):
    """Strategije za biranje generatora."""
    ROUND_ROBIN = "round_robin"
    FAILOVER = "failover"
    WEIGHTED = "weighted"


# ============================================================
# PROVIDER HEALTH TRACKING
# ============================================================

@dataclass
class ProviderHealth:
    """Tracking zdravlja jednog providera."""
    provider_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    last_failure_time: float | None = None
    is_quarantined: bool = False
    quarantine_until: float | None = None
    
    def record_success(self) -> None:
        self.total_calls += 1
        self.successful_calls += 1
        self.consecutive_failures = 0
    
    def record_failure(self) -> None:
        self.total_calls += 1
        self.failed_calls += 1
        self.consecutive_failures += 1
        self.last_failure_time = time.time()
    
    def quarantine(self, duration_seconds: float = 300) -> None:
        """Privremeno iskljuci provider zbog ponovljenih neuspeha."""
        self.is_quarantined = True
        self.quarantine_until = time.time() + duration_seconds
    
    def check_quarantine_expired(self) -> bool:
        """Vrati True ako je quarantine istekao."""
        if not self.is_quarantined:
            return True
        if self.quarantine_until and time.time() >= self.quarantine_until:
            self.is_quarantined = False
            self.quarantine_until = None
            self.consecutive_failures = 0  # daj sansu
            return True
        return False
    
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 1.0  # benefit of doubt
        return self.successful_calls / self.total_calls


# ============================================================
# MULTI-PROVIDER GENERATOR
# ============================================================

class MultiProviderGenerator(BaseGenerator):
    """
    Generator koji rotira izmedju vise provajdera.
    
    Implementira BaseGenerator interface, pa ga pipeline koristi
    kao bilo koji drugi generator - transparentno.
    """
    
    generator_name = "multi_provider"
    generator_version = "1.0.0"
    
    # Threshold za quarantine
    MAX_CONSECUTIVE_FAILURES = 3
    QUARANTINE_DURATION_SECONDS = 300  # 5 minuta
    
    def __init__(
        self,
        providers: list[BaseGenerator],
        strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN,
        weights: dict[str, float] | None = None,
        api_key: str = "n/a",  # nije relevantno za ovaj generator
        **kwargs: Any,
    ) -> None:
        """
        Args:
            providers: Lista BaseGenerator instanci (Gemini, Groq, ...)
            strategy: Kako biramo ko obrađuje sledeci zahtev
            weights: Za WEIGHTED strategy - {"gemini": 0.3, "groq": 0.7}
        """
        super().__init__(api_key=api_key, **kwargs)
        
        if not providers:
            raise ValueError("MultiProviderGenerator zahteva bar jedan provider")
        
        self.providers = providers
        self.strategy = strategy
        self.weights = weights or {}
        
        # Health tracking
        self.health: dict[str, ProviderHealth] = {
            p.generator_name: ProviderHealth(provider_name=p.generator_name)
            for p in providers
        }
        
        # Round-robin state
        self._round_robin_index = 0
        self._round_robin_lock = False  # za thread-safety u buducnosti
        
        self.logger.info(
            f"MultiProviderGenerator inicijalizovan sa {len(providers)} provajdera: "
            f"{[p.generator_name for p in providers]}, strategy={strategy.value}"
        )
    
    # ============================================================
    # IMPLEMENTACIJA ABSTRACT METODA
    # ============================================================
    
    def _call_llm(self, system_role: str, user_prompt: str) -> str:
        """
        Ovo MultiProviderGenerator ne koristi direktno.
        Override-ujemo `generate()` umesto toga.
        """
        raise NotImplementedError(
            "MultiProviderGenerator ne koristi _call_llm direktno. "
            "Generisanje ide kroz `generate()` metodu."
        )
    
    def _parse_response(
        self,
        raw_response: str,
        request: GenerationRequest,
    ) -> list[dict[str, Any]]:
        """Isto - parsiranje delegira na konkretne providere."""
        raise NotImplementedError(
            "MultiProviderGenerator ne parsira direktno."
        )
    
    # ============================================================
    # OVERRIDE PUBLIC API
    # ============================================================
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Glavni generation flow sa multi-provider routing-om.
        
        1. Bira providera prema strategiji
        2. Pokusa sa njim
        3. Ako padne, fallback na sledeceg
        4. Belezi health metrics
        """
        # Lista providera koje cemo probati (po prioritetu)
        provider_order = self._select_provider_order(request)
        
        if not provider_order:
            self.logger.error("Svi provideri su nedostupni!")
            return GenerationResult(
                request_id=request.request_id,
                samples=[],
                success=False,
                error="No healthy providers available",
            )
        
        last_error = None
        
        for provider in provider_order:
            health = self.health[provider.generator_name]
            
            # Skip if quarantined
            if not health.check_quarantine_expired():
                self.logger.debug(
                    f"Provider {provider.generator_name} je u quarantine, "
                    f"preskacem"
                )
                continue
            
            self.logger.info(
                f"Pokusavam sa providerom: {provider.generator_name} "
                f"(success rate: {health.success_rate():.0%})"
            )
            
            try:
                result = provider.generate(request)
                
                if result.success:
                    health.record_success()
                    self._total_calls += 1
                    self._total_tokens += result.tokens_used
                    
                    self.logger.info(
                        f"Provider {provider.generator_name} uspeo: "
                        f"{len(result.samples)} primera"
                    )
                    return result
                else:
                    health.record_failure()
                    last_error = result.error
                    self.logger.warning(
                        f"Provider {provider.generator_name} neuspeh: {result.error}"
                    )
                    
                    # Quarantine if too many failures
                    if health.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                        self.logger.warning(
                            f"Provider {provider.generator_name} stavljen u "
                            f"quarantine na {self.QUARANTINE_DURATION_SECONDS}s"
                        )
                        health.quarantine(self.QUARANTINE_DURATION_SECONDS)
            
            except Exception as e:
                health.record_failure()
                last_error = str(e)
                self._total_errors += 1
                self.logger.error(
                    f"Provider {provider.generator_name} bacio exception: {e}"
                )
                
                if health.consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                    health.quarantine(self.QUARANTINE_DURATION_SECONDS)
        
        # Svi su pali
        return GenerationResult(
            request_id=request.request_id,
            samples=[],
            success=False,
            error=f"Svi provideri su pali. Poslednja greska: {last_error}",
        )
    
    # ============================================================
    # STATS
    # ============================================================
    
    def get_stats(self) -> dict[str, Any]:
        """Vraca agregirane statistike + per-provider breakdown."""
        base_stats = super().get_stats()
        
        per_provider = {}
        for name, health in self.health.items():
            per_provider[name] = {
                "total_calls": health.total_calls,
                "successful": health.successful_calls,
                "failed": health.failed_calls,
                "success_rate": f"{health.success_rate():.1%}",
                "is_quarantined": health.is_quarantined,
            }
        
        # Per-provider real stats
        for provider in self.providers:
            if provider.generator_name in per_provider:
                provider_stats = provider.get_stats()
                per_provider[provider.generator_name]["tokens"] = provider_stats.get("total_tokens", 0)
        
        base_stats["providers"] = per_provider
        base_stats["strategy"] = self.strategy.value
        
        return base_stats
    
    # ============================================================
    # PRIVATE - ROUTING LOGIC
    # ============================================================
    
    def _select_provider_order(
        self,
        request: GenerationRequest,
    ) -> list[BaseGenerator]:
        """
        Vraca uredjen niz providera za pokusavanje (prvi je primarni).
        Ostatak su fallback-ovi po redosledu.
        """
        # Filtriraj zdrave (ili expired-quarantine)
        healthy = [
            p for p in self.providers
            if self.health[p.generator_name].check_quarantine_expired()
        ]
        
        if not healthy:
            return []
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_order(healthy)
        elif self.strategy == RoutingStrategy.FAILOVER:
            return healthy  # zadrzi originalan redosled
        elif self.strategy == RoutingStrategy.WEIGHTED:
            return self._weighted_order(healthy)
        else:
            return healthy
    
    def _round_robin_order(
        self,
        healthy: list[BaseGenerator],
    ) -> list[BaseGenerator]:
        """Round-robin: svaki sledeci poziv ide na sledeceg providera."""
        if not healthy:
            return []
        
        # Pomeri index
        primary_idx = self._round_robin_index % len(healthy)
        self._round_robin_index += 1
        
        # Vrati: primary, pa ostali kao fallback
        primary = healthy[primary_idx]
        fallbacks = [p for i, p in enumerate(healthy) if i != primary_idx]
        
        return [primary] + fallbacks
    
    def _weighted_order(
        self,
        healthy: list[BaseGenerator],
    ) -> list[BaseGenerator]:
        """Weighted: bira primarno na osnovu tezine, ostali kao fallback."""
        import random
        
        if not healthy:
            return []
        
        # Default weights (uniform)
        weights = []
        for provider in healthy:
            w = self.weights.get(provider.generator_name, 1.0)
            weights.append(w)
        
        # Random choice based on weights
        primary = random.choices(healthy, weights=weights, k=1)[0]
        fallbacks = [p for p in healthy if p is not primary]
        
        return [primary] + fallbacks


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    from .gemini_generator import GeminiGenerator
    from .groq_generator import GroqGenerator
    from .prompt_templates import get_template
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if not gemini_key or not groq_key:
        print("ERROR: GEMINI_API_KEY i/ili GROQ_API_KEY nisu u .env")
        exit(1)
    
    print("Testiranje MultiProviderGenerator-a...\n")
    
    # Inicijalizuj sub-providere
    gemini = GeminiGenerator(api_key=gemini_key)
    groq = GroqGenerator(api_key=groq_key)
    
    # Multi-provider sa round-robin
    multi = MultiProviderGenerator(
        providers=[gemini, groq],
        strategy=RoutingStrategy.ROUND_ROBIN,
    )
    
    print(f"Generator info:")
    print(f"  Name: {multi.generator_name}")
    print(f"  Strategy: {multi.strategy.value}")
    print(f"  Providers: {[p.generator_name for p in multi.providers]}")
    
    # Pokusaj 3 zahteva (videcemo round-robin)
    print("\nPravimo 3 zahteva za safe_personal (3 primera svaki)...")
    
    for i in range(3):
        request = GenerationRequest(
            category="safe_personal",
            template=get_template("safe_personal"),
            n_samples=3,
        )
        
        print(f"\n--- Zahtev {i+1} ---")
        result = multi.generate(request)
        
        if result.success:
            print(f"  Uspesno: {len(result.samples)} primera")
            for sample in result.samples[:2]:
                print(f"    [{sample.generator_name}] {sample.text[:60]}")
        else:
            print(f"  Neuspeh: {result.error}")
    
    # Final stats
    print("\n" + "=" * 60)
    print("FINAL STATS:")
    stats = multi.get_stats()
    for k, v in stats.items():
        if k == "providers":
            print(f"  providers:")
            for pname, pstats in v.items():
                print(f"    {pname}: {pstats}")
        else:
            print(f"  {k}: {v}")
    
    print("\nMultiProviderGenerator test PROSAO.")
"""
Safyrix Data Generation - Format Validator
===========================================

Validira da li generisani primer zadovoljava format zahteve.

Pristup: Pydantic-style validacija sa jasnim error porukama.
Svaki problem se logira sa context-om za debugging.
"""

from dataclasses import dataclass, field
from typing import Any
import re
import logging


@dataclass
class ValidationResult:
    """Rezultat validacije jednog primera."""
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    
    def add_error(self, message: str) -> None:
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str) -> None:
        self.warnings.append(message)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
        }


class FormatValidator:
    """
    Validator za format generisanih primera.
    
    Konfigurabilno preko constructor parametara.
    """
    
    def __init__(
        self,
        min_text_length: int = 10,
        max_text_length: int = 500,
        required_fields: list[str] | None = None,
        forbidden_patterns: list[str] | None = None,
    ) -> None:
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.required_fields = required_fields or ["text"]
        self.forbidden_patterns = forbidden_patterns or []
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistika
        self._total_validated = 0
        self._total_passed = 0
        self._total_failed = 0
        self._error_counts: dict[str, int] = {}
    
    def validate(self, sample: dict[str, Any]) -> ValidationResult:
        """
        Validira jedan primer.
        
        Args:
            sample: Dict sa primerom (mora da ima bar 'text' polje)
        
        Returns:
            ValidationResult sa is_valid, errors, warnings
        """
        result = ValidationResult(is_valid=True)
        self._total_validated += 1
        
        # 1. Provera obaveznih polja
        for field_name in self.required_fields:
            if field_name not in sample:
                result.add_error(f"Nedostaje obavezno polje: '{field_name}'")
        
        # Ako fali tekst, dalji testovi nemaju smisla
        text = sample.get("text", "")
        if not isinstance(text, str):
            result.add_error(f"Polje 'text' mora biti string, dobio sam: {type(text).__name__}")
            self._update_stats(result)
            return result
        
        text = text.strip()
        
        # 2. Provera duzine
        if len(text) < self.min_text_length:
            result.add_error(
                f"Tekst je prekratak: {len(text)} karaktera "
                f"(minimum: {self.min_text_length})"
            )
        
        if len(text) > self.max_text_length:
            result.add_error(
                f"Tekst je predugacak: {len(text)} karaktera "
                f"(maksimum: {self.max_text_length})"
            )
        
        # 3. Provera zabranjenih obrazaca
        for pattern in self.forbidden_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                result.add_error(f"Tekst sadrzi zabranjeni obrazac: {pattern}")
        
        # 4. Provera da nije samo whitespace ili ponavljanja
        if not text or text.isspace():
            result.add_error("Tekst je prazan ili sadrzi samo whitespace")
        
        # 5. Provera ponavljanja istog karaktera (npr. "aaaaaaaa")
        if self._has_excessive_repetition(text):
            result.add_warning("Tekst sadrzi sumnjiva ponavljanja karaktera")
        
        # 6. Provera da li je verovatno na srpskom (osnovna heuristika)
        if not self._looks_like_serbian(text):
            result.add_warning("Tekst mozda nije na srpskom")
        
        self._update_stats(result)
        return result
    
    def validate_batch(
        self,
        samples: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[ValidationResult]]:
        """
        Validira listu primera.
        
        Returns:
            Tuple (validni_primeri, svi_rezultati)
        """
        valid_samples = []
        all_results = []
        
        for sample in samples:
            result = self.validate(sample)
            all_results.append(result)
            
            if result.is_valid:
                valid_samples.append(sample)
            else:
                self.logger.debug(
                    f"Primer odbijen: {result.errors}. Tekst: {sample.get('text', '')[:50]}..."
                )
        
        return valid_samples, all_results
    
    def get_stats(self) -> dict[str, Any]:
        """Vraca statistiku validacije."""
        pass_rate = (
            self._total_passed / self._total_validated
            if self._total_validated > 0 else 0.0
        )
        
        return {
            "total_validated": self._total_validated,
            "total_passed": self._total_passed,
            "total_failed": self._total_failed,
            "pass_rate": pass_rate,
            "error_distribution": dict(self._error_counts),
        }
    
    # ============================================================
    # PRIVATE HELPERS
    # ============================================================
    
    def _update_stats(self, result: ValidationResult) -> None:
        if result.is_valid:
            self._total_passed += 1
        else:
            self._total_failed += 1
            for error in result.errors:
                # Brojimo po prvih 50 karaktera error poruke (da grupisemo slicne)
                error_key = error[:50]
                self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
    
    def _has_excessive_repetition(self, text: str) -> bool:
        """Detektuje patoloska ponavljanja kao 'aaaaaa' ili 'lololololo'."""
        # Ako se isti karakter pojavi 5+ puta uzastopno
        if re.search(r"(.)\1{4,}", text):
            return True
        # Ako se ista 2-3 karaktera ponavljaju 4+ puta
        if re.search(r"(.{2,3})\1{3,}", text):
            return True
        return False
    
    def _looks_like_serbian(self, text: str) -> bool:
        """
        Osnovna heuristika za prepoznavanje srpskog teksta.
        Ne savrseno, ali dovoljno za filter.
        """
        # Najcesce srpske reci
        serbian_indicators = [
            "je", "se", "da", "ne", "i", "u", "na", "za", "od", "su",
            "sam", "si", "smo", "ste", "sve", "ovo", "to", "ovaj",
            "kako", "sta", "gde", "kada", "ko", "moj", "tvoj", "nas",
            "ali", "ako", "ili", "samo", "tako", "evo", "treba",
        ]
        
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        
        if len(words) < 2:
            # Prekratak za heuristiku, daj benefit of doubt
            return True
        
        # Bar 1 srpska rec u tekstu
        return any(word in serbian_indicators for word in words)


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Testiranje FormatValidator-a...\n")
    
    validator = FormatValidator(
        min_text_length=10,
        max_text_length=200,
    )
    
    # Test slucajevi
    test_cases = [
        # Validni
        {"text": "Hej, kako si? Vidimo se sutra u kaficu."},
        {"text": "Sastanak je pomeren za 15h u sali za konferencije."},
        
        # Nevalidni - prekratak
        {"text": "Hi"},
        
        # Nevalidni - prazan
        {"text": ""},
        
        # Nevalidni - bez teksta
        {"category": "test"},
        
        # Nevalidan - predugacak
        {"text": "a" * 250},
        
        # Validan ali warning - mozda nije srpski
        {"text": "Hello world this is english text."},
        
        # Warning - ponavljanja
        {"text": "Heeeeeeej kako siiiiii bate."},
    ]
    
    for i, sample in enumerate(test_cases, 1):
        result = validator.validate(sample)
        status = "PROSAO" if result.is_valid else "ODBIJEN"
        text_preview = sample.get("text", "(no text)")[:50]
        print(f"[{i}] {status}: '{text_preview}'")
        for error in result.errors:
            print(f"    GRESKA: {error}")
        for warning in result.warnings:
            print(f"    UPOZORENJE: {warning}")
        print()
    
    # Statistika
    print("=" * 60)
    print("Statistika:")
    stats = validator.get_stats()
    print(f"  Validirano: {stats['total_validated']}")
    print(f"  Proslo: {stats['total_passed']}")
    print(f"  Odbijeno: {stats['total_failed']}")
    print(f"  Pass rate: {stats['pass_rate']:.1%}")
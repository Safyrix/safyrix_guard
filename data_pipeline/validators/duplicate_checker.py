"""
Safyrix Data Generation - Duplicate Checker
============================================

Detektuje duplikate i skoro-duplikate u generisanim primerima.

Strategije:
1. Egzaktni match (normalizovani tekst)
2. Hash-based fuzzy match (n-gram shingles)
3. Levenshtein-style approximate match (za blize duplikate)

Performance: O(1) lookup koristeci set/dict za vec generisane.
"""

from dataclasses import dataclass
from typing import Any
import hashlib
import re
import logging


@dataclass
class DuplicateCheckResult:
    """Rezultat provere duplikata za jedan primer."""
    is_duplicate: bool
    match_type: str | None = None  # "exact" | "near" | None
    matched_against: str | None = None  # tekst sa kojim se poklapa
    similarity_score: float = 0.0


class DuplicateChecker:
    """
    Stateful checker - cuva sve videne primere i proverava nove protiv njih.
    
    Konfigurabilan prag slicnosti za "near duplicate" detekciju.
    """
    
    def __init__(
        self,
        near_duplicate_threshold: float = 0.85,
        ngram_size: int = 3,
    ) -> None:
        """
        Args:
            near_duplicate_threshold: Jaccard similarity prag (0.0-1.0)
                                     Veci broj = strozi (manje duplikata)
            ngram_size: Velicina karakter n-grama za fuzzy match
        """
        self.near_duplicate_threshold = near_duplicate_threshold
        self.ngram_size = ngram_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Storage
        self._exact_hashes: set[str] = set()
        self._all_texts: list[str] = []  # za near-duplicate proveru
        self._ngram_index: dict[str, set[int]] = {}  # n-gram -> indeksi tekstova
        
        # Statistika
        self._total_checked = 0
        self._exact_duplicates = 0
        self._near_duplicates = 0
        self._unique_count = 0
    
    def check_and_register(self, text: str) -> DuplicateCheckResult:
        """
        Proverava da li je tekst duplikat, i ako nije - registruje ga.
        
        Args:
            text: Tekst za proveru
        
        Returns:
            DuplicateCheckResult sa is_duplicate i metadata
        """
        self._total_checked += 1
        
        normalized = self._normalize(text)
        
        # 1. Egzaktni match check
        text_hash = self._hash(normalized)
        if text_hash in self._exact_hashes:
            self._exact_duplicates += 1
            return DuplicateCheckResult(
                is_duplicate=True,
                match_type="exact",
                matched_against=normalized[:50],
                similarity_score=1.0,
            )
        
        # 2. Near-duplicate check
        match_idx, similarity = self._find_near_duplicate(normalized)
        if match_idx is not None and similarity >= self.near_duplicate_threshold:
            self._near_duplicates += 1
            return DuplicateCheckResult(
                is_duplicate=True,
                match_type="near",
                matched_against=self._all_texts[match_idx][:50],
                similarity_score=similarity,
            )
        
        # 3. Nije duplikat - registruj
        self._register(normalized, text_hash)
        self._unique_count += 1
        
        return DuplicateCheckResult(
            is_duplicate=False,
            similarity_score=0.0,
        )
    
    def filter_unique(self, texts: list[str]) -> list[str]:
        """
        Filtrira listu - vraca samo jedinstvene tekstove.
        
        Convenience metoda za batch obradu.
        """
        unique = []
        for text in texts:
            result = self.check_and_register(text)
            if not result.is_duplicate:
                unique.append(text)
        return unique
    
    def reset(self) -> None:
        """Resetuje state - korisno za testiranje."""
        self._exact_hashes.clear()
        self._all_texts.clear()
        self._ngram_index.clear()
        self._total_checked = 0
        self._exact_duplicates = 0
        self._near_duplicates = 0
        self._unique_count = 0
    
    def get_stats(self) -> dict[str, Any]:
        """Vraca statistiku."""
        return {
            "total_checked": self._total_checked,
            "exact_duplicates": self._exact_duplicates,
            "near_duplicates": self._near_duplicates,
            "unique_count": self._unique_count,
            "duplicate_rate": (
                (self._exact_duplicates + self._near_duplicates) / self._total_checked
                if self._total_checked > 0 else 0.0
            ),
        }
    
    # ============================================================
    # PRIVATE HELPERS
    # ============================================================
    
    def _normalize(self, text: str) -> str:
        """Normalizuje tekst za poredjenje (lowercase, trim, whitespace)."""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)  # multiple spaces -> single
        text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
        return text
    
    def _hash(self, text: str) -> str:
        """SHA256 hash teksta za egzaktni match."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _get_ngrams(self, text: str) -> set[str]:
        """Vraca skup karakter n-grama datog teksta."""
        if len(text) < self.ngram_size:
            return {text}
        return {
            text[i:i + self.ngram_size]
            for i in range(len(text) - self.ngram_size + 1)
        }
    
    def _jaccard_similarity(self, set1: set[str], set2: set[str]) -> float:
        """Jaccard similarity izmedju dva skupa."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    def _find_near_duplicate(
        self,
        normalized_text: str,
    ) -> tuple[int | None, float]:
        """
        Trazi najslicniji vec videni tekst.
        
        Optimizacija: koristi n-gram indeks da brzo eliminise nemogucnosti.
        """
        if not self._all_texts:
            return None, 0.0
        
        candidate_ngrams = self._get_ngrams(normalized_text)
        
        # Nadji kandidate koji dele bar jedan n-gram
        candidate_indices: set[int] = set()
        for ngram in candidate_ngrams:
            if ngram in self._ngram_index:
                candidate_indices.update(self._ngram_index[ngram])
        
        # Proveri samo kandidate
        best_idx = None
        best_similarity = 0.0
        
        for idx in candidate_indices:
            existing_ngrams = self._get_ngrams(self._all_texts[idx])
            similarity = self._jaccard_similarity(candidate_ngrams, existing_ngrams)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_idx = idx
        
        return best_idx, best_similarity
    
    def _register(self, normalized_text: str, text_hash: str) -> None:
        """Registruje novi tekst u indekse."""
        # Dodaj u set hashes
        self._exact_hashes.add(text_hash)
        
        # Dodaj u listu
        new_idx = len(self._all_texts)
        self._all_texts.append(normalized_text)
        
        # Dodaj u n-gram index
        ngrams = self._get_ngrams(normalized_text)
        for ngram in ngrams:
            if ngram not in self._ngram_index:
                self._ngram_index[ngram] = set()
            self._ngram_index[ngram].add(new_idx)


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    print("Testiranje DuplicateChecker-a...\n")
    
    checker = DuplicateChecker(near_duplicate_threshold=0.80)
    
    test_texts = [
        # 1. Original
        "Hej, kako si? Vidimo se sutra u kaficu.",
        
        # 2. Egzaktni duplikat (sa drugim case-om)
        "HEJ, KAKO SI? VIDIMO SE SUTRA U KAFICU.",
        
        # 3. Skoro-duplikat (jedan zarez razlicit)
        "Hej kako si? Vidimo se sutra u kaficu.",
        
        # 4. Slican ali jedinstven
        "Hej, kako si? Vidimo se kasnije u parku.",
        
        # 5. Potpuno drugaciji
        "Sastanak je pomeren za 15h u kancelariji.",
        
        # 6. Skoro-duplikat broja 5
        "Sastanak je pomeren za 16h u kancelariji.",
        
        # 7. Skoro identican broju 5
        "Sastanak je pomeren za 15h u kancelariji",
    ]
    
    for i, text in enumerate(test_texts, 1):
        result = checker.check_and_register(text)
        
        if result.is_duplicate:
            status = f"DUPLIKAT ({result.match_type}, sim={result.similarity_score:.2f})"
        else:
            status = "JEDINSTVEN"
        
        print(f"[{i}] {status}")
        print(f"    Tekst: '{text[:60]}'")
        if result.matched_against:
            print(f"    Match: '{result.matched_against}'")
        print()
    
    # Statistika
    print("=" * 60)
    print("Statistika:")
    stats = checker.get_stats()
    print(f"  Provereno: {stats['total_checked']}")
    print(f"  Egzaktni duplikati: {stats['exact_duplicates']}")
    print(f"  Skoro-duplikati: {stats['near_duplicates']}")
    print(f"  Jedinstveni: {stats['unique_count']}")
    print(f"  Duplicate rate: {stats['duplicate_rate']:.1%}")
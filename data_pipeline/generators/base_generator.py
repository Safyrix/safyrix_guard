"""
Safyrix Data Generation - Base Generator
=========================================

Apstraktna bazna klasa za sve LLM generatore.
Definise interface koji svaki generator mora da implementira.

Architecture pattern: Strategy Pattern
- Klijent (run_generation.py) ne zna koji konkretni generator koristi
- Generatori se mogu menjati u runtime-u
- Lako dodavanje novih LLM provajdera bez menjanja klijenta
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
import uuid
import logging

from .prompt_templates import PromptTemplate


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class GenerationRequest:
    """Zahtev za generisanje primera."""
    category: str
    template: PromptTemplate
    n_samples: int
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GeneratedSample:
    """Jedan generisani primer sa metadata."""
    id: str
    text: str
    risk_level: str
    category: str
    metadata: dict[str, Any]
    generator_name: str
    generator_version: str
    prompt_version: str
    created_at: datetime
    
    def to_dict(self) -> dict[str, Any]:
        """Konverzija u dict za JSON serijalizaciju."""
        return {
            "id": self.id,
            "text": self.text,
            "risk_level": self.risk_level,
            "category": self.category,
            "metadata": self.metadata,
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "prompt_version": self.prompt_version,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class GenerationResult:
    """Rezultat jednog generation poziva."""
    request_id: str
    samples: list[GeneratedSample]
    success: bool
    error: str | None = None
    api_calls_made: int = 0
    tokens_used: int = 0
    duration_seconds: float = 0.0


# ============================================================
# ABSTRACT BASE CLASS
# ============================================================

class BaseGenerator(ABC):
    """
    Apstraktna bazna klasa za sve LLM generatore.
    
    Svaka konkretna implementacija mora da:
    1. Implementira _call_llm() metodu
    2. Implementira _parse_response() metodu
    3. Definise generator_name i generator_version
    """
    
    # Subklase MORAJU da definisu ove
    generator_name: str = ""
    generator_version: str = ""
    
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        """
        Args:
            api_key: API kljuc za LLM provajdera
            **kwargs: Dodatni parametri (model_name, temperature, itd.)
        """
        if not self.generator_name or not self.generator_version:
            raise NotImplementedError(
                f"{self.__class__.__name__} mora da definise "
                f"generator_name i generator_version"
            )
        
        self.api_key = api_key
        self.config = kwargs
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Statistika
        self._total_calls = 0
        self._total_tokens = 0
        self._total_errors = 0
    
    # ============================================================
    # PUBLIC API - finalan, subklase ne menjaju
    # ============================================================
    
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """
        Glavni public API za generisanje.
        
        Template Method Pattern - definise tok izvrsenja,
        a konkretne korake delegira na abstract metode.
        """
        start_time = datetime.utcnow()
        
        self.logger.info(
            f"Pocinje generisanje: kategorija={request.category}, "
            f"n_samples={request.n_samples}, request_id={request.request_id}"
        )
        
        try:
            # Step 1: Renderuj prompt
            prompt = request.template.render(n=request.n_samples)
            
            # Step 2: Pozovi LLM (subklasa implementira)
            raw_response = self._call_llm(
                system_role=request.template.system_role,
                user_prompt=prompt,
            )
            self._total_calls += 1
            
            # Step 3: Parsiraj odgovor (subklasa implementira)
            parsed_samples = self._parse_response(raw_response, request)
            
            # Step 4: Konvertuj u GeneratedSample objekte
            samples = self._build_samples(parsed_samples, request)
            
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.info(
                f"Generisanje uspesno: {len(samples)} primera, "
                f"duration={duration:.2f}s"
            )
            
            return GenerationResult(
                request_id=request.request_id,
                samples=samples,
                success=True,
                api_calls_made=1,
                duration_seconds=duration,
            )
        
        except Exception as e:
            self._total_errors += 1
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            self.logger.error(
                f"Generisanje neuspesno: {e}",
                exc_info=True,
            )
            
            return GenerationResult(
                request_id=request.request_id,
                samples=[],
                success=False,
                error=str(e),
                duration_seconds=duration,
            )
    
    def get_stats(self) -> dict[str, Any]:
        """Vraca statistiku ovog generatora."""
        return {
            "generator_name": self.generator_name,
            "generator_version": self.generator_version,
            "total_calls": self._total_calls,
            "total_tokens": self._total_tokens,
            "total_errors": self._total_errors,
        }
    
    # ============================================================
    # ABSTRACT METHODS - subklase MORAJU implementirati
    # ============================================================
    
    @abstractmethod
    def _call_llm(self, system_role: str, user_prompt: str) -> str:
        """
        Poziva konkretni LLM API.
        
        Args:
            system_role: System role (kao ko se ponasa LLM)
            user_prompt: Renderovan user prompt
        
        Returns:
            Sirovi tekst odgovor od LLM-a
        
        Raises:
            Exception: Ako LLM poziv ne uspe
        """
        ...
    
    @abstractmethod
    def _parse_response(
        self,
        raw_response: str,
        request: GenerationRequest,
    ) -> list[dict[str, Any]]:
        """
        Parsira sirovi LLM odgovor u listu dict objekata.
        
        Args:
            raw_response: Sirov tekst od LLM-a
            request: Originalan zahtev (za kontekst)
        
        Returns:
            Lista dict objekata, svaki predstavlja jedan primer
        
        Raises:
            ValueError: Ako odgovor ne moze da se parsira
        """
        ...
    
    # ============================================================
    # PROTECTED HELPERS - subklase mogu da koriste
    # ============================================================
    
    def _build_samples(
        self,
        parsed_samples: list[dict[str, Any]],
        request: GenerationRequest,
    ) -> list[GeneratedSample]:
        """Konvertuje parsiranje rezultate u GeneratedSample objekte."""
        samples = []
        
        for parsed in parsed_samples:
            # Tekst je obavezan
            text = parsed.get("text", "").strip()
            if not text:
                self.logger.warning(f"Preskocen primer bez teksta: {parsed}")
                continue
            
            # Sve ostalo ide u metadata
            metadata = {k: v for k, v in parsed.items() if k != "text"}
            
            sample = GeneratedSample(
                id=str(uuid.uuid4()),
                text=text,
                risk_level=self._infer_risk_level(request.category),
                category=request.category,
                metadata=metadata,
                generator_name=self.generator_name,
                generator_version=self.generator_version,
                prompt_version=request.template.version,
                created_at=datetime.utcnow(),
            )
            samples.append(sample)
        
        return samples
    
    def _infer_risk_level(self, category: str) -> str:
        """
        Mapira kategoriju na risk level.
        Trenutno hardkodovano, kasnije moze da cita iz YAML-a.
        """
        risk_map = {
            "safe_personal": "low",
            "safe_business": "low",
            "safe_informational": "low",
            "suspicious_data_request": "medium",
            "suspicious_link": "medium",
            "suspicious_pressure": "medium",
            "phishing_financial": "high",
            "pii_extraction": "high",
            "threats_extortion": "high",
            "grooming_predatory": "high",
            "scam_romance": "high",
            "scam_authority": "high",
        }
        return risk_map.get(category, "medium")


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    # Demonstracija da se apstraktna klasa NE moze instancirati
    try:
        gen = BaseGenerator(api_key="test")
        print("GRESKA: Trebalo je da pukne!")
    except TypeError as e:
        print(f"OK - Apstraktna klasa pravilno odbija instanciranje:")
        print(f"   {e}")
    
    print("\nBaseGenerator modul ucitan uspesno.")
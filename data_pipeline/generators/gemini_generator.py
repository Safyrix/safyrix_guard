"""
Safyrix Data Generation - Gemini Generator
===========================================

Konkretna implementacija BaseGenerator-a koja koristi 
Google Gemini API za generisanje primera.

Inheritance: BaseGenerator
Pattern: Strategy (jedan od mogucih LLM provajdera)
"""

import json
import re
from typing import Any

import google.generativeai as genai

from .base_generator import BaseGenerator, GenerationRequest


class GeminiGenerator(BaseGenerator):
    """
    Generator koji koristi Google Gemini API.
    
    Konfiguracija (kroz **kwargs):
        - model_name: str (default: "gemini-2.0-flash")
        - temperature: float (default: 0.9 - veca varijabilnost)
        - max_output_tokens: int (default: 8192)
    """
    
    generator_name = "gemini"
    generator_version = "1.0.0"
    
    # Default konfiguracija
    DEFAULT_MODEL = "gemini-2.5-flash-lite"
    DEFAULT_TEMPERATURE = 0.9
    DEFAULT_MAX_TOKENS = 8192
    
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)
        
        # Konfiguracija Gemini SDK
        genai.configure(api_key=self.api_key)
        
        # Citaj parametre iz kwargs ili koristi default
        self.model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        self.temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        self.max_output_tokens = kwargs.get("max_output_tokens", self.DEFAULT_MAX_TOKENS)
        
        # Inicijalizuj model
        self._model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config={
                "temperature": self.temperature,
                "max_output_tokens": self.max_output_tokens,
                "response_mime_type": "application/json",
            },
        )
        
        self.logger.info(
            f"GeminiGenerator inicijalizovan: model={self.model_name}, "
            f"temperature={self.temperature}"
        )
    
    # ============================================================
    # IMPLEMENTACIJA ABSTRACT METODA
    # ============================================================
    
    def _call_llm(self, system_role: str, user_prompt: str) -> str:
        """
        Poziva Gemini API.
        
        Gemini nema direktan "system role" koncept kao OpenAI,
        pa kombinujemo system role + user prompt u jedan tekst.
        """
        combined_prompt = f"{system_role}\n\n---\n\n{user_prompt}"
        
        try:
            response = self._model.generate_content(combined_prompt)
            
            # Brojanje tokena (priblizno)
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                tokens = (
                    response.usage_metadata.prompt_token_count
                    + response.usage_metadata.candidates_token_count
                )
                self._total_tokens += tokens
                self.logger.debug(f"Tokeni potroseni: {tokens}")
            
            return response.text
        
        except Exception as e:
            self.logger.error(f"Gemini API greska: {e}")
            raise
    
    def _parse_response(
        self,
        raw_response: str,
        request: GenerationRequest,
    ) -> list[dict[str, Any]]:
        """
        Parsira Gemini odgovor.
        
        Gemini bi trebao da vrati JSON niz (jer smo postavili 
        response_mime_type: application/json), ali ponekad ipak 
        umota u code block ili doda objasnjenje. Robusno parsiranje.
        """
        # Pokusaj 1: Direktan JSON parse
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # Mozda je vratio dict sa "samples" ili "data" kljucem
                for key in ("samples", "data", "items", "examples"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                # Mozda je dict koji je zapravo jedan primer
                return [parsed]
        except json.JSONDecodeError:
            pass
        
        # Pokusaj 2: Izvuci JSON iz markdown code block-a
        code_block_pattern = r"```(?:json)?\s*([\s\S]+?)\s*```"
        match = re.search(code_block_pattern, raw_response)
        if match:
            try:
                parsed = json.loads(match.group(1))
                if isinstance(parsed, list):
                    return parsed
                if isinstance(parsed, dict):
                    return [parsed]
            except json.JSONDecodeError:
                pass
        
        # Pokusaj 3: Pronadji prvi JSON niz u tekstu
        array_pattern = r"\[[\s\S]+?\]"
        match = re.search(array_pattern, raw_response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Sve je propalo
        self.logger.error(
            f"Ne mogu da parsiram odgovor. Prvih 500 karaktera:\n"
            f"{raw_response[:500]}"
        )
        raise ValueError(
            f"Gemini odgovor nije validan JSON. Request ID: {request.request_id}"
        )


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Ucitaj .env iz root foldera
    root_dir = Path(__file__).resolve().parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY nije postavljen u .env")
        exit(1)
    
    print("Testiranje GeminiGenerator-a...")
    
    # Kreiraj generator
    generator = GeminiGenerator(api_key=api_key)
    
    print(f"\nGenerator info:")
    print(f"  Name: {generator.generator_name}")
    print(f"  Version: {generator.generator_version}")
    print(f"  Model: {generator.model_name}")
    print(f"  Temperature: {generator.temperature}")
    
    # Pokusaj jedan jednostavan poziv
    print("\nPozivam Gemini sa test promptom...")
    
    try:
        response = generator._call_llm(
            system_role="Ti si AI asistent koji odgovara na srpskom.",
            user_prompt="Generisi JSON niz sa 2 primera pozdrava na srpskom. Format: [{\"text\": \"...\"}, ...]"
        )
        
        print(f"\nSiroviodgovor (prvih 300 karaktera):")
        print(response[:300])
        
        # Test parsiranja
        from .base_generator import GenerationRequest
        from .prompt_templates import get_template
        
        fake_request = GenerationRequest(
            category="safe_personal",
            template=get_template("safe_personal"),
            n_samples=2,
        )
        
        parsed = generator._parse_response(response, fake_request)
        print(f"\nParsirano: {len(parsed)} primera")
        for i, sample in enumerate(parsed[:3]):
            print(f"  [{i+1}] {sample}")
        
        print("\nGeminiGenerator test PROSAO.")
    
    except Exception as e:
        print(f"\nGreska: {e}")
        raise
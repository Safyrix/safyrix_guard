"""
Safyrix Data Generation - Groq Generator
=========================================

Konkretna implementacija BaseGenerator-a koja koristi Groq API
za generisanje primera koristeci Llama 3.3 70B.

Inheritance: BaseGenerator
Pattern: Strategy (alternativni LLM provajder)

Prednosti Groq-a:
- Bolji kvalitet srpskog (diakritike)
- Brze generisanje (sub-sekundu)
- Velikodusni free tier (30 RPM, 14400 RPD)
"""

import json
import re
from typing import Any

from groq import Groq

from .base_generator import BaseGenerator, GenerationRequest


class GroqGenerator(BaseGenerator):
    """
    Generator koji koristi Groq API sa Llama modelima.
    
    Konfiguracija (kroz **kwargs):
        - model_name: str (default: "llama-3.3-70b-versatile")
        - temperature: float (default: 0.9)
        - max_tokens: int (default: 8000)
    """
    
    generator_name = "groq"
    generator_version = "1.0.0"
    
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    DEFAULT_TEMPERATURE = 0.9
    DEFAULT_MAX_TOKENS = 8000
    
    def __init__(self, api_key: str, **kwargs: Any) -> None:
        super().__init__(api_key, **kwargs)
        
        self.model_name = kwargs.get("model_name", self.DEFAULT_MODEL)
        self.temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        self.max_tokens = kwargs.get("max_tokens", self.DEFAULT_MAX_TOKENS)
        
        self.client = Groq(api_key=self.api_key)
        
        self.logger.info(
            f"GroqGenerator inicijalizovan: model={self.model_name}, "
            f"temperature={self.temperature}"
        )
    
    # ============================================================
    # IMPLEMENTACIJA ABSTRACT METODA
    # ============================================================
    
    def _call_llm(self, system_role: str, user_prompt: str) -> str:
        """
        Poziva Groq Chat Completion API.
        
        Razlika u odnosu na Gemini:
        - Groq ima native podrsku za system/user role
        - Ne treba kombinovati u jedan prompt
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_role},
                    {"role": "user", "content": user_prompt + 
                     "\n\nVAZNO: Vrati ISKLJUCIVO JSON niz, bez markdown wrapping-a, "
                     "bez dodatnih objasnjenja, samo cisto JSON."},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
            # Brojanje tokena
            if response.usage:
                tokens = response.usage.total_tokens
                self._total_tokens += tokens
                self.logger.debug(f"Tokeni potroseni: {tokens}")
            
            return response.choices[0].message.content
        
        except Exception as e:
            self.logger.error(f"Groq API greska: {e}")
            raise
    
    def _parse_response(
        self,
        raw_response: str,
        request: GenerationRequest,
    ) -> list[dict[str, Any]]:
        """
        Parsira Groq odgovor.
        
        Llama modeli ponekad dodaju komentare ili wrapuju u markdown,
        pa robusno parsiranje.
        """
        # Pokusaj 1: Direktan JSON parse
        try:
            parsed = json.loads(raw_response)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                # Mozda je dict sa "samples" ili "data"
                for key in ("samples", "data", "items", "examples", "results"):
                    if key in parsed and isinstance(parsed[key], list):
                        return parsed[key]
                return [parsed]
        except json.JSONDecodeError:
            pass
        
        # Pokusaj 2: Markdown code block
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
        
        # Pokusaj 3: Pronadji prvi JSON niz
        # Llama ponekad doda preambulu, pa moramo da nadjemo gde JSON pocinje
        array_pattern = r"\[[\s\S]+?\]"
        match = re.search(array_pattern, raw_response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        
        # Pokusaj 4: Llama sometimes wraps in {"data": [...]} bez izricitog kljuca
        # Pokusaj sa nadjenim {} blokom
        object_pattern = r"\{[\s\S]+\}"
        match = re.search(object_pattern, raw_response)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    for key in ("samples", "data", "items", "examples", "results"):
                        if key in parsed and isinstance(parsed[key], list):
                            return parsed[key]
                    return [parsed]
            except json.JSONDecodeError:
                pass
        
        # Sve je propalo
        self.logger.error(
            f"Ne mogu da parsiram Groq odgovor. Prvih 500 karaktera:\n"
            f"{raw_response[:500]}"
        )
        raise ValueError(
            f"Groq odgovor nije validan JSON. Request ID: {request.request_id}"
        )


# ============================================================
# SELF-TEST
# ============================================================

if __name__ == "__main__":
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    root_dir = Path(__file__).resolve().parent.parent.parent
    load_dotenv(root_dir / ".env")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY nije postavljen u .env")
        exit(1)
    
    print("Testiranje GroqGenerator-a...\n")
    
    generator = GroqGenerator(api_key=api_key)
    
    print(f"Generator info:")
    print(f"  Name: {generator.generator_name}")
    print(f"  Version: {generator.generator_version}")
    print(f"  Model: {generator.model_name}")
    print(f"  Temperature: {generator.temperature}")
    
    print("\nPozivam Groq sa test promptom...")
    
    try:
        response = generator._call_llm(
            system_role="Ti si AI koji odgovara iskljucivo na srpskom jeziku (latinica).",
            user_prompt='Generisi JSON niz sa 3 primera pozdrava na srpskom. '
                       'Format: [{"text": "..."}, {"text": "..."}, {"text": "..."}]'
        )
        
        print(f"\nSirovi odgovor (prvih 300 karaktera):")
        print(response[:300])
        
        # Test parsiranja
        from .base_generator import GenerationRequest
        from .prompt_templates import get_template
        
        fake_request = GenerationRequest(
            category="safe_personal",
            template=get_template("safe_personal"),
            n_samples=3,
        )
        
        parsed = generator._parse_response(response, fake_request)
        print(f"\nParsirano: {len(parsed)} primera")
        for i, sample in enumerate(parsed[:3]):
            print(f"  [{i+1}] {sample}")
        
        print("\nGroqGenerator test PROSAO.")
    
    except Exception as e:
        print(f"\nGreska: {e}")
        raise
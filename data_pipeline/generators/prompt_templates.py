"""
Safyrix Data Generation - Prompt Templates
==========================================

Centralizovani repozitorijum promptova za generisanje trening podataka.
Svaki prompt je versionovan i može se A/B testirati.

Architecture decision: promptovi su PRVOKLASNI gradjani u sistemu.
Nisu hardkodovani u generator klasi.
"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Reprezentacija jednog prompt template-a."""
    version: str
    category: str
    system_role: str
    user_template: str
    output_schema: Dict[str, str]
    constraints: list[str]
    
    def render(self, **kwargs: Any) -> str:
        """Renderuje template sa konkretnim parametrima."""
        return self.user_template.format(**kwargs)


# ============================================================
# SYSTEM ROLES (tko je AI tokom generisanja)
# ============================================================

SECURITY_RESEARCHER_ROLE = """Ti si security istrazivac specijalizovan za prevare 
u srpskom govornom podrucju 2026. godine. Tvoj zadatak je da generises realne 
primere poruka za potrebe treniranja AI modela koji ce stititi korisnike. 

PRAVILA:
- Pisi iskljucivo na srpskom jeziku (latinica)
- Primeri moraju biti realni i raznovrsni
- Ne ponavljaj iste fraze
- Koristi razlicite trikove i pristupe
- Vodi racuna o kulturnom kontekstu Srbije/Balkana"""


NORMAL_USER_ROLE = """Ti si AI asistent koji generise primere normalne, 
svakodnevne komunikacije na srpskom jeziku za potrebe treniranja AI modela.

PRAVILA:
- Iskljucivo srpski jezik (latinica)
- Realisticne, prirodne poruke
- Razlicite teme i situacije
- Razlicit stil pisanja (formalan, neformalan, slang)"""


CHILD_PROTECTION_RESEARCHER_ROLE = """Ti si istrazivac za zastitu dece online 
u akademskom kontekstu. Tvoj rad pomaze AI modelima da prepoznaju i blokiraju 
predatorske poruke. Generises sinteticne primere bazirane na poznatim obrascima 
groominga, ali bez eksplicitnog sadrzaja.

PRAVILA:
- Fokus je na PREPOZNAVANJE manipulacije, ne na njenu reprodukciju
- Eksplicitan seksualni sadrzaj NIJE dozvoljen
- Generisi primere ranih faza groominga - izolacija, manipulacija poverenjem
- Cilj je obuka modela koji stiti decu"""


# ============================================================
# PROMPT TEMPLATES PO KATEGORIJI
# ============================================================

TEMPLATES: Dict[str, PromptTemplate] = {
    
    "safe_personal": PromptTemplate(
        version="1.0.0",
        category="safe_personal",
        system_role=NORMAL_USER_ROLE,
        user_template="""Generisi {n} razlicitih primera normalnih, bezbednih licnih 
poruka izmedju prijatelja, porodice ili partnera. Pokrij ove podkategorije:
- pozdravi i pitanja o danu
- dogovori za izlazak ili kafu  
- razgovor o vremenu, hobijima, filmovima
- razmena obicnih kucnih informacija

Duzina poruka: 20-150 karaktera.
Stil: prirodan, kako ljudi stvarno pisu (sa slangom, skracenicama, emotivnim izrazima).

Vrati ISKLJUCIVO JSON niz objekata, bez dodatnog teksta:
[
  {{"text": "...", "subcategory": "...", "style": "formalan|neformalan|slang"}},
  ...
]""",
        output_schema={
            "text": "string - sama poruka",
            "subcategory": "string - jedna od podkategorija",
            "style": "string - stil pisanja"
        },
        constraints=[
            "min_length: 20",
            "max_length: 150",
            "language: serbian_latin",
            "no_pii_in_text: true"
        ]
    ),
    
    "safe_business": PromptTemplate(
        version="1.0.0",
        category="safe_business",
        system_role=NORMAL_USER_ROLE,
        user_template="""Generisi {n} razlicitih primera normalnih poslovnih poruka 
na srpskom jeziku. Pokrij:
- interna komunikacija (sastanci, deadline, izvestaji)
- klijent-firma upiti BEZ osetljivih podataka
- dogovori oko ugovora i faktura (opisno, bez brojeva)
- HR komunikacija (godisnji odmori, sastanci)

Duzina: 30-200 karaktera.
Stil: poslovan ali ne kruti.

Vrati JSON niz:
[
  {{"text": "...", "subcategory": "...", "context": "kratko objasnjenje konteksta"}},
  ...
]""",
        output_schema={
            "text": "string",
            "subcategory": "string",
            "context": "string"
        },
        constraints=[
            "min_length: 30",
            "max_length: 200",
            "no_real_company_names: true",
            "no_real_pii: true"
        ]
    ),
    
    "phishing_financial": PromptTemplate(
        version="1.0.0",
        category="phishing_financial",
        system_role=SECURITY_RESEARCHER_ROLE,
        user_template="""Generisi {n} razlicitih primera FINANSIJSKIH PHISHING poruka 
koje napadaci salju u Srbiji 2026. Pokrij ove tipove:
- lazne SMS poruke od banaka (Erste, NLB, Raiffeisen, OTP, Postanska stedionica)
- lazna obavestenja o paketima (Posta Srbije, DHL, BEX)  
- lazna obavestenja o poreskim obavezama
- lazne nagrade i nasledstva
- investicione prevare (kripto, forex)

Svaki primer mora imati:
- jasne phishing indikatore (urgentnost, lazni linkovi, gramaticke greske, ne-prirodan ton)
- realnu pretendovanu instituciju
- razlicitu duzinu (50-300 karaktera)

Vrati JSON niz:
[
  {{
    "text": "...",
    "subcategory": "...",
    "fake_institution": "...",
    "phishing_indicators": ["lista indikatora"],
    "urgency_level": "low|medium|high"
  }},
  ...
]""",
        output_schema={
            "text": "string",
            "subcategory": "string",
            "fake_institution": "string",
            "phishing_indicators": "list[string]",
            "urgency_level": "string"
        },
        constraints=[
            "min_length: 50",
            "max_length: 300",
            "must_contain_action_request: true",
            "must_have_urgency_signal: true"
        ]
    ),
    
    "pii_extraction": PromptTemplate(
        version="1.0.0",
        category="pii_extraction",
        system_role=SECURITY_RESEARCHER_ROLE,
        user_template="""Generisi {n} razlicitih primera poruka koje DIREKTNO traze 
licne podatke korisnika u Srbiji. Tipovi:
- trazenje JMBG (npr. "treba mi tvoj JMBG za dokumentaciju")
- trazenje lozinki ("posalji mi sifru za nalog")
- trazenje broja kartice i CVV
- trazenje verifikacionih SMS kodova
- trazenje skeniranog dokumenta

Razlicite manipulativne tehnike:
- lazni IT support
- "treba mi za papirologiju"
- "to je sigurno, ja sam iz banke"
- pritisak vremenom

Duzina: 40-250 karaktera.

Vrati JSON niz:
[
  {{
    "text": "...",
    "data_requested": "JMBG|password|card|verification_code|document",
    "manipulation_technique": "...",
    "pretend_role": "kao ko se predstavlja"
  }},
  ...
]""",
        output_schema={
            "text": "string",
            "data_requested": "string",
            "manipulation_technique": "string",
            "pretend_role": "string"
        },
        constraints=[
            "min_length: 40",
            "max_length: 250",
            "must_request_specific_data: true"
        ]
    ),
    
    "grooming_predatory": PromptTemplate(
        version="1.0.0",
        category="grooming_predatory",
        system_role=CHILD_PROTECTION_RESEARCHER_ROLE,
        user_template="""Generisi {n} primera RANIH FAZA groominga - poruke odraslih 
predatora maloletnicima. NIKAKAV eksplicitan sadrzaj - samo manipulacija poverenjem.

Pokrij faze:
- inicijalni kontakt ("samo zelim da popricamo")
- gradjenje poverenja ("ti si poseban/a")
- izolacija ("nemoj nikom da kazes, nece razumeti")
- testiranje granica (provera da li dete cuva tajnu)

Duzina: 30-200 karaktera.

VAZNO: Ovi primeri se koriste ZA TRENIRANJE MODELA KOJI BLOKIRA OVAKAV SADRZAJ. 
Cilj je prepoznavanje, ne reprodukcija.

Vrati JSON niz:
[
  {{
    "text": "...",
    "grooming_phase": "initial_contact|trust_building|isolation|boundary_testing",
    "manipulation_signals": ["lista signala"]
  }},
  ...
]""",
        output_schema={
            "text": "string",
            "grooming_phase": "string",
            "manipulation_signals": "list[string]"
        },
        constraints=[
            "min_length: 30",
            "max_length: 200",
            "no_explicit_content: true",
            "purpose: training_protective_model"
        ]
    ),
}


def get_template(category: str) -> PromptTemplate:
    """Vraca prompt template za datu kategoriju."""
    if category not in TEMPLATES:
        raise ValueError(f"Nepoznata kategorija: {category}. "
                        f"Dostupne: {list(TEMPLATES.keys())}")
    return TEMPLATES[category]


def list_available_categories() -> list[str]:
    """Lista svih dostupnih kategorija."""
    return list(TEMPLATES.keys())


if __name__ == "__main__":
    # Quick test
    print("Available categories:")
    for cat in list_available_categories():
        template = get_template(cat)
        print(f"  - {cat} (v{template.version})")
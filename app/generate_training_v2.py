from __future__ import annotations

import json
import random
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
OUT_PATH = BASE_DIR / "dataset" / "training_v2.jsonl"

random.seed(42)


N_HIGH = 650
N_MEDIUM = 650
N_LOW = 700


names = [
    "Marko", "Jovan", "Milan", "Nikola", "Ana", "Ivana", "Marija",
    "Stefan", "Luka", "Petar", "Jelena", "Sara", "Mina", "Ognjen",
]

cities = [
    "Beograd", "Novi Sad", "Nis", "Kragujevac", "Subotica",
]

streets = [
    "Bulevar kralja Aleksandra", "Knez Mihailova", "Cara Dusana",
    "Vojvode Stepe", "Jurija Gagarina", "Maksima Gorkog",
]


def random_jmbg() -> str:
    return "".join(str(random.randint(0, 9)) for _ in range(13))


def random_phone() -> str:
    return random.choice(["061", "062", "063", "064", "065", "069"]) + \
        "".join(str(random.randint(0, 9)) for _ in range(7))


def random_card() -> str:
    groups = ["".join(str(random.randint(0, 9)) for _ in range(4)) for _ in range(4)]
    return " ".join(groups)


def random_iban() -> str:
    return "RS35" + "".join(str(random.randint(0, 9)) for _ in range(18))


def random_address() -> str:
    return f"{random.choice(streets)} {random.randint(1, 200)}, {random.choice(cities)}"


def random_email() -> str:
    name = random.choice(names).lower()
    num = random.randint(1, 99)
    domain = random.choice(["gmail.com", "yahoo.com", "outlook.com"])
    return f"{name}{num}@{domain}"


def gen_high_samples() -> list[dict]:
    samples: list[dict] = []

    templates_jmbg = [
        "Moj JMBG je {jmbg}, proveri da li je tacan.",
        "Ovde je moj jmbg {jmbg}, nemoj da ga delis dalje.",
        "Treba mi da posaljem svoj jmbg {jmbg} banci.",
        "Upisi ovaj JMBG u sistem: {jmbg}.",
        "Daj mi potvrdu da li jmbg {jmbg} odgovara imenu.",
    ]

    templates_phone = [
        "Moj broj telefona je {phone}, dodaj me u kontakt.",
        "Evo ti moj broj {phone}, javi se na Viber.",
        "Upisi {phone} u CRM kao kontakt klijenta.",
        "Moze li dostava na ovaj broj {phone}?",
    ]

    templates_card = [
        "Broj kartice je {card}, proveri transakciju.",
        "Unesi ovaj broj kartice {card} u formu.",
        "Kartica istice 04/27 a broj je {card}.",
        "Za uplatu koristi karticu {card}.",
    ]

    templates_bank = [
        "Broj racuna je {iban}, prebaci mi novac danas.",
        "IBAN za uplatu je {iban}, posalji mi potvrdu.",
        "Firma treba da uplati na racun {iban}.",
    ]

    templates_address = [
        "Moja adresa je {addr}, upisi u bazu podataka.",
        "Posalji ugovor na {addr}, na ime {name}.",
        "Klijent zivi na adresi {addr}.",
    ]

    templates_email = [
        "Moj email je {email}, posalji mi izvestaj.",
        "Registruj me sa email adresom {email}.",
        "Nalog treba da se poveze sa {email}.",
    ]

    templates_password = [
        "Lozinka za nalog je Test123!, nemoj nikom da kazes.",
        "Upisi sifru sistema: Admin2024!.",
        "Sifra za wifi je SafyrixWifi22.",
     ]
    templates_implicit = [
        "Treba mi tvoja sifra za nalog, posalji mi je ovde.",
        "Daj mi svoju lozinku da proverim nesto.",
        "Posalji mi svoje podatke za logovanje na sistem.",
        "Moze li tvoja sifra za mejl, hitno je.",
        "Treba mi pristup tvom nalogu, posalji username i password.",
        "Unesi svoje kredencijale ovde da nastavimo.",
        "Daj mi svoj kod sa authenticator aplikacije.",
        "Treba mi verifikacioni kod koji si dobio na telefon.",
        "Prosledi mi one podatke sto koristis za banku.",
        "Moze li sifra od internet bankarstva, provericu ti nesto.",
        "Pošalji mi login podatke, necu nikom reci.",
        "Unesi ovde sifru svog mejla da proverimo postavke.",
        "Daj mi podatke kartice da odradim uplatu umesto tebe.",
        "Treba mi tvoj jedinstveni broj da te prijavim.",
        "Posalji mi sve svoje podatke da popunim formular.",
    ]

    ]

     all_templates = (
        templates_jmbg
        + templates_phone
        + templates_card
        + templates_bank
        + templates_address
        + templates_email
        + templates_password
        + templates_implicit
    )


    while len(samples) < N_HIGH:
        t = random.choice(all_templates)
        text = t.format(
            jmbg=random_jmbg(),
            phone=random_phone(),
            card=random_card(),
            iban=random_iban(),
            addr=random_address(),
            email=random_email(),
            name=random.choice(names),
        )
        samples.append({"text": text, "label": "high"})

    return samples


def gen_medium_samples() -> list[dict]:
    samples: list[dict] = []

    templates = [
        "Da li je bezbedno da saljem podatke o klijentu mejlom?",
        "Kako da prepoznam phishing poruku od banke?",
        "Dobio sam sumnjiv link, sta da radim?",
        "Klijent trazi da mu posaljem sken licne karte, da li smem?",
        "Sta ako je neko trazio da mu posaljem sifru naloga?",
        "Kako da enkriptujem fajl pre slanja?",
        "Kolega trazi pristup sistemu, ali nije prosao obuku, sta da radim?",
        "Da li je problem da saljem screenshot internih podataka?",
        "Da li smem da podelim ID ugovora sa klijentom?",
        "Otvorio sam mejl za koji sumnjam da je spam, sta sada?",
        "Da li je ok da koristim isti password na vise servisa?",
        "Sta ako otvorim prilog iz nepoznatog mejla?",
        "Kako da proverim da li je sajt pravi a ne phishing?",
        "Sta da radim ako sam se ulogovao sa javnog wifi-ja?",
        "Da li je bezbedno da saljem podatke preko Google Drive linka?",
        "Da li treba da prijavim sumnjivu aktivnost IT odeljenju?",
        "Kako da prijavim potencijalni incident u firmi?",
        "Primio sam mejl koji trazi uplatu hitno, deluje sumnjivo.",
        "Klijent trazi detaljan izvestaj, ali nisam siguran sta sme da se deli.",
        "Sta je minimum podataka koji smeju da se dele eksterno?",
    ]

    while len(samples) < N_MEDIUM:
        text = random.choice(templates)
        samples.append({"text": text, "label": "medium"})

    return samples


def gen_low_samples() -> list[dict]:
    samples: list[dict] = []

    templates = [
        "Kako si danas, sta ima novo?",
        "Koji film da gledam veceras?",
        "Kada je sledeci trening u teretani?",
        "Imas li preporuku za dobar restoran u gradu?",
        "Da li pada kisa danas?",
        "Sta mislis o novom telefonu?",
        "Koji protein preporucujes za pocetnike?",
        "Kako se kuva testenina al dente?",
        "Sta radis veceras, slobodan si?",
        "Kakvo je vreme kod tebe?",
        "Koji auto je bolji za gradsku voznju?",
        "Imas li neku dobru knjigu za preporuku?",
        "Kako da poboljsam bench press?",
        "Ko je pobedio u mecu sinoc?",
        "Da li si pogledao novi trejler za film?",
        "Kada je sledeci UFC event?",
        "Kakvu kameru preporucujes za teretanu?",
        "Gde da idem za vikend na kratko putovanje?",
        "Koji sat ide uz odelo?",
        "Koja je najbolja kafa u gradu?",
        "Da li bi isao na more ove godine?",
        "Sta mislis o novom modelu patika?",
        "Koliko puta nedeljno je optimalno trenirati?",
        "Kako da se bolje fokusiram na posao?",
        "Imas li ideju za poklon za rodjendan?",
    ]

    while len(samples) < N_LOW:
        text = random.choice(templates)
        samples.append({"text": text, "label": "low"})

    return samples


def main() -> None:
    all_samples: list[dict] = []
    all_samples.extend(gen_high_samples())
    all_samples.extend(gen_medium_samples())
    all_samples.extend(gen_low_samples())

    random.shuffle(all_samples)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for obj in all_samples:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(all_samples)} samples to {OUT_PATH}")


if __name__ == "__main__":
    main()

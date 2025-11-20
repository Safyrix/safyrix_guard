from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


# --------------- 1. Ručno definisan dataset (primeri na srpskom) ---------------

def build_dataset() -> pd.DataFrame:
    data: List[Dict[str, str]] = [

        # HIGH RISK – ucena, pretnje, lokacija, tajnost
        {"message": "Ako ne posaljes slike svima cu reci sta si mi slao.", "risk_level": "high"},
        {"message": "Ako mi ne das novac, objavicu tvoje slike na internetu.", "risk_level": "high"},
        {"message": "Nemoj da kazes roditeljima, ovo je nasa tajna, salji mi svoju adresu.", "risk_level": "high"},
        {"message": "Ako mi odmah ne posaljes pare, doci cu kod tebe kuci.", "risk_level": "high"},
        {"message": "Salji mi svoju lokaciju ili cu svima poslati tvoj profil.", "risk_level": "high"},
        {"message": "Ako im bilo sta kazes, bices u velikom problemu.", "risk_level": "high"},
        {"message": "Ne smes nikome da kazes da se dopisujemo, posebno ne roditeljima.", "risk_level": "high"},
        {"message": "Hajde da se nadjemo sami, ali nikome nemoj da kazes gde ides.", "risk_level": "high"},
        {"message": "Posalji mi sliku bez odece ili prekidam da pricam sa tobom.", "risk_level": "high"},
        {"message": "Ako ne posaljes jos slika, pokazacu ovima iz skole sta vec imam.", "risk_level": "high"},
        {"message": "Ne smes da blokiras ovaj nalog, pratim te i znam gde zivis.", "risk_level": "high"},
        {"message": "Moramo ovo da cuvamo u tajnosti, niko ne sme da zna sta radimo.", "risk_level": "high"},

        # MEDIUM RISK – pritisak, manipulacija, krivica
        {"message": "Ako me stvarno volis, poslaces mi jos neku svoju sliku.", "risk_level": "medium"},
        {"message": "Bas si dosadan, pa valjda mozes malo da popustis.", "risk_level": "medium"},
        {"message": "Nemoj da odustajes bas sad kad sam sve isplanirao.", "risk_level": "medium"},
        {"message": "Pa sta ako su roditelji strogi, ne moras im sve reci.", "risk_level": "medium"},
        {"message": "Nemoj da dramis, samo mi posalji broj telefona.", "risk_level": "medium"},
        {"message": "Ako izadjes sa mnom, necu vise da ti smaram porukama.", "risk_level": "medium"},
        {"message": "Zasto si toliko paranoican, samo je slika, opusti se.", "risk_level": "medium"},
        {"message": "Ako me odbijes, ispast cu kao idiot pred svima.", "risk_level": "medium"},
        {"message": "Stvarno si sebican ako neces malo da rizikujes zbog mene.", "risk_level": "medium"},

        # LOW RISK – normalna komunikacija
        {"message": "Hoces sutra u skolu da dodjes ranije da ucimo zajedno?", "risk_level": "low"},
        {"message": "Jesi stigao kuci, javi kad budes slobodan za igricu.", "risk_level": "low"},
        {"message": "Kako ti je prosao kontrolni iz matematike danas?", "risk_level": "low"},
        {"message": "Da li ides na trening veceras ili preskaces?", "risk_level": "low"},
        {"message": "Posalji mi link do onog videa sto si pominjao.", "risk_level": "low"},
        {"message": "Mozemo da igramo kasnije ako zavrsim domaci na vreme.", "risk_level": "low"},
        {"message": "Sta planiras za vikend, ides li kod bake i deke?", "risk_level": "low"},
        {"message": "Hvala ti za pomoc oko zadace, mnogo je znacilo.", "risk_level": "low"},
        {"message": "Poslacu ti slike sa rodjendana u grupu kasnije.", "risk_level": "low"},
    ]

    return pd.DataFrame(data)


# --------------- 2. Trening modela ---------------

def train_and_save():
    df = build_dataset()

    X = df["message"]
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipeline: Pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("=== REPORT ===")
    print(classification_report(y_test, y_pred))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "guardian_text_model.pkl"
    joblib.dump(pipeline, model_path)

    print(f"[ML] Model sacuvan u {model_path.resolve()}")


if __name__ == "__main__":
    train_and_save()

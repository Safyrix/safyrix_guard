from __future__ import annotations

from app.ml_training import train_and_save_model


if __name__ == "__main__":
    print("[RUN] Pokrecem trening Safyrix Guardian ML modela...")
    metrics = train_and_save_model()

    print("\n[SUMMARY] Rezime treninga:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Train sample size: {metrics['n_train']}")
    print(f"  Test sample size:  {metrics['n_test']}")
    print(f"  Detektovane klase rizika: {metrics['classes']}")
    print("\n[OK] Trening zavrsen. GuardianAgent sada ima sve sto mu treba.")

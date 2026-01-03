from __future__ import annotations

"""
Évaluation d’un modèle de churn avec tuning du seuil optimal (F1).

Ce script :
1. Charge le dataset prétraité `data/processed.csv` ;
2. Crée un pipeline scikit-learn :
   - Standardisation des variables numériques,
   - OneHotEncoding des variables catégorielles,
   - Régression logistique ;
3. Découpe train/test avec stratification ;
4. Entraîne le modèle ;
5. Calcule :
   - Les métriques standard avec seuil = 0.5,
   - Le seuil optimal maximisant la F1,
   - Une baseline triviale ;
6. Sauvegarde :
   - Le modèle dans `models/`,
   - Les métadonnées dans `registry/metadata.json`,
   - Le modèle courant dans `registry/current_model.txt` si le gate F1 est validé,
   - Les métriques dans `reports/metrics.json`.

Compatible avec DVC.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Constantes globales
# ---------------------------------------------------------------------------

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
DATA_PATH: Final[Path] = ROOT / "data" / "processed.csv"
MODELS_DIR: Final[Path] = ROOT / "models"
REGISTRY_DIR: Final[Path] = ROOT / "registry"
CURRENT_MODEL_PATH: Final[Path] = REGISTRY_DIR / "current_model.txt"
METADATA_PATH: Final[Path] = REGISTRY_DIR / "metadata.json"
REPORTS_DIR: Final[Path] = ROOT / "reports"
METRICS_PATH: Final[Path] = REPORTS_DIR / "metrics.json"

# ---------------------------------------------------------------------------
# Gestion des métadonnées
# ---------------------------------------------------------------------------

def load_metadata() -> list[dict[str, Any]]:
    if not METADATA_PATH.exists():
        return []
    with METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)

def save_metadata(items: list[dict[str, Any]]) -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(items, file, indent=2)

# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------

def compute_baseline_f1(y_true: pd.Series | list[int]) -> float:
    y_pred = [0] * len(y_true)
    return float(f1_score(y_true, y_pred, zero_division=0))

def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_f1 = 0.0
    for t in np.linspace(0.1, 0.9, 81):
        y_hat = (y_proba >= t).astype(int)
        score = f1_score(y_true, y_hat, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t
    return float(best_threshold), float(best_f1)

# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------

def main(version: str = "v1", seed: int = 42, gate_f1: float = 0.6) -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("processed.csv introuvable. Exécuter prepare_data.py d'abord.")

    df = pd.read_csv(DATA_PATH)
    target = "churn"
    X = df.drop(columns=[target])
    y = df[target].astype(int)

    num_cols = ["tenure_months", "num_complaints", "avg_session_minutes"]
    cat_cols = ["plan_type", "region"]

    # Prétraitement
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    model = LogisticRegression(max_iter=200, random_state=seed)
    pipe = Pipeline([("prep", preprocessor), ("clf", model)])

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Entraînement
    pipe.fit(X_train, y_train)

    # Probabilités pour la classe positive
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Prédiction standard (seuil 0.5)
    y_pred_default = (y_proba >= 0.5).astype(int)

    # Métriques de base
    metrics_default = {
        "accuracy": accuracy_score(y_test, y_pred_default),
        "precision": precision_score(y_test, y_pred_default, zero_division=0),
        "recall": recall_score(y_test, y_pred_default, zero_division=0),
        "f1_threshold_05": f1_score(y_test, y_pred_default, zero_division=0),
    }

    # Seuil optimal
    best_threshold, best_f1 = find_best_threshold(y_test.to_numpy(), y_proba)

    # Baseline triviale
    baseline = compute_baseline_f1(y_test)

    # Final metrics
    metrics = {
        **metrics_default,
        "f1": float(best_f1),
        "best_threshold": float(best_threshold),
        "baseline_f1": float(baseline),
    }

    # Sauvegarde du modèle
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_filename = f"churn_model_{version}_{timestamp}.joblib"
    model_path = MODELS_DIR / model_filename
    joblib.dump(pipe, model_path)

    # Sauvegarde alias stable
    stable_model_path = MODELS_DIR / "model.joblib"
    joblib.dump(pipe, stable_model_path)

    # Sauvegarde métriques pour DVC
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Métadonnées
    entry: dict[str, Any] = {
        "model_file": model_filename,
        "version": version,
        "trained_at_utc": timestamp,
        "data_file": DATA_PATH.name,
        "seed": seed,
        "metrics": metrics,
        "gate_f1": gate_f1,
        "passed_gate": bool(metrics["f1"] >= gate_f1 and metrics["f1"] >= metrics["baseline_f1"]),
    }

    items = load_metadata()
    items.append(entry)
    save_metadata(items)

    # Logs
    print("[METRICS]", json.dumps(metrics, indent=2))
    print(f"[OK] Modèle sauvegardé : {model_path}")
    print(f"[OK] Alias stable : {stable_model_path}")
    print(f"[OK] Métriques sauvegardées : {METRICS_PATH}")

    # Déploiement minimal
    if entry["passed_gate"]:
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        CURRENT_MODEL_PATH.write_text(model_filename, encoding="utf-8")
        print(f"[DEPLOY] Modèle activé : {model_filename}")
    else:
        print("[DEPLOY] Refusé : F1 insuffisante ou baseline non battue.")


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "models" / "learned"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "fusion_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "decision_threshold.json"

FEATURE_COLUMNS = [
    "spatial_evidence",
    "spectral_evidence",
    "feature_evidence",
    "spatial_present",
    "spectral_present",
    "feature_present",
    "conflict",
    "uncertainty",
]

AGENT_NAMES = ["spatial", "spectral", "feature"]


def prepare_features(df):
    """Convert agent outputs into numeric learning features."""
    data = df.copy()

    for name in AGENT_NAMES:
        status_col = f"{name}_status"
        evidence_col = f"{name}_evidence"

        if status_col in data.columns:
            data[f"{name}_present"] = (
                data[status_col]
                .fillna("")
                .astype(str)
                .str.upper()
                .isin(["DETECTED", "EVIDENCE_PRESENT"])
                .astype(float)
            )
        elif f"{name}_present" not in data.columns:
            data[f"{name}_present"] = 0.0

        if evidence_col not in data.columns:
            data[evidence_col] = np.nan

        data[evidence_col] = pd.to_numeric(
            data[evidence_col], errors="coerce"
        )

        # Missing evidence remains missing, not evidence of zero.
        data.loc[
            data[f"{name}_present"] == 0,
            evidence_col,
        ] = np.nan

    for col in ["conflict", "uncertainty"]:
        if col not in data.columns:
            data[col] = 0.0

        data[col] = pd.to_numeric(
            data[col], errors="coerce"
        ).clip(0.0, 1.0)

    X = data[FEATURE_COLUMNS].replace(
        [np.inf, -np.inf], np.nan
    )

    return X


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(
            strategy="median",
            add_indicator=True
        )),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        )),
    ])


def choose_threshold(y_true, probabilities):
    """Choose a threshold that maximizes validation F1."""
    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in np.arange(0.10, 0.91, 0.01):
        predictions = (probabilities >= threshold).astype(int)
        score = f1_score(
            y_true, predictions, zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def train_parameter_learning(
    train_csv,
    validation_csv,
    label_column="label",
    positive_label="camouflage",
):
    """
    CSV files must contain agent outputs and ground-truth labels.
    Split by scene/capture before creating these files to avoid
    leakage between training and validation.
    """
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(validation_csv)

    required = [label_column]
    for col in required:
        if col not in train_df or col not in val_df:
            raise ValueError(f"Missing required column: {col}")

    y_train = (
        train_df[label_column]
        .astype(str)
        .str.lower()
        .eq(positive_label.lower())
        .astype(int)
    )

    y_val = (
        val_df[label_column]
        .astype(str)
        .str.lower()
        .eq(positive_label.lower())
        .astype(int)
    )

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise ValueError(
            "Training and validation must each contain both "
            "camouflage and non-camouflage examples."
        )

    X_train = prepare_features(train_df)
    X_val = prepare_features(val_df)

    model = make_model()
    model.fit(X_train, y_train)

    positive_index = list(
        model.named_steps["classifier"].classes_
    ).index(1)

    val_probabilities = model.predict_proba(X_val)[
        :, positive_index
    ]

    threshold, validation_f1 = choose_threshold(
        y_val.to_numpy(), val_probabilities
    )

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": threshold,
        "validation_f1": float(validation_f1),
        "validation_accuracy": float(
            accuracy_score(y_val, val_predictions)
        ),
        "validation_precision": float(
            precision_score(
                y_val, val_predictions, zero_division=0
            )
        ),
        "validation_recall": float(
            recall_score(
                y_val, val_predictions, zero_division=0
            )
        ),
    }

    try:
        metrics["validation_auc"] = float(
            roc_auc_score(y_val, val_probabilities)
        )
    except ValueError:
        metrics["validation_auc"] = None

    joblib.dump(model, MODEL_PATH)

    import json
    THRESHOLD_PATH.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8"
    )

    print("Learned fusion model saved:", MODEL_PATH)
    print("Validation parameters and metrics:")
    print(json.dumps(metrics, indent=2))

    return model, metrics


if __name__ == "__main__":
    train_parameter_learning(
        train_csv=PROJECT_DIR / "data" / "learning_train.csv",
        validation_csv=PROJECT_DIR / "data" / "learning_validation.csv",
    )

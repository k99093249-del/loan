"""
Train and compare loan prediction models; save the best sklearn Pipeline (preprocess + classifier).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
CAT_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Employment_Status",
    "Property_Area",
]
TARGET = "Loan_Status"


def build_preprocess() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", build_preprocess()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, class_weight="balanced"),
                ),
            ]
        ),
        "decision_tree": Pipeline(
            [
                ("preprocess", build_preprocess()),
                ("model", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", build_preprocess()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
    }


def evaluate(name: str, pipe: Pipeline, X_test, y_test) -> float:
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", round(acc, 4))
    print(
        "Precision (Y):",
        round(precision_score(y_test, y_pred, pos_label="Y", zero_division=0), 4),
    )
    print(
        "Recall (Y):",
        round(recall_score(y_test, y_pred, pos_label="Y", zero_division=0), 4),
    )
    print("Confusion matrix [rows=true Y/N, cols=pred Y/N]:")
    labels = sorted(y_test.unique().tolist())
    print(confusion_matrix(y_test, y_pred, labels=labels))
    print(classification_report(y_test, y_pred, zero_division=0))
    return float(acc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to training CSV")
    ap.add_argument("--out", required=True, help="Output pickle path")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    for c in NUMERIC_COLS + CAT_COLS + [TARGET]:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    X = df[NUMERIC_COLS + CAT_COLS]
    y = df[TARGET]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    best_name = None
    best_acc = -1.0
    best_pipe = None
    for name, pipe in build_models().items():
        pipe.fit(X_train, y_train)
        acc = evaluate(name, pipe, X_test, y_test)
        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_pipe = pipe

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(best_pipe, f)

    print(f"\nSaved best pipeline ({best_name}, acc={best_acc:.4f}) -> {out_path}")


if __name__ == "__main__":
    main()

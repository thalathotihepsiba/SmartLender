"""
train_model.py
----------------
Smart Lender - Model Training Pipeline

Loads the loan applicant dataset, cleans/preprocesses it, trains four
classification models (Decision Tree, Random Forest, KNN, XGBoost),
compares their performance, and persists the best-performing model
(along with the fitted encoders/scaler) for use by the Flask app.
"""

import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# XGBoost is preferred, but fall back to GradientBoosting if the
# xgboost package isn't installed in the current environment.
try:
    from xgboost import XGBClassifier

    def make_xgb():
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    XGB_LABEL = "XGBoost"
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier

    def make_xgb():
        return GradientBoostingClassifier(
            n_estimators=120,
            max_depth=2,
            learning_rate=0.08,
            subsample=0.85,
            random_state=42,
        )

    XGB_LABEL = "XGBoost (GradientBoosting fallback - install xgboost for the real thing)"


RANDOM_STATE = 42

CATEGORICAL_COLS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_COLS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS
TARGET_COL = "Loan_Status"


def load_and_clean(path="data/loan_data.csv"):
    df = pd.read_csv(path)
    df = df.drop(columns=["Loan_ID"])

    # Impute categorical columns with the mode
    for col in CATEGORICAL_COLS:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Impute numeric columns with the median
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(df[col].median())

    return df


def encode_features(df):
    """Label-encode categorical columns; returns df + fitted encoders."""
    encoders = {}
    df = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    target_le = LabelEncoder()
    df[TARGET_COL] = target_le.fit_transform(df[TARGET_COL])  # N=0, Y=1
    encoders[TARGET_COL] = target_le

    return df, encoders


def main():
    print("=" * 60)
    print("Smart Lender - Training Pipeline")
    print("=" * 60)

    df = load_and_clean()
    df_encoded, encoders = encode_features(df)

    X = df_encoded[FEATURE_COLS]
    y = df_encoded[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.22, random_state=RANDOM_STATE, stratify=y
    )

    # Scale numeric features (helps KNN especially)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test_scaled[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=4, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=7, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=9),
        XGB_LABEL: make_xgb(),
    }

    results = {}
    fitted_models = {}

    for name, model in models.items():
        # Tree ensembles don't need scaling, but scaled input doesn't hurt them;
        # KNN benefits from it. We use scaled features consistently everywhere.
        model.fit(X_train_scaled, y_train)
        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)

        results[name] = {
            "train_accuracy": round(train_acc * 100, 2),
            "test_accuracy": round(test_acc * 100, 2),
        }
        fitted_models[name] = model

        print(f"\n{name}")
        print(f"  Train Accuracy: {train_acc * 100:.2f}%")
        print(f"  Test Accuracy:  {test_acc * 100:.2f}%")

    best_name = max(results, key=lambda k: results[k]["test_accuracy"])
    best_model = fitted_models[best_name]

    print("\n" + "=" * 60)
    print(f"Best model: {best_name} (Test Accuracy: {results[best_name]['test_accuracy']}%)")
    print("=" * 60)

    # Persist artifacts for the Flask app
    joblib.dump(best_model, "model/model.pkl")
    joblib.dump(encoders, "model/encoders.pkl")
    joblib.dump(scaler, "model/scaler.pkl")
    joblib.dump(FEATURE_COLS, "model/feature_cols.pkl")

    with open("model/model_comparison.json", "w") as f:
        json.dump(
            {
                "results": results,
                "best_model": best_name,
                "feature_columns": FEATURE_COLS,
            },
            f,
            indent=2,
        )

    print("\nSaved: model/model.pkl, model/encoders.pkl, model/scaler.pkl,")
    print("       model/feature_cols.pkl, model/model_comparison.json")


if __name__ == "__main__":
    main()

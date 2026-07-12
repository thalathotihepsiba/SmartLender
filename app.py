"""
app.py
-------
Smart Lender - Flask Web Application

Serves a form where a credit officer / applicant can enter loan
application details and get an instant creditworthiness prediction
from the trained model (Decision Tree / Random Forest / KNN / XGBoost
- whichever scored best during training).
"""

import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "encoders.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.pkl"))
    return model, encoders, scaler, feature_cols


MODEL, ENCODERS, SCALER, FEATURE_COLS = load_artifacts()

CATEGORICAL_COLS = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]
NUMERIC_COLS = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    form = request.form

    raw = {
        "Gender": form.get("gender"),
        "Married": form.get("married"),
        "Dependents": form.get("dependents"),
        "Education": form.get("education"),
        "Self_Employed": form.get("self_employed"),
        "ApplicantIncome": float(form.get("applicant_income", 0)),
        "CoapplicantIncome": float(form.get("coapplicant_income", 0)),
        "LoanAmount": float(form.get("loan_amount", 0)),
        "Loan_Amount_Term": float(form.get("loan_term", 360)),
        "Credit_History": float(form.get("credit_history", 1)),
        "Property_Area": form.get("property_area"),
    }

    df = pd.DataFrame([raw])

    # Encode categoricals using the fitted LabelEncoders, guarding against
    # unseen categories by falling back to the most common training class.
    for col in CATEGORICAL_COLS:
        le = ENCODERS[col]
        val = df.at[0, col]
        if val not in le.classes_:
            val = le.classes_[0]
        df[col] = le.transform([val])

    df[NUMERIC_COLS] = SCALER.transform(df[NUMERIC_COLS])

    X = df[FEATURE_COLS]
    pred = MODEL.predict(X)[0]

    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(np.max(MODEL.predict_proba(X)[0])) * 100

    target_le = ENCODERS["Loan_Status"]
    label = target_le.inverse_transform([pred])[0]

    # Gauge geometry: semicircle from (15,105) to (185,105), center (100,105), r=68.
    # confidence 0-100% maps to sweep angle t in [0, 180] degrees.
    conf_for_gauge = proba if proba is not None else 50.0
    t = (conf_for_gauge / 100.0) * 180.0
    theta_std = math.radians(180.0 - t)
    needle_x = 100 + 68 * math.cos(theta_std)
    needle_y = 105 - 68 * math.sin(theta_std)
    arc_length = (t / 180.0) * 267.0  # 267 ~= path length of the outer arc

    result = {
        "approved": label == "Y",
        "confidence": round(proba, 1) if proba is not None else None,
        "applicant": raw,
        "needle_x": round(needle_x, 1),
        "needle_y": round(needle_y, 1),
        "arc_length": round(arc_length, 1),
    }

    return render_template("result.html", result=result)


if __name__ == "__main__":
    # For local development. On IBM Cloud / production, use a WSGI server
    # (e.g. gunicorn) as described in the README.
    app.run(host="0.0.0.0", port=5000, debug=True)

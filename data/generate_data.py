"""
generate_data.py
-----------------
Generates a synthetic loan applicant dataset for the Smart Lender project.
The schema mirrors the well-known "Loan Prediction" dataset structure:
Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status

Run this once to create data/loan_data.csv, which train_model.py consumes.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 614  # matches the size of the classic loan-prediction training set

genders = np.random.choice(["Male", "Female"], size=N, p=[0.8, 0.2])
married = np.random.choice(["Yes", "No"], size=N, p=[0.65, 0.35])
dependents = np.random.choice(["0", "1", "2", "3+"], size=N, p=[0.55, 0.2, 0.15, 0.1])
education = np.random.choice(["Graduate", "Not Graduate"], size=N, p=[0.78, 0.22])
self_employed = np.random.choice(["Yes", "No"], size=N, p=[0.14, 0.86])
property_area = np.random.choice(["Urban", "Semiurban", "Rural"], size=N, p=[0.38, 0.38, 0.24])

applicant_income = np.random.gamma(shape=4.0, scale=1500, size=N).astype(int) + 1500
coapplicant_income = np.where(
    married == "Yes",
    np.random.gamma(shape=2.5, scale=900, size=N).astype(int),
    0,
)
loan_amount = (
    (applicant_income + coapplicant_income) / 1000 * np.random.uniform(2.0, 4.5, size=N)
).astype(int) + np.random.randint(-20, 20, size=N)
loan_amount = np.clip(loan_amount, 9, 700)

loan_term_choices = [360, 180, 300, 240, 120, 60, 84, 36, 12]
loan_term_probs = [0.73, 0.08, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.01]
loan_amount_term = np.random.choice(loan_term_choices, size=N, p=loan_term_probs).astype(float)

credit_history = np.random.choice([1.0, 0.0], size=N, p=[0.84, 0.16])

# Introduce some realistic missingness
for arr, frac in [
    (genders, 0.02),
    (married, 0.005),
    (dependents, 0.025),
    (self_employed, 0.05),
    (loan_amount_term, 0.023),
    (credit_history, 0.08),
]:
    idx = np.random.choice(N, size=int(N * frac), replace=False)
    if arr.dtype.kind in "fc":
        arr[idx] = np.nan
    else:
        arr[idx] = None

loan_amount = loan_amount.astype(float)
lm_missing_idx = np.random.choice(N, size=int(N * 0.035), replace=False)
loan_amount[lm_missing_idx] = np.nan

# ----- Determine Loan_Status with a signal-bearing rule + noise -----
# Higher total income, lower requested amount relative to income,
# good credit history, and graduate education all raise approval odds.
total_income = applicant_income + coapplicant_income
income_to_loan_ratio = total_income / (loan_amount.clip(min=1) * 1000 + 1)

score = (
    2.6 * np.nan_to_num(credit_history, nan=0.6)
    + 0.9 * (education == "Graduate").astype(float)
    + 0.5 * (property_area == "Semiurban").astype(float)
    + 1.1 * np.clip(income_to_loan_ratio, 0, 3)
    - 0.6 * (self_employed == "Yes").astype(float)
    + np.random.normal(0, 0.9, size=N)
)
score = np.nan_to_num(score, nan=np.nanmedian(score))
loan_status = np.where(score > np.nanmedian(score) - 0.15, "Y", "N")

df = pd.DataFrame(
    {
        "Loan_ID": [f"LP{str(i).zfill(6)}" for i in range(1, N + 1)],
        "Gender": genders,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
        "Loan_Status": loan_status,
    }
)

df.to_csv("data/loan_data.csv", index=False)
print(f"Generated data/loan_data.csv with {len(df)} rows")
print(df["Loan_Status"].value_counts())

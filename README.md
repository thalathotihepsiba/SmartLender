# Smart Lender

Machine learning-powered web application that predicts loan applicant creditworthiness, so credit officers and financial analysts can make faster, data-driven approval decisions.

The pipeline trains four classifiers — **Decision Tree**, **Random Forest**, **K-Nearest Neighbors**, and **XGBoost** — on applicant data (income, loan amount, credit history, employment status, etc.), automatically selects whichever model tests best, and serves it through a Flask web app for real-time predictions.

## Project structure

```
SmartLender/
├── app.py                          # Flask web application
├── train_model.py                  # Preprocessing + training + model selection pipeline
├── requirements.txt
├── data/
│   ├── generate_data.py            # Generates the synthetic applicant dataset
│   └── loan_data.csv               # Applicant data (Gender, Income, Credit_History, Loan_Status, ...)
├── model/                          # Created by train_model.py
│   ├── model.pkl                   # Best-performing trained classifier
│   ├── encoders.pkl                # Fitted LabelEncoders for categorical fields
│   ├── scaler.pkl                  # Fitted StandardScaler for numeric fields
│   ├── feature_cols.pkl            # Ordered feature list expected by the model
│   └── model_comparison.json       # Train/test accuracy for all four models
├── notebooks/
│   └── EDA_and_Model_Comparison.ipynb
├── templates/
│   ├── index.html                  # Application form
│   └── result.html                 # Prediction result page
└── static/css/style.css
```

## How it works

1. **`data/generate_data.py`** produces `data/loan_data.csv`, a dataset shaped like the classic loan-prediction schema: `Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area, Loan_Status`.
   - **Use your own data instead:** drop a CSV with the same column names in `data/loan_data.csv` and skip this step — everything downstream reads from that file.
2. **`train_model.py`**:
   - Imputes missing categorical values with the mode and numeric values with the median
   - Label-encodes categorical fields and scales numeric fields
   - Trains Decision Tree, Random Forest, KNN, and XGBoost
   - Prints a train/test accuracy comparison and saves the best model + preprocessing artifacts to `model/`
3. **`app.py`** loads those artifacts and serves:
   - `GET /` — the applicant intake form
   - `POST /predict` — encodes the submitted form, runs the saved model, and renders an approval/decline result with a confidence gauge

> **Note on accuracy numbers:** the project brief cites XGBoost at 94.7% train / 81.1% test accuracy. Exact figures depend on your real dataset — the included `data/loan_data.csv` is synthetic (for demonstration), so re-running `train_model.py` on it will show different, but directionally similar, numbers. Swap in a real historical loan dataset to reproduce production-grade metrics.
>
> **Note on XGBoost:** if the `xgboost` package isn't installed, `train_model.py` automatically falls back to scikit-learn's `GradientBoostingClassifier` so the pipeline still runs end-to-end. Install `xgboost` (included in `requirements.txt`) for the real thing.

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the dataset (skip if you're supplying your own data/loan_data.csv)
python data/generate_data.py

# 4. Train the models and save the best one
python train_model.py

# 5. Run the web app
python app.py
```

Then open **http://localhost:5000** in your browser.

## Retraining

Re-run `python train_model.py` any time `data/loan_data.csv` changes. It always overwrites `model/model.pkl` and the other artifacts with the current best performer — no code changes needed in `app.py`.

## Deploying to IBM Cloud

The app is a standard Flask/WSGI application, so it deploys cleanly to **IBM Cloud Foundry** or as a container on **IBM Cloud Code Engine**:

1. Add a `Procfile`:
   ```
   web: gunicorn app:app
   ```
2. Push with the Cloud Foundry CLI (`ibmcloud cf push`) or containerize with the included `requirements.txt` and deploy to Code Engine.
3. Make sure `model/*.pkl` are trained and committed (or trained as a build step) before the app starts, since `app.py` loads them at import time.

## Example scenarios

- **Fast-track approval:** salaried applicant, good credit history, stable income → high-confidence approval, eligible for fast-track processing.
- **High-risk detection:** self-employed applicant with irregular income and no credit history → flagged for manual review/document verification.
- **Batch evaluation:** analysts can evaluate many applicants quickly by scripting repeated calls to `POST /predict` (or extending `app.py` with a bulk-upload CSV endpoint).

## Tech stack

Python · Flask · NumPy · Pandas · scikit-learn · XGBoost · Matplotlib · Seaborn · SciPy

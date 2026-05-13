# Project report — Loan Prediction System

## Introduction

This project is an end-to-end **loan approval screening demo**: users sign up, submit application fields, receive an **Approved/Rejected** decision with an optional **PDF summary**, and view **personal analytics** on a dashboard.

## Problem statement

Financial institutions pre-screen loan applications to estimate repayment risk. This system automates a **binary decision** (approve vs reject) using supervised learning trained on historical application records, while adding **explicit student eligibility rules** for transparency.

## Dataset description

- File: `data/sample_loan_data.csv`
- Target: `Loan_Status` (`Y`/`N`)
- Features include income, loan amount/term, credit history flag, employment status (Student/Employed/Self Employed), demographics, education, and property area.

## Methodology

1. **Train/test split** with stratification when possible (`train_model.py`).
2. **Preprocessing** via `ColumnTransformer`:
  - Numeric median imputation + standardization
  - Categorical mode imputation + one-hot encoding (`handle_unknown="ignore"`)
3. **Models compared**: Logistic Regression, Decision Tree, Random Forest; **best by accuracy** is saved as `models/loan_pipeline.pkl`.
4. **Inference**: the saved `Pipeline` transforms inputs identically at prediction time.
5. **Student policy**: rule-based checks override the ML label for `Employment_Status=Student` when policy passes/fails.

## Tools used

- Python, pandas, scikit-learn, pickle
- Flask + SQLite + session auth
- ReportLab (PDF)
- HTML/CSS + Chart.js (dashboard)

## Results and interpretation

- **Accuracy / precision / recall** are printed during training; with small demo data, metrics are illustrative only.
- **Rejected** outputs include **high-level reasons** (not protected-attribute explanations).
- **Confusion matrix interpretation** (training output): rows represent true labels, columns predicted labels for `Y` vs `N`.

## Conclusion

The system demonstrates a reproducible ML pipeline with a simple web layer and per-user record keeping. Future improvements include hosted database storage, model monitoring, calibrated probabilities, and richer fairness checks.
# Test cases — Loan Prediction System

Run the app from **`C:\Users\Sanjai\new loan`** after installing deps and training the model.

## Valid inputs (happy path)

- **Employed applicant**: `Credit_History=1`, reasonable `LoanAmount` vs income, complete address/mobile/country/state.
- **Expected**: HTTP 200 on `POST /predict`, decision rendered, optional PDF download link.

## Student policy — valid

- `Employment_Status=Student`
- Upload a small PDF/JPG/PNG as college ID
- `Student_Part_Time_Job` ∈ {`Food Servent`, `Shop Keeping`, `Xerox Shop`, `Other`}
- `ApplicantIncome` between **3000** and **5000**
- **Expected**: final decision **Approved** (`Y`) with `decision_source=student_policy` when all checks pass.

## Edge cases

- Very high income + small loan: should generally lean **Approved** (model dependent).
- `CoapplicantIncome` large with low applicant income: affordability signal changes.
- `Loan_Amount_Term` very small (e.g., 12): may trigger a high-level “short term” reason if rejected.

## Invalid inputs

- Mobile not digits / wrong length: flash error / API 400.
- Missing address/country/state: validation error.
- Student missing file or invalid job or income outside 3000–5000: **Rejected** with student-specific reasons.

## API tests (`POST /api/predict`)

- Missing required keys → `400` with `error` message.
- Valid JSON → `200` with `prediction`, `prediction_id`, and `rejection_reasons` when rejected.

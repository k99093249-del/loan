# Deploy on Render (Flask)

Project root: `**C:\Users\Sanjai\new loan**` (build/train from your machine or CI before deploy, or add a build step).

## 1) Prepare repository

- Include `requirements.txt`, `app.py`, `templates/`, `static/`, `data/sample_loan_data.csv`.
- Commit a trained model `**models/loan_pipeline.pkl**` *or* run training in Render **build command** (see below).

## 2) Render service settings

- **Runtime**: Python 3
- **Build command** (example):

```bash
pip install -r requirements.txt && python train_model.py --data data/sample_loan_data.csv --out models/loan_pipeline.pkl
```

- **Start command**:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT
```

## 3) Environment variables

- `**SECRET_KEY**`: long random string (required for sessions in production)

## 4) SQLite persistence note

Render **ephemeral disks** reset on redeploy. SQLite (`db.sqlite3`) is fine for demos; for production data retention use a managed database (for example Render Postgres) and replace `sqlite3` usage accordingly.

## 5) Smoke test after deploy

- Open `/` (landing)
- Create user, login, submit `/predict`
- Visit `/dashboard`


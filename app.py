from __future__ import annotations

import json
import math
import os
import pickle
import sqlite3
import time
from datetime import datetime
from functools import wraps
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "db.sqlite3"
MODEL_PATH = BASE_DIR / "models" / "loan_pipeline.pkl"
UPLOAD_DIR = BASE_DIR / "instance" / "uploads"
ANALYTICS_XLSX_PATH = BASE_DIR / "loan_prediction_500_dataset.xlsx"

NUMERIC_ML = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
CAT_ML = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Employment_Status",
    "Property_Area",
]

STUDENT_JOBS = {"Food Servent", "Shop Keeping", "Xerox Shop", "Other"}

# Dashboard + validation: exactly these three employment types
EMPLOYMENT_ORDER = ("Student", "Employed", "Self Employed")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-change-me")

_ANALYTICS_CACHE: dict = {"mtime": None, "df": None, "loaded_at": None}


def _canon(s: str) -> str:
    return (
        str(s)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def load_analytics_df() -> pd.DataFrame:
    """
    Load analytics dataset from Excel with a tiny in-memory cache.
    Requires `openpyxl` installed (pandas engine).
    """
    if not ANALYTICS_XLSX_PATH.exists():
        raise FileNotFoundError(
            f"Analytics Excel file not found: {ANALYTICS_XLSX_PATH}"
        )

    mtime = ANALYTICS_XLSX_PATH.stat().st_mtime
    if _ANALYTICS_CACHE["df"] is not None and _ANALYTICS_CACHE["mtime"] == mtime:
        return _ANALYTICS_CACHE["df"].copy()

    df = pd.read_excel(ANALYTICS_XLSX_PATH)

    # Column mapping layer: guess likely headers and normalize to canonical names.
    cols = {_canon(c): c for c in df.columns}

    def pick(*keys: str) -> str | None:
        for k in keys:
            if _canon(k) in cols:
                return cols[_canon(k)]
        return None

    column_map = {
        # Exact matches for your Excel + fallbacks if headers change slightly
        "application_id": pick("Application_ID", "application_id", "applicationid", "id", "loan_id", "loanid"),
        "applicant_name": pick("Applicant_Name", "applicant_name", "name", "customer_name", "applicant"),
        "loan_type": pick("Loan_Type", "loan_type", "loan_purpose", "purpose", "category"),
        "amount": pick("Loan_Amount", "loan_amount", "amount", "loanamount", "loan_amt"),
        "status": pick("Loan_Status", "loan_status", "status", "approval_status", "decision", "loanstatus"),
        "probability": pick("Prediction_Probability", "prediction_probability", "probability", "approval_probability", "proba", "score"),
        "state": pick("State", "state", "region", "province"),
        "date": pick("Application_Date", "application_date", "date", "created_at", "submitted_at"),
        "income": pick("Income", "income", "applicant_income", "applicantincome", "monthly_income"),
        # Optional / not required but kept for risk heuristics
        "employment": pick("Employment_Type", "employment_type", "employment", "employment_status", "job_type"),
        "credit_score": pick("Credit_Score", "credit_score", "score"),
        "age": pick("Age", "age"),
        "gender": pick("Gender", "gender"),
    }

    # Rename columns if we found them; keep others as-is.
    rename = {src: dst for dst, src in column_map.items() if src}
    df = df.rename(columns=rename)

    # Ensure at least these columns exist (create if missing).
    for need in ("status", "amount", "date"):
        if need not in df.columns:
            df[need] = pd.NA

    # Normalize basic fields.
    df["status"] = df["status"].astype(str).str.strip()
    df["status_norm"] = (
        df["status"]
        .str.lower()
        .str.replace(r"[^a-z]+", " ", regex=True)
        .str.strip()
    )

    # Try to parse dates if present.
    df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce")

    # Amount numeric
    df["amount_num"] = pd.to_numeric(df.get("amount"), errors="coerce")

    # Income numeric
    if "income" in df.columns:
        df["income_num"] = pd.to_numeric(df.get("income"), errors="coerce")
    else:
        df["income_num"] = pd.NA

    # Probability numeric
    if "probability" in df.columns:
        # Handles values like \"92%\" as well as floats
        p = df.get("probability").astype(str).str.strip()
        p = p.str.replace("%", "", regex=False)
        df["probability_num"] = pd.to_numeric(p, errors="coerce")
    else:
        df["probability_num"] = pd.NA

    _ANALYTICS_CACHE.update({"mtime": mtime, "df": df.copy(), "loaded_at": time.time()})
    return df.copy()


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            features_json TEXT NOT NULL,
            final_label TEXT NOT NULL,
            approval_proba REAL,
            model_label TEXT,
            model_proba REAL,
            decision_source TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """
    )
    conn.commit()
    conn.close()


def load_model():
    if not MODEL_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


MODEL = load_model()


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("user_id") is None:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def compute_rejection_reasons(feat: dict) -> list[str]:
    reasons: list[str] = []
    try:
        ch = int(feat.get("Credit_History"))
    except (TypeError, ValueError):
        ch = None
    if ch == 0:
        reasons.append("Credit history does not meet the minimum requirement.")

    try:
        inc = float(feat.get("ApplicantIncome") or 0) + float(
            feat.get("CoapplicantIncome") or 0
        )
        loan = float(feat.get("LoanAmount") or 0)
    except (TypeError, ValueError):
        inc, loan = 0.0, 0.0

    if inc > 0 and loan / inc > 20:
        reasons.append("Loan amount is high compared to total income.")

    if inc < 3000:
        reasons.append("Total income appears low relative to typical approval profiles.")

    try:
        term = float(feat.get("Loan_Amount_Term") or 0)
    except (TypeError, ValueError):
        term = 0.0
    if term and term < 60:
        reasons.append("Loan term is unusually short for this profile.")

    if not reasons:
        reasons.append("The application does not match typical approval patterns.")
    return reasons[:5]


def student_policy_check(feat: dict, uploaded_filename: str | None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    job = (feat.get("Student_Part_Time_Job") or "").strip()
    if not uploaded_filename:
        failures.append("College ID proof is required for student applications.")
    if job not in STUDENT_JOBS:
        failures.append("A valid student part-time job selection is required.")
    try:
        ai = float(feat.get("ApplicantIncome") or 0)
    except (TypeError, ValueError):
        ai = 0.0
    if ai < 3000 or ai > 5000:
        failures.append(
            "For students, applicant income must be between ₹3000 and ₹5000 (INR)."
        )
    return (len(failures) == 0, failures)


def ml_row_from_features(feat: dict) -> pd.DataFrame:
    row = {k: feat[k] for k in NUMERIC_ML + CAT_ML}
    return pd.DataFrame([row])


def predict_with_model(feat: dict) -> tuple[str, float | None]:
    if MODEL is None:
        raise RuntimeError("Model file missing. Run train_model.py first.")
    X = ml_row_from_features(feat)
    label = str(MODEL.predict(X)[0])
    proba = None
    if hasattr(MODEL, "predict_proba"):
        probs = MODEL.predict_proba(X)[0]
        classes = list(MODEL.classes_)
        if "Y" in classes:
            proba = float(probs[classes.index("Y")])
    return label, proba


def render_pdf(
    *,
    title: str,
    final_label: str,
    approval_proba: float | None,
    reasons: list[str],
    feat: dict,
) -> BytesIO:
    buf = BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title)
    y -= 30
    c.setFont("Helvetica", 11)
    c.drawString(50, y, f"Decision: {final_label}")
    y -= 18
    if approval_proba is not None:
        c.drawString(50, y, f"Estimated approval chance: {approval_proba*100:.1f}%")
        y -= 18
    if reasons:
        y -= 10
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Reasons (high-level)")
        y -= 16
        c.setFont("Helvetica", 11)
        for r in reasons:
            c.drawString(60, y, f"- {r}")
            y -= 16
            if y < 80:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 11)
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Application details")
    y -= 16
    c.setFont("Helvetica", 10)
    skip = {"password"}
    for k, v in sorted(feat.items()):
        if k in skip:
            continue
        line = f"{k}: {v}"
        c.drawString(50, y, line[:110])
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 50
    c.showPage()
    c.save()
    buf.seek(0)
    return buf


def parse_core_features_from_form() -> dict:
    def num(name: str) -> float:
        return float(request.form[name])

    def txt(name: str) -> str:
        return (request.form.get(name) or "").strip()

    employment = txt("Employment_Status")
    loan_raw = num("LoanAmount")
    feat = {
        "Gender": txt("Gender"),
        "Married": txt("Married"),
        "Dependents": txt("Dependents"),
        "Education": txt("Education"),
        "Employment_Status": employment,
        "ApplicantIncome": num("ApplicantIncome"),
        "CoapplicantIncome": num("CoapplicantIncome"),
        "LoanAmount": normalize_loan_amount_inr(loan_raw),
        "LoanAmount_entered": loan_raw,
        "Loan_Amount_Term": num("Loan_Amount_Term"),
        "Credit_History": int(request.form["Credit_History"]),
        "Property_Area": txt("Property_Area"),
        "Mobile_Number": txt("Mobile_Number"),
        "Address": txt("Address"),
        "Country": txt("Country"),
        "State": txt("State"),
        "Student_Part_Time_Job": txt("Student_Part_Time_Job") or None,
    }
    return feat


def validate_contact(feat: dict) -> str | None:
    mobile = feat.get("Mobile_Number", "")
    if not mobile.isdigit() or not (10 <= len(mobile) <= 15):
        return "Mobile number must be digits only (10-15 length)."
    for k in ("Address", "Country", "State"):
        v = (feat.get(k) or "").strip()
        if len(v) < 2:
            return f"{k} is required."
    return None


def validate_employment(emp: str) -> str | None:
    if emp not in EMPLOYMENT_ORDER:
        return "Employment status must be one of: Student, Employed, Self Employed."
    return None


def normalize_loan_amount_inr(value: float) -> float:
    """Treat small positive values as thousands (120 → 120000 INR) for demo shorthand."""
    if value <= 0:
        return value
    if value < 1000:
        return value * 1000
    return value


def analytics_date_range_label(df: pd.DataFrame) -> str:
    if df["date_parsed"].notna().any():
        mn = df["date_parsed"].min()
        mx = df["date_parsed"].max()
        return f"{mn:%b %d, %Y} – {mx:%b %d, %Y}"
    return "Dataset"


def analytics_column_overview(df: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for c in df.columns:
        col = df[c]
        nn = float(col.notna().mean() * 100.0)
        rows.append(
            {
                "name": c,
                "dtype": str(col.dtype),
                "non_null_pct": round(nn, 1),
                "n_unique": int(col.nunique(dropna=True)),
            }
        )
    return rows


def analytics_numeric_summary(df: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    for c in ("amount_num", "income_num", "credit_score", "age"):
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            out.append(
                {
                    "name": c.replace("_num", "").replace("_", " ").title(),
                    "mean": float(s.mean()),
                    "median": float(s.median()),
                    "min": float(s.min()),
                    "max": float(s.max()),
                }
            )
    return out


def analytics_status_counts(df: pd.DataFrame) -> list[dict]:
    vc = df["status"].astype(str).value_counts()
    return [{"status": k, "count": int(v)} for k, v in vc.items()]


def analytics_build_alerts(df: pd.DataFrame) -> list[dict]:
    alerts: list[dict] = []
    if "state" in df.columns and len(df):
        g = df.copy()
        g["apr"] = g["status_norm"].apply(_is_approved)
        st = (
            g.groupby(g["state"].astype(str).str.strip(), dropna=False)
            .agg(applications=("application_id", "count"), approved=("apr", "sum"))
            .reset_index()
        )
        st = st[st["applications"] >= 10]
        st["rate"] = st["approved"] / st["applications"]
        bad = st[st["rate"] < 0.35].sort_values("applications", ascending=False)
        for _, r in bad.head(8).iterrows():
            alerts.append(
                {
                    "severity": "high",
                    "title": f"Low approval in {r['state']}",
                    "detail": f"{int(r['applications'])} applications, approval rate {r['rate']*100:.1f}%.",
                }
            )
    if "credit_score" in df.columns and "amount_num" in df.columns:
        cs = pd.to_numeric(df["credit_score"], errors="coerce")
        amt = pd.to_numeric(df["amount_num"], errors="coerce")
        risky = int(((cs < 600) & (amt > 800_000)).sum())
        if risky:
            alerts.append(
                {
                    "severity": "medium",
                    "title": "High loan amount with low credit score",
                    "detail": f"{risky} applications have credit score under 600 and loan amount above ₹8,00,000.",
                }
            )
    if "employment" in df.columns:
        g = df.copy()
        g["apr"] = g["status_norm"].apply(_is_approved)
        emp = (
            g.groupby(g["employment"].astype(str).str.strip(), dropna=False)
            .agg(n=("application_id", "count"), approved=("apr", "sum"))
            .reset_index()
        )
        emp = emp[emp["n"] >= 20]
        if len(emp):
            emp["rate"] = emp["approved"] / emp["n"]
            worst = emp.sort_values("rate").iloc[0]
            alerts.append(
                {
                    "severity": "low",
                    "title": f"Lowest approval employment type: {worst['employment']}",
                    "detail": f"{int(worst['n'])} applications, approval rate {float(worst['rate'])*100:.1f}%.",
                }
            )
    if not alerts:
        alerts.append(
            {
                "severity": "low",
                "title": "No critical alerts",
                "detail": "No automated risk thresholds were crossed in this dataset slice.",
            }
        )
    return alerts


@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/analytics")
@login_required
def analytics():
    df = load_analytics_df()
    return render_template(
        "analytics.html",
        analytics_nav="dashboard",
        date_range_label=analytics_date_range_label(df),
    )


@app.route("/analytics/data-overview")
@login_required
def analytics_data_overview():
    df = load_analytics_df()
    return render_template(
        "analytics_data_overview.html",
        analytics_nav="data_overview",
        date_range_label=analytics_date_range_label(df),
        n_rows=int(len(df)),
        n_cols=int(len(df.columns)),
        column_stats=analytics_column_overview(df),
        numeric_summary=analytics_numeric_summary(df),
        status_counts=analytics_status_counts(df),
    )


@app.route("/analytics/loan-applications")
@login_required
def analytics_loan_applications():
    df = load_analytics_df()
    q = (request.args.get("q") or "").strip().lower()
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(10, int(request.args.get("per_page", 30))))

    view = df.copy()
    if q:
        mask = view.astype(str).apply(
            lambda row: any(q in str(v).lower() for v in row.values), axis=1
        )
        view = view[mask]

    total_rows = int(len(view))
    total_pages = max(1, (total_rows + per_page - 1) // per_page)
    page = min(page, total_pages)
    start = (page - 1) * per_page
    chunk = view.iloc[start : start + per_page]

    rows = []
    for _, r in chunk.iterrows():
        ad = r.get("date_parsed")
        rows.append(
            {
                "application_id": str(r.get("application_id", "")),
                "applicant_name": str(r.get("applicant_name", "")),
                "age": r.get("age", ""),
                "gender": str(r.get("gender", "")),
                "state": str(r.get("state", "")),
                "loan_type": str(r.get("loan_type", "")),
                "loan_amount": float(r.get("amount_num") or 0),
                "income": float(r.get("income_num") or 0)
                if pd.notna(r.get("income_num"))
                else None,
                "credit_score": r.get("credit_score", ""),
                "employment": str(r.get("employment", "")),
                "loan_status": str(r.get("status", "")),
                "probability": str(r.get("probability", "")),
                "application_date": ad.strftime("%Y-%m-%d")
                if pd.notna(ad)
                else "",
            }
        )

    return render_template(
        "analytics_loan_applications.html",
        analytics_nav="loan_applications",
        date_range_label=analytics_date_range_label(df),
        search_q=request.args.get("q") or "",
        rows=rows,
        page=page,
        per_page=per_page,
        total_rows=total_rows,
        total_pages=total_pages,
    )


@app.route("/analytics/customers")
@login_required
def analytics_customers():
    df = load_analytics_df()
    d2 = df.copy()
    d2["apr"] = d2["status_norm"].apply(_is_approved)
    cust = (
        d2.groupby(d2["applicant_name"].astype(str).str.strip(), dropna=False)
        .agg(
            applications=("application_id", "count"),
            approved=("apr", "sum"),
            total_loan=("amount_num", "sum"),
            avg_income=("income_num", "mean"),
        )
        .reset_index()
    )
    cust = cust.rename(columns={"applicant_name": "customer_name"})
    cust["rejected"] = cust["applications"] - cust["approved"].astype(int)
    cust["approval_rate_pct"] = (cust["approved"] / cust["applications"] * 100).round(1)
    cust = cust.sort_values("applications", ascending=False)
    customers = cust.head(200).to_dict(orient="records")

    for c in customers:
        ai = c.get("avg_income")
        if isinstance(ai, (int, float)) and not (isinstance(ai, float) and math.isnan(ai)):
            c["avg_income_fmt"] = f"{float(ai):,.0f}"
        else:
            c["avg_income_fmt"] = "—"
        tl = c.get("total_loan")
        c["total_loan_fmt"] = f"{float(tl):,.0f}" if tl is not None else "—"

    return render_template(
        "analytics_customers.html",
        analytics_nav="customers",
        date_range_label=analytics_date_range_label(df),
        customers=customers,
        unique_customers=int(cust.shape[0]),
    )


@app.route("/analytics/reports")
@login_required
def analytics_reports():
    df = load_analytics_df()
    total = int(len(df))
    approved = int(df["status_norm"].apply(_is_approved).sum())
    rejected = int(df["status_norm"].apply(_is_rejected).sum())
    total_amount = float(df["amount_num"].fillna(0).sum())

    top_states: list[dict] = []
    if "state" in df.columns:
        ts = df["state"].astype(str).str.strip().value_counts().head(10).reset_index()
        ts.columns = ["state", "applications"]
        top_states = ts.to_dict(orient="records")

    top_types: list[dict] = []
    if "loan_type" in df.columns:
        tt = df["loan_type"].astype(str).str.strip().value_counts().head(8).reset_index()
        tt.columns = ["loan_type", "applications"]
        top_types = tt.to_dict(orient="records")

    monthly: list[dict] = []
    if df["date_parsed"].notna().any():
        t = df.dropna(subset=["date_parsed"]).copy()
        t["month"] = t["date_parsed"].dt.to_period("M").astype(str)
        for month, g in t.groupby("month", dropna=True):
            monthly.append(
                {
                    "month": month,
                    "applications": int(len(g)),
                    "approved": int(g["status_norm"].apply(_is_approved).sum()),
                    "rejected": int(g["status_norm"].apply(_is_rejected).sum()),
                }
            )
        monthly = sorted(monthly, key=lambda x: x["month"])

    return render_template(
        "analytics_reports.html",
        analytics_nav="reports",
        date_range_label=analytics_date_range_label(df),
        summary={
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate_pct": round((approved / total) * 100, 1) if total else 0.0,
            "total_loan_amount": total_amount,
        },
        top_states=top_states,
        top_types=top_types,
        monthly=monthly,
    )


@app.route("/analytics/alerts")
@login_required
def analytics_alerts():
    df = load_analytics_df()
    alerts = analytics_build_alerts(df)
    return render_template(
        "analytics_alerts.html",
        analytics_nav="alerts",
        date_range_label=analytics_date_range_label(df),
        alerts=alerts,
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if len(name) < 2 or "@" not in email or len(password) < 6:
            flash("Please provide a valid name, email, and password (6+ chars).")
            return render_template("signup.html")
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO users (name, email, password_hash, created_at) VALUES (?,?,?,?)",
                (
                    name,
                    email,
                    generate_password_hash(password),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            flash("Email already registered.")
            return render_template("signup.html")
        finally:
            conn.close()
        flash("Account created. Please log in.")
        return redirect(url_for("login"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        conn = get_db()
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        conn.close()
        if row and check_password_hash(row["password_hash"], password):
            session.clear()
            session["user_id"] = row["id"]
            session["user_name"] = row["name"]
            nxt = request.args.get("next") or url_for("predict")
            return redirect(nxt)
        flash("Invalid credentials.")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    prediction_id = None
    if request.method == "POST":
        try:
            feat = parse_core_features_from_form()
            err = validate_contact(feat)
            if err:
                flash(err)
                return render_template("predict.html")
            err = validate_employment(feat.get("Employment_Status", ""))
            if err:
                flash(err)
                return render_template("predict.html")

            f = request.files.get("Student_ID_Proof")
            id_filename = None
            if f and f.filename:
                id_filename = secure_filename(f.filename)
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                dest = UPLOAD_DIR / f"{session['user_id']}_{int(datetime.utcnow().timestamp())}_{id_filename}"
                f.save(dest)

            feat["Student_ID_Proof_Filename"] = id_filename

            model_label, model_proba = predict_with_model(feat)
            decision_source = "ml"
            final_label = model_label
            approval_proba = model_proba

            if feat["Employment_Status"] == "Student":
                ok, fails = student_policy_check(feat, id_filename)
                if ok:
                    final_label = "Y"
                    approval_proba = max(model_proba or 0.0, 0.85)
                    decision_source = "student_policy"
                else:
                    final_label = "N"
                    approval_proba = min(model_proba or 0.35, 0.35)
                    decision_source = "student_policy"

            reasons = (
                compute_rejection_reasons(feat)
                if final_label == "N"
                else []
            )
            if feat["Employment_Status"] == "Student" and final_label == "N":
                ok, fails = student_policy_check(feat, id_filename)
                for msg in fails:
                    if msg not in reasons:
                        reasons.insert(0, msg)

            conn = get_db()
            cur = conn.execute(
                """
                INSERT INTO predictions
                (user_id, created_at, features_json, final_label, approval_proba, model_label, model_proba, decision_source)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (
                    session["user_id"],
                    datetime.utcnow().isoformat(),
                    json.dumps(feat, default=str),
                    final_label,
                    approval_proba,
                    model_label,
                    model_proba,
                    decision_source,
                ),
            )
            conn.commit()
            prediction_id = cur.lastrowid
            conn.close()

            result = {
                "final_label": final_label,
                "approval_proba": approval_proba,
                "reasons": reasons,
                "prediction_id": prediction_id,
            }
        except Exception as e:  # noqa: BLE001
            flash(str(e))
    return render_template("predict.html", result=result)


@app.route("/download/<int:prediction_id>")
@login_required
def download_decision(prediction_id: int):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM predictions WHERE id = ? AND user_id = ?",
        (prediction_id, session["user_id"]),
    ).fetchone()
    conn.close()
    if not row:
        return "Not found", 404

    feat = json.loads(row["features_json"])
    final = "Approved" if row["final_label"] == "Y" else "Rejected"
    reasons = compute_rejection_reasons(feat) if row["final_label"] == "N" else []
    if feat.get("Employment_Status") == "Student" and row["final_label"] == "N":
        _, fails = student_policy_check(feat, feat.get("Student_ID_Proof_Filename"))
        for msg in fails:
            if msg not in reasons:
                reasons.insert(0, msg)

    pdf = render_pdf(
        title="Loan Decision Summary",
        final_label=final,
        approval_proba=row["approval_proba"],
        reasons=reasons,
        feat=feat,
    )
    fname = f"loan_decision_{prediction_id}_{final.lower()}.pdf"
    return send_file(
        pdf,
        as_attachment=True,
        download_name=fname,
        mimetype="application/pdf",
    )


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/dashboard")
@login_required
def api_dashboard():
    conn = get_db()
    rows = conn.execute(
        """
        SELECT final_label, features_json
        FROM predictions
        WHERE user_id = ?
        """,
        (session["user_id"],),
    ).fetchall()
    conn.close()

    total = len(rows)
    approvals = sum(1 for r in rows if r["final_label"] == "Y")
    by_employment: dict[str, int] = {}
    by_property: dict[str, int] = {}
    employment_breakdown: dict[str, dict[str, int]] = {
        k: {"Y": 0, "N": 0} for k in EMPLOYMENT_ORDER
    }
    for r in rows:
        try:
            f = json.loads(r["features_json"])
        except json.JSONDecodeError:
            continue
        emp = f.get("Employment_Status") or "Unknown"
        by_employment[emp] = by_employment.get(emp, 0) + 1
        prop = f.get("Property_Area") or "Unknown"
        by_property[prop] = by_property.get(prop, 0) + 1
        if emp in employment_breakdown:
            lbl = r["final_label"]
            if lbl == "Y":
                employment_breakdown[emp]["Y"] += 1
            elif lbl == "N":
                employment_breakdown[emp]["N"] += 1

    series = []
    for i in range(6):
        series.append({"x": i, "y": approvals + i * 2})

    return jsonify(
        {
            "total": total,
            "approvals": approvals,
            "rejections": total - approvals,
            "approval_rate": (approvals / total) if total else 0.0,
            "by_employment_status": by_employment,
            "employment_order": list(EMPLOYMENT_ORDER),
            "employment_breakdown": employment_breakdown,
            "by_property_area": by_property,
            "trend": series,
        }
    )


def _is_approved(status_norm: str) -> bool:
    s = (status_norm or "").strip().lower()
    return s in {"approved", "approve", "yes", "y", "success"}


def _is_rejected(status_norm: str) -> bool:
    s = (status_norm or "").strip().lower()
    return s in {"rejected", "reject", "no", "n", "failed", "declined"}


@app.route("/api/analytics")
@login_required
def api_analytics():
    df = load_analytics_df()

    total = int(len(df))
    approved = int(df["status_norm"].apply(_is_approved).sum())
    rejected = int(df["status_norm"].apply(_is_rejected).sum())
    total_amount = float(df["amount_num"].fillna(0).sum())

    # Monthly trend (fallback to row index if no date)
    trend = []
    if df["date_parsed"].notna().any():
        t = df.dropna(subset=["date_parsed"]).copy()
        t["month"] = t["date_parsed"].dt.to_period("M").astype(str)
        grp = t.groupby("month", dropna=True)
        for month, g in grp:
            trend.append(
                {
                    "month": month,
                    "applications": int(len(g)),
                    "approved": int(g["status_norm"].apply(_is_approved).sum()),
                    "rejected": int(g["status_norm"].apply(_is_rejected).sum()),
                }
            )
        trend = sorted(trend, key=lambda x: x["month"])
    else:
        # Create a small synthetic trend so UI doesn't break
        trend = [{"month": f"M{i+1}", "applications": 0, "approved": 0, "rejected": 0} for i in range(6)]

    approval_rate = (approved / total) if total else 0.0

    # Income distribution buckets (use income_num; if missing, show zeros)
    buckets = [
        ("0–25K", 0, 25000),
        ("25K–50K", 25000, 50000),
        ("50K–75K", 50000, 75000),
        ("75K–100K", 75000, 100000),
        ("100K+", 100000, float("inf")),
    ]
    income_dist = []
    inc = pd.to_numeric(df["income_num"], errors="coerce")
    for label, lo, hi in buckets:
        if hi == float("inf"):
            count = int((inc >= lo).sum())
        else:
            count = int(((inc >= lo) & (inc < hi)).sum())
        income_dist.append({"range": label, "count": count})

    # Loan purpose distribution
    if "loan_type" in df.columns:
        purpose_counts = (
            df["loan_type"].astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(8)
        )
        purposes = [{"label": k, "count": int(v)} for k, v in purpose_counts.items()]
    else:
        purposes = [
            {"label": "Home Loan", "count": 0},
            {"label": "Personal Loan", "count": 0},
            {"label": "Education Loan", "count": 0},
            {"label": "Vehicle Loan", "count": 0},
            {"label": "Others", "count": 0},
        ]

    # Region (State) distribution for map
    if "state" in df.columns:
        state_counts = (
            df["state"].astype(str).str.strip().replace({"": "Unknown"}).value_counts().head(40)
        )
        region = [{"state": k, "applications": int(v)} for k, v in state_counts.items()]
    else:
        region = []

    # Risk factors (heuristics; if fields missing, use simple defaults)
    risk = []
    if "credit_score" in df.columns:
        cs = pd.to_numeric(df["credit_score"], errors="coerce")
        pct = float((cs.fillna(0) < 600).mean() * 100.0) if len(cs) else 0.0
        risk.append({"label": "Credit score (<600)", "percent": round(pct, 1)})
    elif "credit_history" in df.columns:
        ch = pd.to_numeric(df["credit_history"], errors="coerce")
        pct = float((ch.fillna(0) <= 0).mean() * 100.0) if len(ch) else 0.0
        risk.append({"label": "Credit History", "percent": round(pct, 1)})
    else:
        risk.append({"label": "Credit History", "percent": 32.0})

    # DTI
    if "dti" in df.columns:
        dti = pd.to_numeric(df["dti"], errors="coerce")
        pct = float((dti.fillna(0) >= 0.4).mean() * 100.0) if len(dti) else 0.0
    else:
        pct = 26.0
    risk.append({"label": "Debt-to-Income Ratio", "percent": round(pct, 1)})

    # Income stability (proxy: missing/very low income)
    pct = float((inc.fillna(0) < 25000).mean() * 100.0) if len(inc) else 0.0
    risk.append({"label": "Income Stability", "percent": round(pct, 1)})

    # Loan amount risk (top quartile)
    amt = pd.to_numeric(df["amount_num"], errors="coerce")
    if amt.notna().any():
        q = float(amt.quantile(0.75))
        pct = float((amt.fillna(0) >= q).mean() * 100.0)
    else:
        pct = 14.0
    risk.append({"label": "Loan Amount", "percent": round(pct, 1)})

    # Employment type (if available: percent of Student/Unknown)
    if "employment" in df.columns:
        emp = df["employment"].astype(str).str.lower()
        pct = float((emp.str.contains("student") | (emp.str.strip() == "")).mean() * 100.0)
    else:
        pct = 10.0
    risk.append({"label": "Employment Type", "percent": round(pct, 1)})

    # Recent applications table (top 20)
    recent = []
    table_df = df.copy()
    if table_df["date_parsed"].notna().any():
        table_df = table_df.sort_values("date_parsed", ascending=False)
    for _, r in table_df.head(25).iterrows():
        status = str(r.get("status", "")).strip() or "Pending"
        pred = r.get("prediction", "")
        proba = r.get("probability_num")
        if pd.isna(proba):
            proba = None
        recent.append(
            {
                "application_id": str(r.get("application_id", "")) or "-",
                "applicant_name": str(r.get("applicant_name", "")) or "-",
                "loan_type": str(r.get("loan_type", "")) or "-",
                "amount": float(r.get("amount_num") or 0),
                "status": status,
                "prediction": str(pred) if pred is not None else "",
                "probability": float(proba) if proba is not None else None,
            }
        )

    insights = [
        {
            "title": "Approval rate improved",
            "body": f"Current approval rate is {approval_rate*100:.1f}% based on the dataset.",
            "icon": "trend",
        },
        {
            "title": "Highest volume purpose",
            "body": (purposes[0]["label"] if purposes else "Unknown") + " has the most applications.",
            "icon": "star",
        },
        {
            "title": "Regional concentration",
            "body": (region[0]["state"] if region else "N/A") + " shows the highest application volume.",
            "icon": "map",
        },
        {
            "title": "Recommendation",
            "body": "Focus verification on credit history and affordability to reduce high-risk approvals.",
            "icon": "spark",
        },
    ]

    return jsonify(
        {
            "kpis": {
                "total_applications": total,
                "approved_loans": approved,
                "rejected_loans": rejected,
                "total_loan_amount": total_amount,
                "approval_rate": approval_rate,
            },
            "trend": trend,
            "income_distribution": income_dist,
            "loan_purpose": purposes,
            "region": region,
            "risk_factors": risk,
            "recent": recent,
            "insights": insights,
        }
    )


@app.route("/api/predict", methods=["POST"])
@login_required
def api_predict():
    payload = request.get_json(force=True, silent=True) or {}
    try:
        feat = {
            "Gender": str(payload["Gender"]).strip(),
            "Married": str(payload["Married"]).strip(),
            "Dependents": str(payload["Dependents"]).strip(),
            "Education": str(payload["Education"]).strip(),
            "Employment_Status": str(payload["Employment_Status"]).strip(),
            "ApplicantIncome": float(payload["ApplicantIncome"]),
            "CoapplicantIncome": float(payload.get("CoapplicantIncome", 0)),
            "LoanAmount": normalize_loan_amount_inr(float(payload["LoanAmount"])),
            "LoanAmount_entered": float(payload["LoanAmount"]),
            "Loan_Amount_Term": float(payload["Loan_Amount_Term"]),
            "Credit_History": int(payload["Credit_History"]),
            "Property_Area": str(payload["Property_Area"]).strip(),
            "Mobile_Number": str(payload.get("Mobile_Number", "")).strip(),
            "Address": str(payload.get("Address", "")).strip(),
            "Country": str(payload.get("Country", "")).strip(),
            "State": str(payload.get("State", "")).strip(),
            "Student_Part_Time_Job": payload.get("Student_Part_Time_Job"),
            "Student_ID_Proof_Filename": payload.get("Student_ID_Proof_Filename"),
        }
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({"error": f"Invalid payload: {e}"}), 400

    err = validate_contact(feat)
    if err:
        return jsonify({"error": err}), 400
    err = validate_employment(feat.get("Employment_Status", ""))
    if err:
        return jsonify({"error": err}), 400

    model_label, model_proba = predict_with_model(feat)
    decision_source = "ml"
    final_label = model_label
    approval_proba = model_proba

    if feat["Employment_Status"] == "Student":
        ok, _ = student_policy_check(
            feat, feat.get("Student_ID_Proof_Filename")
        )
        if ok:
            final_label = "Y"
            approval_proba = max(model_proba or 0.0, 0.85)
            decision_source = "student_policy"
        else:
            final_label = "N"
            approval_proba = min(model_proba or 0.35, 0.35)
            decision_source = "student_policy"

    reasons = compute_rejection_reasons(feat) if final_label == "N" else []
    if feat["Employment_Status"] == "Student" and final_label == "N":
        _, fails = student_policy_check(
            feat, feat.get("Student_ID_Proof_Filename")
        )
        for msg in fails:
            if msg not in reasons:
                reasons.insert(0, msg)

    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO predictions
        (user_id, created_at, features_json, final_label, approval_proba, model_label, model_proba, decision_source)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            session["user_id"],
            datetime.utcnow().isoformat(),
            json.dumps(feat, default=str),
            final_label,
            approval_proba,
            model_label,
            model_proba,
            decision_source,
        ),
    )
    conn.commit()
    pid = cur.lastrowid
    conn.close()

    return jsonify(
        {
            "prediction_id": pid,
            "prediction": "Approved" if final_label == "Y" else "Rejected",
            "label": final_label,
            "approval_probability": approval_proba,
            "decision_source": decision_source,
            "rejection_reasons": reasons,
        }
    )


init_db()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=True)

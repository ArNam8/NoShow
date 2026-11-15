"""
No-Show Rescue - Flask edition.
One page, one button: upload a clinic's appointment book (CSV),
get a ranked call list with the money at stake on screen.

The brain (features + model) is identical to the Colab-tested version.
"""
import io
import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# --- 1. Load the trained model (made in Colab, saved with joblib) ---
model = joblib.load("noshow_model.pkl")

FEATURES = ["Age", "LeadTime", "SMS_received", "Scholarship", "Hipertension",
            "Diabetes", "Alcoholism", "Handcap", "PriorNoShowRate",
            "PriorApps", "AppWeekday", "SchedHour"]

VALUE_PER_APPT = 200  # dollars a missed appointment costs a clinic

# --- 2. Turn raw appointment rows into the model's 12 clues ---
def build_features(df):
    df = df.copy()
    df["ScheduledDay"] = pd.to_datetime(df["ScheduledDay"], utc=True).dt.tz_localize(None)
    df["AppointmentDay"] = pd.to_datetime(df["AppointmentDay"], utc=True).dt.tz_localize(None)
    df["NoShow"] = (df["No-show"] == "Yes").astype(int)

    df["LeadTime"] = (df["AppointmentDay"] - df["ScheduledDay"].dt.normalize()).dt.days.clip(lower=0)

    df = df.sort_values(["PatientId", "AppointmentDay", "ScheduledDay"])
    df["PriorApps"] = df.groupby("PatientId").cumcount()
    df["PriorNoShowRate"] = (
        df.groupby("PatientId")["NoShow"]
        .apply(lambda s: s.shift().expanding().mean())
        .reset_index(level=0, drop=True)
    )
    df["PriorNoShowRate"] = df["PriorNoShowRate"].fillna(df["NoShow"].mean())

    df["AppWeekday"] = df["AppointmentDay"].dt.weekday
    df["SchedHour"] = df["ScheduledDay"].dt.hour
    return df

# --- 3. Rank today's appointments and count the money ---
def make_call_list(file_bytes, filename, call_budget):
    if filename.lower().endswith((".xlsx", ".xls")):
        raw = pd.read_excel(io.BytesIO(file_bytes))
    else:
        raw = pd.read_csv(io.BytesIO(file_bytes))
    df = build_features(raw)

    today = df["AppointmentDay"].max()
    book = df[df["AppointmentDay"] == today].copy()

    book["risk"] = model.predict_proba(book[FEATURES])[:, 1]

    # appointment times are not in the data, so we give everyone a slot from 8:00
    def slot_time(i):
        mins = 8 * 60 + i * 5
        h, m = divmod(mins, 60)
        ampm = "AM" if h < 12 else "PM"
        h12 = h if h <= 12 else h - 12
        return str(h12) + ":" + str(m).zfill(2) + " " + ampm
    book = book.reset_index(drop=True)
    book["time"] = [slot_time(i) for i in range(len(book))]

    def phone_for(i):
        return "+55 27 9" + str(8000 + (i * 37) % 200).zfill(4) + "-" + str(1000 + (i * 91) % 9000).zfill(4)

    def reason_for(r):
        if r["PriorNoShowRate"] >= 0.3: return "Prior no-show history", "r"
        if r["LeadTime"] > 30: return "Booked >30 days ago", "y"
        if r["PriorApps"] == 0: return "First visit", "g"
        if r["SMS_received"] == 0: return "No SMS reminder", "y"
        return "Pattern match", "g"

    book = book.sort_values("risk", ascending=False).reset_index(drop=True)

    calls = book.head(int(call_budget))
    expected = float(calls["risk"].sum())
    at_stake = expected * VALUE_PER_APPT

    rule = book.sort_values("LeadTime", ascending=False).head(int(call_budget))
    rule_expected = float(rule["risk"].sum())
    edge = (expected - rule_expected) * VALUE_PER_APPT

    if "Name" in book.columns:
        names = calls["Name"].tolist()
    else:
        names = ["Patient " + str(pid)[-4:] for pid in calls["PatientId"]]

    rows = []
    for i in range(len(calls)):
        reason, dot = reason_for(calls.iloc[i])
        rows.append({
            "rank": i + 1,
            "name": names[i],
            "phone": phone_for(int(str(calls["PatientId"].iloc[i]).split(".")[0]) % 900 if "PatientId" in calls.columns else i),
            "time": calls["time"].iloc[i],
            "risk": round(float(calls["risk"].iloc[i]) * 100),
            "reason": reason,
            "dot": dot,
        })

    all_names = book["Name"].tolist() if "Name" in book.columns else ["Patient " + str(pid)[-4:] for pid in book["PatientId"]]
    all_people = [{
        "name": all_names[i],
        "time": book["time"].iloc[i],
        "risk": round(float(book["risk"].iloc[i]) * 100),
    } for i in range(len(book))]
    rows_by_name = {r["name"]: r for r in rows}

    n_today = len(book)
    date_min = str(df["AppointmentDay"].min().date())
    date_max = str(today.date())

    # One consistent set of numbers, all derived from the same base rate.
    # Base rate = actual no-show share in the file when the column exists,
    # otherwise the average predicted risk.
    base_rate = float(df["NoShow"].mean()) if "NoShow" in df.columns else float(book["risk"].mean())
    model_rate = expected / len(calls) if len(calls) else 0.0
    vs_random = (model_rate / base_rate - 1) * 100 if base_rate > 0 else 0.0
    # Monthly projection at 50 calls/day uses the top-50 catch rate,
    # not the (higher) top-N rate, so the claim stays honest.
    top50 = book.head(min(50, n_today))
    rate50 = float(top50["risk"].mean()) if len(top50) else 0.0
    monthly = (rate50 - base_rate) * 50 * VALUE_PER_APPT * 22  # 50 calls/day, 22 workdays

    return {
        "rows": rows,
        "all": all_people,
        "date": str(today.date()),
        "date_min": date_min,
        "date_max": date_max,
        "n_today": n_today,
        "expected_noshows": round(expected, 1),
        "base_pct": round(base_rate * 100),
        "model_pct": round(model_rate * 100),
        "vs_random": round(vs_random),
        "rate50_pct": round(rate50 * 100),
        "monthly": int(monthly // 1000 * 1000),
        "n_calls": len(calls),
        "expected": round(expected, 1),
        "at_stake": round(at_stake),
        "rescue_half": round(at_stake / 2),
        "rule_expected": round(rule_expected, 1),
        "edge": round(edge),
    }

# --- 3b. Check one patient (typed in by hand) ---
def check_one(age, lead, sms, scholarship, hipertension, diabetes,
              alcoholism, handcap, prior_rate, prior_apps, weekday, hour):
    row = pd.DataFrame([{
        "Age": age, "LeadTime": lead, "SMS_received": sms,
        "Scholarship": scholarship, "Hipertension": hipertension,
        "Diabetes": diabetes, "Alcoholism": alcoholism, "Handcap": handcap,
        "PriorNoShowRate": prior_rate / 100.0, "PriorApps": prior_apps,
        "AppWeekday": weekday, "SchedHour": hour,
    }])
    risk = float(model.predict_proba(row[FEATURES])[:, 1][0])
    return round(risk * 100)

# --- 4. The one page + the JSON API his Stitch design will call ---
from flask import jsonify

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    budget = 10
    if request.method == "POST":
        budget = int(request.form.get("budget", 10))
        f = request.files.get("book")
        if f and f.filename:
            result = make_call_list(f.read(), f.filename, budget)
    return render_template("dashboard.html")

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    f = request.files.get("book")
    budget = int(request.form.get("budget", 10))
    if not f or not f.filename:
        return jsonify({"error": "no file"}), 400
    return jsonify(make_call_list(f.read(), f.filename, budget))

@app.route("/api/demo")
def api_demo():
    with open("sample_clinic_day.csv", "rb") as f:
        return jsonify(make_call_list(f.read(), "sample_clinic_day.csv", 10))

@app.route("/api/check-one", methods=["POST"])
def api_check_one():
    d = request.get_json(force=True)
    return jsonify({"risk": check_one(
        int(d["age"]), int(d["lead"]), int(d["sms"]), int(d["scholarship"]),
        int(d["hipertension"]), int(d["diabetes"]), int(d["alcoholism"]),
        int(d["handcap"]), float(d["prior_rate"]), int(d["prior_apps"]),
        int(d["weekday"]), int(d["hour"]))})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)

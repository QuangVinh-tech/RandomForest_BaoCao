import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from src.predictor import Predictor  # noqa: E402

app = Flask(__name__)
app.json.ensure_ascii = False
predictor = Predictor()

METRICS = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
FEATURES = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
HISTORY_PATH = ROOT / "data" / "processed" / "history.csv"
HISTORY = (pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
           if HISTORY_PATH.exists() else pd.DataFrame())


def to_int(value, default, lo, hi):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


# ---------- Trang ----------
# ---------- Trang ----------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/predict")
def predict_page():
    return render_template("predict.html")


# ---------- API ----------
@app.get("/api/summary")
def summary():
    return jsonify({
        "matches": int(len(FEATURES)),
        "seasons": int(FEATURES["Season"].nunique()),
        "teams": len(predictor.teams),
        "accuracy": round(METRICS["accuracy"] * 100, 1),
        "baseline": round(METRICS["baseline"] * 100, 1),
        "test_season": METRICS["test_season"],
    })


@app.get("/api/teams")
def teams():
    return jsonify({
        "season": predictor.season,
        "data_until": predictor.last_date,
        "teams": predictor.teams,
    })


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    home, away = data.get("home"), data.get("away")

    if not home or not away:
        return jsonify({"error": "Thiếu đội chủ nhà hoặc đội khách"}), 400
    if home == away:
        return jsonify({"error": "Hai đội phải khác nhau"}), 400
    if home not in predictor.teams or away not in predictor.teams:
        return jsonify({"error": "Tên đội không hợp lệ"}), 400

    try:
        return jsonify(predictor.predict(home, away))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.get("/api/history")
def history():
    df = HISTORY.copy()
    if df.empty:
        return jsonify({"items": [], "total": 0, "pages": 1, "page": 1,
                        "accuracy": None, "seasons": []})

    seasons = sorted(df["Season"].unique())
    team = request.args.get("team")
    result = request.args.get("result")
    season = request.args.get("season")
    date_from = pd.to_datetime(request.args.get("from"), errors="coerce")
    date_to = pd.to_datetime(request.args.get("to"), errors="coerce")

    if team:
        df = df[(df["HomeTeam"] == team) | (df["AwayTeam"] == team)]
    if result in ("H", "D", "A"):
        df = df[df["FTR"] == result]
    if season:
        df = df[df["Season"] == season]
    if pd.notna(date_from):
        df = df[df["Date"] >= date_from]
    if pd.notna(date_to):
        df = df[df["Date"] <= date_to]

    df = df.sort_values("Date", ascending=False)
    total = len(df)
    per_page = to_int(request.args.get("per_page"), 10, 1, 50)
    pages = max(1, -(-total // per_page))
    page = to_int(request.args.get("page"), 1, 1, pages)
    acc = round(float(df["correct"].mean()) * 100, 1) if total else None

    chunk = df.iloc[(page - 1) * per_page: page * per_page]
    items = [{
        "date": r.Date.strftime("%d/%m/%Y"),
        "season": r.Season,
        "home": r.HomeTeam, "away": r.AwayTeam,
        "score": f"{int(r.FTHG)} - {int(r.FTAG)}",
        "actual": r.FTR, "pred": r.pred, "correct": bool(r.correct),
        "p_h": r.p_h, "p_d": r.p_d, "p_a": r.p_a,
    } for r in chunk.itertuples(index=False)]

    return jsonify({"items": items, "total": total, "pages": pages,
                    "page": page, "accuracy": acc, "seasons": seasons})


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, port=5000)
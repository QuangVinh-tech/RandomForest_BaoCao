import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402

from src import update_data  # noqa: E402
from src.predictor import Predictor  # noqa: E402

app = Flask(__name__)
app.json.ensure_ascii = False

PROCESSED = ROOT / "data" / "processed"
METRICS = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
HISTORY_PATH = PROCESSED / "history.csv"
HISTORY = (pd.read_csv(HISTORY_PATH, parse_dates=["Date"])
           if HISTORY_PATH.exists() else pd.DataFrame())

ORDER = ["H", "D", "A"]
MIN_MANUAL_GAP = 120   # giây giữa hai lần bấm cập nhật thủ công


def load_data():
    """Nạp (hoặc nạp lại) dữ liệu và bộ dự đoán từ các file đã xử lý."""
    global predictor, FEATURES, CLEAN
    predictor = Predictor()
    FEATURES = pd.read_csv(PROCESSED / "features.csv")
    CLEAN = pd.read_csv(PROCESSED / "matches_clean.csv", parse_dates=["Date"])


load_data()


def to_int(value, default, lo, hi):
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def eval_from_cm(cm, labels):
    """Tính accuracy, precision, recall, F1 từ ma trận nhầm lẫn (hàng = thực tế, cột = dự đoán)."""
    m = np.array(cm, dtype=float)
    idx = [list(labels).index(c) for c in ORDER]
    m = m[np.ix_(idx, idx)]
    total = float(m.sum())
    per_class, f1s = [], []
    for i, c in enumerate(ORDER):
        tp = m[i, i]
        n_pred = m[:, i].sum()
        n_real = m[i, :].sum()
        prec = tp / n_pred if n_pred else None
        rec = tp / n_real if n_real else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec and (prec + rec) else 0.0
        f1s.append(f1)
        per_class.append({
            "label": c,
            "precision": None if prec is None else round(float(prec) * 100, 1),
            "recall": round(float(rec) * 100, 1),
            "f1": round(float(f1) * 100, 1),
            "support": int(n_real),
            "predicted": int(n_pred),
        })
    return {
        "n": int(total),
        "accuracy": round(float(np.trace(m)) / total * 100, 1),
        "baseline": round(float(m[0, :].sum()) / total * 100, 1),
        "macro_f1": round(float(np.mean(f1s)), 3),
        "per_class": per_class,
        "confusion": m.astype(int).tolist(),
        "labels": ORDER,
    }


def build_eval():
    sets, per_season = {}, []

    h = eval_from_cm(METRICS["confusion_matrix"], METRICS["labels"])
    h["title"] = f"Holdout {METRICS['test_season']}"
    h["desc"] = (f"Holdout: huấn luyện trên {METRICS['n_train']} trận của các mùa trước, "
                 f"kiểm tra trên {METRICS['n_test']} trận mùa {METRICS['test_season']}.")
    sets["holdout"] = h

    if not HISTORY.empty:
        cm = (pd.crosstab(HISTORY["FTR"], HISTORY["pred"])
              .reindex(index=ORDER, columns=ORDER, fill_value=0))
        w = eval_from_cm(cm.values.tolist(), ORDER)
        for season, g in HISTORY.groupby("Season"):
            per_season.append({
                "season": season,
                "n": int(len(g)),
                "accuracy": round(float(g["correct"].mean()) * 100, 1),
                "baseline": round(float((g["FTR"] == "H").mean()) * 100, 1),
            })
        w["title"] = f"Walk-forward ({len(per_season)} mùa gộp)"
        w["desc"] = ("Walk-forward: mỗi mùa được dự đoán bởi mô hình chỉ huấn luyện trên "
                     f"các mùa trước đó, gộp {w['n']} trận của {len(per_season)} mùa.")
        sets["walkforward"] = w

    return {"sets": sets, "per_season": per_season}


EVAL = build_eval()


# ---------- Trang ----------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/predict")
def predict_page():
    return render_template("predict.html")


@app.get("/stats")
def stats_page():
    return render_template("stats.html")


@app.get("/about")
def about_page():
    return render_template("about.html")


# ---------- API ----------
@app.get("/api/summary")
def summary():
    return jsonify({
        "matches": int(len(FEATURES)),
        "raw_matches": int(len(CLEAN)),
        "seasons": int(FEATURES["Season"].nunique()),
        "season_list": sorted(FEATURES["Season"].unique().tolist()),
        "date_from": CLEAN["Date"].min().strftime("%d/%m/%Y"),
        "date_to": CLEAN["Date"].max().strftime("%d/%m/%Y"),
        "teams": len(predictor.teams),
        "accuracy": round(METRICS["accuracy"] * 100, 1),
        "baseline": round(METRICS["baseline"] * 100, 1),
        "test_season": METRICS["test_season"],
        "n_train": METRICS["n_train"],
        "n_test": METRICS["n_test"],
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

    match_date = None
    if data.get("date"):
        d = pd.to_datetime(data["date"], errors="coerce")
        if pd.isna(d):
            return jsonify({"error": "Ngày thi đấu không hợp lệ"}), 400
        match_date = d.strftime("%d/%m/%Y")

    try:
        result = predictor.predict(home, away)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    result["match_date"] = match_date
    return jsonify(result)


@app.get("/api/metrics")
def metrics():
    return jsonify(METRICS)


@app.get("/api/eval")
def evaluation():
    return jsonify(EVAL)


@app.get("/api/status")
def status():
    s = update_data.read_status()
    return jsonify({
        "data_until": predictor.last_date,
        "season": predictor.season,
        "last_check": s.get("last_check"),
        "last_success": s.get("last_success"),
        "message": s.get("message", ""),
        "ok": s.get("ok"),
    })


@app.post("/api/update")
def update_now():
    s = update_data.read_status()
    last = s.get("last_check")
    if last:
        try:
            age = (datetime.now() - datetime.fromisoformat(last)).total_seconds()
        except ValueError:
            age = 1e9
        if age < MIN_MANUAL_GAP:
            wait = int(MIN_MANUAL_GAP - age)
            return jsonify({"error": f"Vừa kiểm tra cách đây ít phút. Hãy thử lại sau {wait} giây."}), 429

    result = update_data.update_all()
    if result.get("busy"):
        return jsonify({"error": result["message"]}), 409
    if result.get("changed"):
        load_data()
    if not result.get("ok"):
        return jsonify({"error": result.get("message", "Cập nhật thất bại")}), 502
    return jsonify(result)


@app.get("/api/upcoming")
def upcoming():
    limit = to_int(request.args.get("limit"), 8, 1, 20)
    path = update_data.FIXTURES_PATH
    if not path.exists():
        return jsonify({"items": [], "data_until": predictor.last_date,
                        "note": "Chưa có lịch thi đấu. Bấm “Cập nhật dữ liệu” khi có kết nối mạng."})

    df = pd.read_csv(path, parse_dates=["Date"])
    now_uk = datetime.now(update_data.UK_TZ)

    rows = []
    for r in df.itertuples(index=False):
        if r.HomeTeam not in predictor.teams or r.AwayTeam not in predictor.teams:
            continue
        t = getattr(r, "Time", None)
        kickoff = None
        if isinstance(t, str) and ":" in t:
            try:
                kickoff = update_data.uk_to_vn(r.Date, t)
            except ValueError:
                kickoff = None
        if kickoff is not None:
            if kickoff < now_uk:
                continue
            sort_key = kickoff
        else:
            if r.Date.date() < now_uk.date():
                continue
            sort_key = datetime(r.Date.year, r.Date.month, r.Date.day, tzinfo=update_data.UK_TZ)
        rows.append((sort_key, kickoff, r))
    rows.sort(key=lambda x: x[0])

    items = []
    for _, kickoff, r in rows:
        try:
            p = predictor.predict(r.HomeTeam, r.AwayTeam)
        except ValueError:
            continue
        items.append({
            "date": (kickoff if kickoff is not None else r.Date).strftime("%d/%m/%Y"),
            "time": kickoff.strftime("%H:%M") if kickoff is not None else None,
            "home": r.HomeTeam, "away": r.AwayTeam,
            "home_win": p["home_win"], "draw": p["draw"], "away_win": p["away_win"],
        })
        if len(items) >= limit:
            break

    note = "" if items else "Hiện chưa có trận sắp diễn ra trong lịch đã tải."
    return jsonify({"items": items, "data_until": predictor.last_date, "note": note})


@app.get("/api/history")
def history():
    df = HISTORY.copy()
    if df.empty:
        return jsonify({"items": [], "total": 0, "pages": 1, "page": 1,
                        "accuracy": None, "seasons": [], "teams": []})

    seasons = sorted(df["Season"].unique())
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
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
                    "page": page, "accuracy": acc, "seasons": seasons,
                    "teams": teams})


if __name__ == "__main__":
    # Chỉ tự cập nhật khi chạy trực tiếp (không chạy khi kiểm thử).
    # Đặt EPL_AUTO_UPDATE=0 để tắt, ví dụ khi demo không có mạng.
    if os.environ.get("EPL_AUTO_UPDATE", "1") != "0":
        update_data.start_background(on_change=load_data)
    app.run(debug=True, use_reloader=False, port=5000)
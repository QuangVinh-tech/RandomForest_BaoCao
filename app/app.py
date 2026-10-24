import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request  # noqa: E402

from src.predictor import Predictor  # noqa: E402

app = Flask(__name__)
app.json.ensure_ascii = False
predictor = Predictor()


@app.get("/")
def index():
    return "EPL Predictor API đang chạy. Thử /api/teams và /api/metrics"


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
    path = ROOT / "models" / "metrics.json"
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    app.run(debug=False, port=5000)
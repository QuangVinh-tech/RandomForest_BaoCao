from collections import defaultdict, deque
from pathlib import Path

import joblib
import pandas as pd

from src.features import HFA, K, MIN_HIST, N, START_ELO, form, points

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "rf_epl.joblib"
DATA_PATH = ROOT / "data" / "processed" / "matches_clean.csv"


class Predictor:
    def __init__(self):
        bundle = joblib.load(MODEL_PATH)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.classes = list(self.model.classes_)   # ['A', 'D', 'H']

        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        self.hist = defaultdict(lambda: deque(maxlen=N))
        self.elo = defaultdict(lambda: START_ELO)
        for m in df.itertuples(index=False):
            h, a = m.HomeTeam, m.AwayTeam
            self.hist[h].append((points(m.FTHG, m.FTAG), m.FTHG, m.FTAG, m.HST))
            self.hist[a].append((points(m.FTAG, m.FTHG), m.FTAG, m.FTHG, m.AST))
            exp_h = 1 / (1 + 10 ** ((self.elo[a] - self.elo[h] - HFA) / 400))
            s_h = 1 if m.FTR == "H" else (0.5 if m.FTR == "D" else 0)
            delta = K * (s_h - exp_h)
            self.elo[h] += delta
            self.elo[a] -= delta

        # Chỉ cho chọn các đội đã thi đấu ở mùa mới nhất trong dữ liệu
        last_season = df["Season"].iloc[-1]
        self.season = last_season
        self.last_date = df["Date"].max().strftime("%d/%m/%Y")
        self.teams = sorted(df[df["Season"] == last_season]["HomeTeam"].unique())

    def _recent(self, team):
        letters = {3: "W", 1: "D", 0: "L"}
        return [letters[x[0]] for x in self.hist[team]]

    def predict(self, home, away):
        hh, ha = self.hist[home], self.hist[away]
        if len(hh) < MIN_HIST or len(ha) < MIN_HIST:
            raise ValueError("Không đủ dữ liệu lịch sử cho một trong hai đội")

        feat = {}
        feat.update(form(hh, "h"))
        feat.update(form(ha, "a"))
        feat["h_elo"] = self.elo[home]
        feat["a_elo"] = self.elo[away]
        feat["elo_diff"] = self.elo[home] - self.elo[away]
        feat["pts_diff"] = feat["h_pts"] - feat["a_pts"]
        feat["gd_diff"] = ((feat["h_gf"] - feat["h_ga"])
                           - (feat["a_gf"] - feat["a_ga"]))

        X = pd.DataFrame([feat])[self.features]
        proba = dict(zip(self.classes, self.model.predict_proba(X)[0]))
        return {
            "home": home,
            "away": away,
            "home_win": round(float(proba["H"]) * 100, 1),
            "draw": round(float(proba["D"]) * 100, 1),
            "away_win": round(float(proba["A"]) * 100, 1),
            "home_form": self._recent(home),
            "away_form": self._recent(away),
            "home_elo": round(self.elo[home]),
            "away_elo": round(self.elo[away]),
        }
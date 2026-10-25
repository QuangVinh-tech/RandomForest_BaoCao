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
        self.matches = df

        # Mỗi phần tử: (điểm, bàn thắng, bàn thua, sút trúng đích, tổng cú sút, phạt góc, thẻ vàng)
        self.hist = defaultdict(lambda: deque(maxlen=N))
        self.elo = defaultdict(lambda: START_ELO)
        for m in df.itertuples(index=False):
            h, a = m.HomeTeam, m.AwayTeam
            self.hist[h].append((points(m.FTHG, m.FTAG), m.FTHG, m.FTAG, m.HST, m.HS, m.HC, m.HY))
            self.hist[a].append((points(m.FTAG, m.FTHG), m.FTAG, m.FTHG, m.AST, m.AS, m.AC, m.AY))
            exp_h = 1 / (1 + 10 ** ((self.elo[a] - self.elo[h] - HFA) / 400))
            s_h = 1 if m.FTR == "H" else (0.5 if m.FTR == "D" else 0)
            delta = K * (s_h - exp_h)
            self.elo[h] += delta
            self.elo[a] -= delta

        last_season = df["Season"].iloc[-1]
        self.season = last_season
        self.last_date = df["Date"].max().strftime("%d/%m/%Y")
        self.teams = sorted(df[df["Season"] == last_season]["HomeTeam"].unique())

    def _recent(self, team):
        letters = {3: "W", 1: "D", 0: "L"}
        return [letters[x[0]] for x in self.hist[team]]

    def _stats(self, team):
        hist = self.hist[team]
        n = len(hist)

        def avg(i):
            return round(float(sum(x[i] for x in hist)) / n, 2)

        return {"pts": avg(0), "gf": avg(1), "ga": avg(2), "sot": avg(3),
                "shots": avg(4), "corners": avg(5), "yellows": avg(6),
                "wins": int(sum(1 for x in hist if x[0] == 3))}

    def h2h(self, home, away, n=5):
        """Lịch sử đối đầu trong dữ liệu, tính theo góc nhìn của đội được chọn làm chủ nhà."""
        m = self.matches
        sel = m[((m["HomeTeam"] == home) & (m["AwayTeam"] == away)) |
                ((m["HomeTeam"] == away) & (m["AwayTeam"] == home))]
        sel = sel.sort_values("Date", ascending=False)

        hw = dw = aw = hg = ag = 0
        last = []
        for r in sel.itertuples(index=False):
            if r.HomeTeam == home:
                gf, ga = int(r.FTHG), int(r.FTAG)
            else:
                gf, ga = int(r.FTAG), int(r.FTHG)
            hg += gf
            ag += ga
            if gf > ga:
                hw += 1
                winner = "home"
            elif gf < ga:
                aw += 1
                winner = "away"
            else:
                dw += 1
                winner = "draw"
            if len(last) < n:
                last.append({"date": r.Date.strftime("%d/%m/%Y"),
                             "home": r.HomeTeam, "away": r.AwayTeam,
                             "score": f"{int(r.FTHG)} - {int(r.FTAG)}",
                             "winner": winner})
        return {"total": int(len(sel)), "home_wins": hw, "draws": dw,
                "away_wins": aw, "home_goals": hg, "away_goals": ag, "last": last}

    def predict(self, home, away):
        thieu = [t for t in (home, away) if len(self.hist[t]) < MIN_HIST]
        if thieu:
            raise ValueError(
                f"Không đủ dữ liệu lịch sử cho {', '.join(thieu)} "
                f"(cần ít nhất {MIN_HIST} trận gần nhất).")

        hh, ha = self.hist[home], self.hist[away]
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
            "home_stats": self._stats(home),
            "away_stats": self._stats(away),
            "h2h": self.h2h(home, away),
            "season": self.season,
            "data_until": self.last_date,
        }
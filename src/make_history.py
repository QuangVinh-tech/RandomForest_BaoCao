import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES_CSV = Path("data/processed/features.csv")
CLEAN_CSV = Path("data/processed/matches_clean.csv")
OUT = Path("data/processed/history.csv")
METRICS = Path("models/metrics.json")
NON = ["Date", "Season", "HomeTeam", "AwayTeam", "FTR", "FTHG", "FTAG"]


def main():
    params = json.loads(METRICS.read_text(encoding="utf-8"))["params"]
    df = (pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
          .sort_values("Date").reset_index(drop=True))
    clean = pd.read_csv(CLEAN_CSV, parse_dates=["Date"])[
        ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    df = df.merge(clean, on=["Date", "HomeTeam", "AwayTeam"], how="left")
    features = [c for c in df.columns if c not in NON]

    seasons = sorted(df["Season"].unique())
    parts = []
    for i in range(1, len(seasons)):
        tr = df[df["Season"].isin(seasons[:i])]
        te = df[df["Season"] == seasons[i]].copy()
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
        rf.fit(tr[features], tr["FTR"])
        proba = rf.predict_proba(te[features])
        cls = list(rf.classes_)
        te["pred"] = rf.predict(te[features])
        te["p_h"] = (proba[:, cls.index("H")] * 100).round(1)
        te["p_d"] = (proba[:, cls.index("D")] * 100).round(1)
        te["p_a"] = (proba[:, cls.index("A")] * 100).round(1)
        parts.append(te[["Date", "Season", "HomeTeam", "AwayTeam",
                         "FTHG", "FTAG", "FTR", "pred", "p_h", "p_d", "p_a"]])

    out = pd.concat(parts).sort_values("Date")
    out["correct"] = (out["pred"] == out["FTR"]).astype(int)
    out.to_csv(OUT, index=False)
    print("So tran trong lich su:", len(out))
    print(out.groupby("Season")["correct"].mean().round(3))
    print("Da luu:", OUT)


if __name__ == "__main__":
    main()
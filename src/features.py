from collections import defaultdict, deque
from pathlib import Path
import pandas as pd

IN = Path("data/processed/matches_clean.csv")
OUT = Path("data/processed/features.csv")

N = 5            # số trận gần nhất
MIN_HIST = 3     # cần ít nhất 3 trận trước đó mới tạo đặc trưng
K = 20           # hệ số cập nhật Elo
HFA = 60         # lợi thế sân nhà trong Elo
START_ELO = 1500


def points(gf, ga):
    return 3 if gf > ga else (1 if gf == ga else 0)


def form(hist, prefix):
    n = len(hist)
    return {
        f"{prefix}_pts": sum(x[0] for x in hist) / n,
        f"{prefix}_gf": sum(x[1] for x in hist) / n,
        f"{prefix}_ga": sum(x[2] for x in hist) / n,
        f"{prefix}_sot": sum(x[3] for x in hist) / n,
        f"{prefix}_wins": sum(1 for x in hist if x[0] == 3),
    }


def main():
    df = pd.read_csv(IN, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    hist = defaultdict(lambda: deque(maxlen=N))
    elo = defaultdict(lambda: START_ELO)
    rows = []

    for m in df.itertuples(index=False):
        h, a = m.HomeTeam, m.AwayTeam

        # 1) TẠO ĐẶC TRƯNG từ lịch sử TRƯỚC trận này
        if len(hist[h]) >= MIN_HIST and len(hist[a]) >= MIN_HIST:
            feat = {"Date": m.Date, "Season": m.Season,
                    "HomeTeam": h, "AwayTeam": a}
            feat.update(form(hist[h], "h"))
            feat.update(form(hist[a], "a"))
            feat["h_elo"] = elo[h]
            feat["a_elo"] = elo[a]
            feat["elo_diff"] = elo[h] - elo[a]
            feat["pts_diff"] = feat["h_pts"] - feat["a_pts"]
            feat["gd_diff"] = ((feat["h_gf"] - feat["h_ga"])
                               - (feat["a_gf"] - feat["a_ga"]))
            feat["FTR"] = m.FTR
            rows.append(feat)

        # 2) CHỈ SAU ĐÓ mới cập nhật lịch sử và Elo bằng kết quả trận này
        hist[h].append((points(m.FTHG, m.FTAG), m.FTHG, m.FTAG, m.HST))
        hist[a].append((points(m.FTAG, m.FTHG), m.FTAG, m.FTHG, m.AST))

        exp_h = 1 / (1 + 10 ** ((elo[a] - elo[h] - HFA) / 400))
        s_h = 1 if m.FTR == "H" else (0.5 if m.FTR == "D" else 0)
        delta = K * (s_h - exp_h)
        elo[h] += delta
        elo[a] -= delta

    out = pd.DataFrame(rows)
    print("Tong so tran goc:", len(df))
    print("So tran co dac trung:", len(out), "| bi bo:", len(df) - len(out))
    print("So gia tri thieu:", int(out.isna().sum().sum()))
    print("\nPhan bo nhan:")
    print(out["FTR"].value_counts(normalize=True).round(3))
    print("\n5 dong dau:")
    print(out.head().T)
    print("\nElo cao nhat cuoi du lieu:")
    last = sorted(elo.items(), key=lambda kv: -kv[1])[:5]
    print(last)

    out.to_csv(OUT, index=False)
    print("\nDa luu:", OUT)


if __name__ == "__main__":
    main()
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/matches_clean.csv")

COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
        "HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]


def load_season(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    df = df[[c for c in COLS if c in df.columns]].copy()
    df["Season"] = path.stem.split("_")[1]
    return df


def main():
    files = sorted(RAW.glob("E0_*.csv"))
    if not files:
        raise SystemExit("Khong tim thay file E0_*.csv trong data/raw")

    df = pd.concat([load_season(f) for f in files], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")

    for c in ["HomeTeam", "AwayTeam"]:
        df[c] = df[c].str.strip()

    df = df.dropna(subset=["Date", "FTHG", "FTAG", "FTR"])
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)
    df = df.sort_values("Date").reset_index(drop=True)

    print("Kich thuoc:", df.shape)
    print("\nSo tran moi mua:\n", df["Season"].value_counts().sort_index())
    print("\nPhan bo ket qua (H=chu nha thang, D=hoa, A=khach thang):")
    print(df["FTR"].value_counts(normalize=True).round(3))
    print("\nGia tri thieu:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nSo doi:", df["HomeTeam"].nunique())
    print(sorted(df["HomeTeam"].unique()))

    df.to_csv(OUT, index=False)
    print("\nDa luu:", OUT)


if __name__ == "__main__":
    main()

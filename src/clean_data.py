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
        raise SystemExit("Không tìm thấy file E0_*.csv trong data/raw")

    df = pd.concat([load_season(f) for f in files], ignore_index=True)

   
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
    start = df["Date"].dt.year.where(df["Date"].dt.month >= 7, df["Date"].dt.year - 1)
    df["Season"] = start.astype(str) + "/" + ((start + 1) % 100).astype(str).str.zfill(2)


    for c in ["HomeTeam", "AwayTeam"]:
        df[c] = df[c].str.strip()

  
    df = df.dropna(subset=["Date", "FTHG", "FTAG", "FTR"])
    df[["FTHG", "FTAG"]] = df[["FTHG", "FTAG"]].astype(int)

    df = df.sort_values("Date").reset_index(drop=True)

   
    print("Kích thước:", df.shape)
    print("\nSố trận mỗi mùa:\n", df["Season"].value_counts().sort_index())
    print("\nPhân bố kết quả (H=chủ nhà thắng, D=hòa, A=khách thắng):")
    print(df["FTR"].value_counts(normalize=True).round(3))
    print("\nGiá trị thiếu:\n", df.isna().sum()[df.isna().sum() > 0])
    print("\nSố đội:", df["HomeTeam"].nunique())
    print(sorted(df["HomeTeam"].unique()))

    df.to_csv(OUT, index=False)
    print("\nĐã lưu:", OUT)


if __name__ == "__main__":
    main()
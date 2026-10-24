import json
from itertools import product
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, log_loss)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

DATA = Path("data/processed/features.csv")
MODEL = Path("models/rf_epl.joblib")
METRICS = Path("models/metrics.json")

NON_FEATURES = ["Date", "Season", "HomeTeam", "AwayTeam", "FTR"]
TEST_SEASON = "2024/25"
SEED = 42


def make_rf(**params):
    return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)


def main():
    df = pd.read_csv(DATA, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    features = [c for c in df.columns if c not in NON_FEATURES]

    train = df[df["Season"] != TEST_SEASON]
    test = df[df["Season"] == TEST_SEASON]
    X_tr, y_tr = train[features], train["FTR"]
    X_te, y_te = test[features], test["FTR"]
    print(f"Train: {len(train)} tran | Test ({TEST_SEASON}): {len(test)} tran")
    print(f"So dac trung: {len(features)}")

    baseline = accuracy_score(y_te, ["H"] * len(y_te))
    print(f"\nMoc co so (luon doan H): {baseline:.3f}")

    grid = list(product(["gini", "entropy"], [3, 4, 6, None], [10, 20, 30, 50]))
    cv = TimeSeriesSplit(n_splits=5)
    results = []
    for criterion, depth, leaf in grid:
        rf = make_rf(n_estimators=300, criterion=criterion,
                     max_depth=depth, min_samples_leaf=leaf)
        score = cross_val_score(rf, X_tr, y_tr, cv=cv,
                                scoring="neg_log_loss").mean()
        results.append((score, criterion, depth, leaf))
    results.sort(key=lambda r: -r[0])
    print("\nTop 5 cau hinh theo CV (neg_log_loss, cang gan 0 cang tot):")
    for s, c, d, l in results[:5]:
        print(f"  {s:.3f} | criterion={c}, max_depth={d}, min_samples_leaf={l}")

    _, best_c, best_d, best_l = results[0]
    best = dict(n_estimators=300, criterion=best_c,
                max_depth=best_d, min_samples_leaf=best_l)
    print("\nCau hinh duoc chon:", best)


    model = make_rf(**best).fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    ll = log_loss(y_te, proba, labels=model.classes_)
    labels = list(model.classes_)         
    cm = confusion_matrix(y_te, pred, labels=labels)

    print(f"\n=== KET QUA HOLDOUT ({TEST_SEASON}) ===")
    print(f"Accuracy : {acc:.3f}  (moc co so {baseline:.3f})")
    print(f"F1 macro : {f1:.3f}")
    print(f"Log loss : {ll:.3f}")
    print("\nConfusion matrix (hang = thuc te, cot = du doan), thu tu", labels)
    print(cm)
    print("\n", classification_report(y_te, pred, zero_division=0))

    imp = (pd.Series(model.feature_importances_, index=features)
           .sort_values(ascending=False))
    print("Top 10 dac trung quan trong:")
    print(imp.head(10).round(3))

    seasons = sorted(df["Season"].unique())
    per_season = {}
    for i in range(1, len(seasons)):
        tr = df[df["Season"].isin(seasons[:i])]
        te = df[df["Season"] == seasons[i]]
        m = make_rf(**best).fit(tr[features], tr["FTR"])
        per_season[seasons[i]] = float(accuracy_score(te["FTR"], m.predict(te[features])))
    print("\nAccuracy walk-forward theo mua:")
    for s, a in per_season.items():
        print(f"  {s}: {a:.3f}")

    final = make_rf(**best).fit(df[features], df["FTR"])
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": features}, MODEL)

    metrics = {
        "test_season": TEST_SEASON,
        "n_train": int(len(train)), "n_test": int(len(test)),
        "accuracy": float(acc), "baseline": float(baseline),
        "f1_macro": float(f1), "log_loss": float(ll),
        "labels": labels, "confusion_matrix": cm.tolist(),
        "per_season_accuracy": per_season,
        "feature_importance": {k: float(v) for k, v in imp.head(15).items()},
        "params": {k: (None if v is None else v) for k, v in best.items()},
    }
    METRICS.write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    print("\nDa luu:", MODEL, "va", METRICS)


if __name__ == "__main__":
    main()
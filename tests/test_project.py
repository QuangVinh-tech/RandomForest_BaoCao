import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---------- Hàm tạo đặc trưng ----------
def test_points():
    from src.features import points
    assert points(2, 0) == 3
    assert points(1, 1) == 1
    assert points(0, 3) == 0


def test_form_tinh_dung_trung_binh():
    from src.features import form
    hist = [(3, 2, 0, 5), (0, 0, 1, 2)]     
    f = form(hist, "h")
    assert f["h_pts"] == 1.5
    assert f["h_gf"] == 1.0
    assert f["h_ga"] == 0.5
    assert f["h_sot"] == 3.5
    assert f["h_wins"] == 1


# ---------- Dữ liệu ----------
def test_features_csv_hop_le():
    df = pd.read_csv(ROOT / "data" / "processed" / "features.csv")
    assert len(df) > 1500
    assert df.isna().sum().sum() == 0
    assert set(df["FTR"].unique()) <= {"H", "D", "A"}
    assert df["Season"].nunique() == 7


def test_khong_ro_ri_thoi_gian():
    """Mỗi đội phải có đủ lịch sử TRƯỚC trận được dự đoán."""
    clean = pd.read_csv(ROOT / "data" / "processed" / "matches_clean.csv",
                        parse_dates=["Date"])
    feats = pd.read_csv(ROOT / "data" / "processed" / "features.csv",
                        parse_dates=["Date"])
    first = clean["Date"].min()
    
    assert feats["Date"].min() > first


# ---------- Mô hình ----------
@pytest.fixture(scope="module")
def predictor():
    from src.predictor import Predictor
    return Predictor()


def test_xac_suat_cong_lai_bang_100(predictor):
    r = predictor.predict("Liverpool", "Chelsea")
    total = r["home_win"] + r["draw"] + r["away_win"]
    assert abs(total - 100) < 0.5
    assert len(r["home_form"]) == 5


def test_doi_la_ngoai_le(predictor):
    with pytest.raises(ValueError):
        predictor.predict("Doi Khong Ton Tai", "Chelsea")


# ---------- API ----------
@pytest.fixture(scope="module")
def client():
    import sys
    spec = importlib.util.spec_from_file_location("epl_app", ROOT / "app" / "app.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["epl_app"] = module          
    spec.loader.exec_module(module)
    module.app.config["TESTING"] = True
    return module.app.test_client()

@pytest.mark.parametrize("path", ["/", "/predict", "/stats", "/about"])
def test_cac_trang_mo_duoc(client, path):
    assert client.get(path).status_code == 200


def test_api_teams(client):
    data = client.get("/api/teams").get_json()
    assert len(data["teams"]) == 20


def test_api_predict_hop_le(client):
    r = client.post("/api/predict", json={"home": "Liverpool", "away": "Chelsea"})
    assert r.status_code == 200
    assert {"home_win", "draw", "away_win"} <= set(r.get_json())


@pytest.mark.parametrize("body", [
    {"home": "Liverpool", "away": "Liverpool"},     
    {"home": "Liverpool"},                         
    {"home": "Abc", "away": "Chelsea"},             
    {},                                            
])
def test_api_predict_dau_vao_sai(client, body):
    r = client.post("/api/predict", json=body)
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_api_history_phan_trang(client):
    data = client.get("/api/history?page=1&per_page=5").get_json()
    assert len(data["items"]) <= 5
    assert data["total"] >= 0

def test_api_eval_nhat_quan(client):
    d = client.get("/api/eval").get_json()
    m = client.get("/api/metrics").get_json()
    h = d["sets"]["holdout"]
    assert abs(h["accuracy"] - round(m["accuracy"] * 100, 1)) < 0.1
    assert h["n"] == m["n_test"]
    for s in d["sets"].values():
        assert sum(sum(r) for r in s["confusion"]) == s["n"]

def test_h2h_va_thong_ke_mo_rong(predictor):
    r = predictor.predict("Liverpool", "Chelsea")
    h = r["h2h"]
    assert h["home_wins"] + h["draws"] + h["away_wins"] == h["total"]
    assert len(h["last"]) <= 5
    assert {"shots", "corners", "yellows"} <= set(r["home_stats"])

def test_season_code():
    from datetime import date
    from src.update_data import season_code
    assert season_code(date(2026, 10, 25)) == "2627"
    assert season_code(date(2027, 3, 1)) == "2627"
    assert season_code(date(2026, 6, 30)) == "2526"


def test_api_status_va_upcoming(client):
    s = client.get("/api/status")
    assert s.status_code == 200
    assert "data_until" in s.get_json()
    u = client.get("/api/upcoming")
    assert u.status_code == 200
    assert isinstance(u.get_json()["items"], list)
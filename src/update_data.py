"""Cập nhật dữ liệu từ Football-Data.co.uk: kết quả mùa hiện tại và lịch thi đấu sắp tới."""
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
STATUS_PATH = PROCESSED / "update_status.json"
FIXTURES_PATH = PROCESSED / "fixtures_E0.csv"

BASE = "https://www.football-data.co.uk"
FIXTURES_URL = f"{BASE}/fixtures.csv"
USER_AGENT = "Mozilla/5.0 (EPL-Predictor student project)"
REQUIRED = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
LOCK = threading.Lock()


def season_code(today=None):
    """Mã mùa của Football-Data: tháng 7 trở đi thuộc mùa bắt đầu năm đó (2026-10 -> '2627')."""
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def download(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def read_csv_bytes(raw):
    return pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", encoding_errors="replace")


# ---------- Trạng thái ----------
def read_status():
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(**fields):
    status = read_status()
    status.update(fields)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    return status


# ---------- Tải kết quả ----------
def count_played(df):
    return int(df["FTR"].notna().sum()) if "FTR" in df.columns else 0


def update_results():
    """Tải E0.csv của mùa hiện tại. Trả về số trận mới (0 nếu không có gì mới)."""
    code = season_code()
    raw = download(f"{BASE}/mmz4281/{code}/E0.csv")
    df = read_csv_bytes(raw)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"File tải về thiếu cột {sorted(missing)}")
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])
    played_new = count_played(df)

    target = RAW / f"E0_{code}.csv"
    played_old = 0
    if target.exists():
        try:
            old = pd.read_csv(target, encoding="utf-8-sig", encoding_errors="replace")
            played_old = count_played(old.dropna(subset=["HomeTeam", "AwayTeam"]))
        except Exception:
            played_old = 0

    if played_new < played_old:
        raise ValueError(f"File tải về có ít trận hơn bản hiện có ({played_new} < {played_old}), giữ nguyên dữ liệu cũ")
    if played_new == played_old:
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / f"_tmp_E0_{code}.csv"
    tmp.write_bytes(raw)
    os.replace(tmp, target)
    return played_new - played_old


# ---------- Tải lịch thi đấu ----------
def update_fixtures():
    """Tải fixtures.csv, giữ lại các trận Premier League (Div = E0). Trả về số trận."""
    raw = download(FIXTURES_URL)
    df = read_csv_bytes(raw)
    if "Div" not in df.columns:
        raise ValueError("fixtures.csv không đúng định dạng")
    df = df[df["Div"] == "E0"].copy()
    keep = [c for c in ["Date", "Time", "HomeTeam", "AwayTeam"] if c in df.columns]
    df = df[keep].dropna(subset=["HomeTeam", "AwayTeam"])
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["Date"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURES_PATH, index=False, encoding="utf-8")
    return int(len(df))


# ---------- Dựng lại dữ liệu ----------
def needs_rebuild():
    clean = PROCESSED / "matches_clean.csv"
    feats = PROCESSED / "features.csv"
    if not clean.exists() or not feats.exists():
        return True
    raws = list(RAW.glob("E0_*.csv"))
    newest_raw = max((p.stat().st_mtime for p in raws), default=0)
    return newest_raw > clean.stat().st_mtime or clean.stat().st_mtime > feats.stat().st_mtime


def rebuild():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    for script in ("clean_data.py", "features.py"):
        r = subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"{script} lỗi: {r.stderr.strip()[-300:]}")


def update_all():
    if not LOCK.acquire(blocking=False):
        return {"ok": False, "busy": True, "message": "Đang có một lần cập nhật khác, vui lòng chờ."}
    try:
        now = datetime.now().isoformat(timespec="seconds")
        notes, ok, changed = [], True, False

        try:
            n = update_results()
            if n > 0:
                notes.append(f"Thêm {n} trận mới.")
            else:
                notes.append("Chưa có trận mới.")
            if n > 0 or needs_rebuild():
                rebuild()
                changed = True
        except Exception as e:
            ok = False
            notes.append(f"Không cập nhật được kết quả: {e}")

        try:
            k = update_fixtures()
            notes.append(f"Lịch thi đấu: {k} trận.")
        except Exception as e:
            notes.append(f"Không tải được lịch thi đấu: {e}")

        fields = {"last_check": now, "message": " ".join(notes), "ok": ok}
        if ok:
            fields["last_success"] = now
        status = write_status(**fields)
        status.update({"ok": ok, "changed": changed})
        return status
    finally:
        LOCK.release()


# ---------- Tự cập nhật nền ----------
def should_auto_update(max_age_hours=12, retry_hours=1):
    s = read_status()
    last = s.get("last_check")
    if not last:
        return True
    try:
        age = (datetime.now() - datetime.fromisoformat(last)).total_seconds() / 3600
    except ValueError:
        return True
    return age >= (max_age_hours if s.get("ok") else retry_hours)


def start_background(on_change, first_delay=5, interval=1800):
    def loop():
        time.sleep(first_delay)
        while True:
            try:
                if should_auto_update():
                    result = update_all()
                    if result.get("changed"):
                        on_change()
            except Exception as e:
                print("Tự cập nhật lỗi:", e)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
# EPL Predictor – Dự đoán kết quả Ngoại hạng Anh bằng Random Forest

Đồ án môn Khai phá dữ liệu. Ứng dụng web dự đoán xác suất thắng, hòa, thua của một trận đấu Ngoại hạng Anh dựa trên phong độ 5 trận gần nhất và điểm Elo của hai đội.

Tác giả: Vinh. Mã nguồn: https://github.com/QuangVinh-tech/RandomForest_BaoCao

## Kết quả

| Chỉ số | Giá trị |
|---|---|
| Dữ liệu | 7 mùa (2020/21 đến 2026/27, mùa cuối đang diễn ra), 2.350 trận, 2.289 trận có đặc trưng |
| Cách kiểm tra | Huấn luyện trên 5 mùa đầu (1.847 trận), kiểm tra trên mùa 2025/26 (377 trận) |
| Độ chính xác | 49,1% (mốc cơ sở luôn đoán chủ nhà thắng: 42,2%) |
| F1 macro / Log loss | 0,371 / 1,045 |
| Walk-forward theo mùa | 2021/22: 51,5%, 2022/23: 54,8%, 2023/24: 57,6%, 2024/25: 52,5%, 2025/26: 49,1%, 2026/27: 43,1% (mới 70 trận) |

Tham số mô hình (chọn bằng cross-validation theo thời gian, chấm theo log loss): `n_estimators=300`, `criterion=gini`, `max_depth=None`, `min_samples_leaf=30`.

Lưu ý:
- Mô hình chỉ hơn mốc cơ sở khoảng 7 điểm phần trăm ở mùa kiểm tra; với 377 trận, độ chính xác có sai số khoảng ±5 điểm phần trăm.
- Các chỉ số trên là của mô hình huấn luyện trên các mùa trước mùa kiểm tra. Mô hình dùng trên web (`models/rf_epl.joblib`) được huấn luyện lại trên toàn bộ dữ liệu.
- Mô hình không bao giờ dự đoán nhãn "Hòa" (recall bằng 0), nên web hiển thị cả ba xác suất thay vì chỉ nhãn.
- Kết quả chỉ mang tính tham khảo, không dùng để cá cược.

## Cấu trúc

```
data/raw/          CSV gốc từ Football-Data.co.uk
data/processed/    matches_clean.csv, features.csv, history.csv
src/               clean_data.py, features.py, train.py, make_history.py, predictor.py
models/            rf_epl.joblib, metrics.json
app/               Flask: app.py, templates/, static/
tests/             pytest (19 test)
```

## Cài đặt và chạy

Yêu cầu Python 3.10 trở lên (đã thử trên Python 3.14).

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app\app.py             # chạy web tại http://127.0.0.1:5000
```

Repo đã kèm sẵn dữ liệu đã xử lý và mô hình, nên chạy thẳng `python app\app.py` là dùng được. Kiểm thử: `python -m pytest`.

## Tái tạo toàn bộ kết quả

1. Tải file `E0.csv` của từng mùa từ Football-Data.co.uk (England, Premier League) vào `data/raw/`, đặt tên dạng `E0_2021.csv`, `E0_2122.csv`... Cột `Season` được tính từ ngày thi đấu (từ tháng 7 trở đi thuộc mùa bắt đầu năm đó), không phụ thuộc tên file.
2. Chạy lần lượt:

```powershell
python src\clean_data.py      # làm sạch, gộp, chuẩn hóa ngày và tên đội
python src\features.py        # tạo đặc trưng (chỉ dùng thông tin trước trận)
python src\train.py           # chọn tham số, đánh giá, lưu mô hình
python src\make_history.py    # lịch sử dự đoán walk-forward cho trang Thống kê
```

Muốn thêm mùa mới, tải thêm file `E0_xxxx.csv` rồi chạy lại bốn lệnh trên. Nếu thêm một mùa đã kết thúc, đổi `TEST_SEASON` trong `src/train.py` sang mùa đó.

## API

| Phương thức | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/api/summary` | Số liệu tổng quan cho trang chủ |
| GET | `/api/teams` | Danh sách đội mùa mới nhất |
| POST | `/api/predict` | Body `{"home": "...", "away": "..."}`, trả về xác suất, phong độ, chỉ số và lịch sử đối đầu |
| GET | `/api/metrics` | Chỉ số mô hình |
| GET | `/api/eval` | Accuracy, Precision, Recall, F1, ma trận nhầm lẫn theo từng tập đánh giá |
| GET | `/api/history` | Lịch sử dự đoán, có lọc và phân trang |

## Phương pháp

1. **Đặc trưng:** trung bình 5 trận gần nhất (điểm, bàn thắng, bàn thua, sút trúng đích, số trận thắng), điểm Elo và các cột chênh lệch. Đặc trưng của một trận được tạo trước khi cập nhật lịch sử bằng kết quả trận đó nên không bị rò rỉ dữ liệu.
2. **Chọn tham số:** cross-validation theo thời gian (`TimeSeriesSplit`), chấm theo log loss.
3. **Đánh giá:** holdout trên mùa 2025/26 và walk-forward qua các mùa, so với mốc cơ sở luôn đoán chủ nhà thắng.

## Nguồn dữ liệu và bản quyền

Dữ liệu: [Football-Data.co.uk](https://www.football-data.co.uk). Logo, hình ảnh và ảnh banner thuộc về chủ sở hữu tương ứng, chỉ dùng cho mục đích học tập.
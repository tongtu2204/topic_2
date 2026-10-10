PHƯƠNG PHÁP SỐ 4 — SETAR/TAR CHO USD/VND
=======================================

Mục tiêu: dự báo log-return một phiên kế tiếp bằng SETAR hai chế độ.
Đối chứng: Zero/Naive, Mean và AR tuyến tính trong cùng thí nghiệm.

Dữ liệu:
- Nguồn được cung cấp: data/raw/USD_VND_history.csv.
- Dữ liệu đã xử lý: data/processed/; giữ nguyên để tái lập thí nghiệm.
- Input bắt buộc: price_train.csv, price_validation.csv, price_test.csv,
  return_train.csv, return_validation.csv, return_test.csv.
- Train: 04/01/2016–31/12/2024; Validation: năm 2025;
  Test: 01/01/2026–31/08/2026, gồm 175 dự báo.
- Close là biến giá chính, log_return = 100 log(P_t/P_(t-1)).
- Các dòng không thỏa logic OHLC được giữ nguyên; kiểm tra lại trong
  results/model_4_setar_tar/data_audit.csv.

Chạy lại từ thư mục gốc repo:
  pip install -r requirements.txt
  python code/run_setar_tar_usd_vnd.py
  python -m unittest discover -s code -p 'test_setar_tar_usd_vnd.py'

Hoặc mở code/08_mo_hinh_4_setar_tar_usd_vnd.ipynb và Run All từ
thư mục gốc repo hoặc thư mục code. Các module .py phải đi kèm notebook.
Các file xử lý và chia tập của bước chuẩn bị không còn trong repo;
thí nghiệm hiện tại bắt đầu từ sáu file dữ liệu đã xử lý nêu trên.

Cấu trúc:
- code/: notebook và năm module Python của phương pháp 4.
- results/model_4_setar_tar/: CSV, cấu hình, môi trường và README kết quả.
- figures/model_4_setar_tar/: 11 biểu đồ của SETAR và các đối chứng.
  Giữ số thứ tự hình hiện có; không có hình số 10.
- report/Chuong_SETAR_TAR_USD_VND_temp.tex: bản LaTeX tạm một chương.

Đánh giá:
- Chọn p,d theo RMSE Validation; refit trên Train+Validation;
  tham số cố định khi chạy rolling one-step trên Test.
- MAE/MSE/RMSE, median/max absolute error, bias, R², MASE, rRMSE,
  WAPE; MAPE/sMAPE chỉ tính trên giá.
- Dự báo hướng: toàn bộ phiên, phiên có giá thực tế thay đổi và balanced recall.
- Khoảng dự báo kinh nghiệm 95%: coverage, độ rộng và interval score.
- Chẩn đoán residual/forecast error; CI bootstrap chênh lệch loss.
- Coverage 96% không phải tỷ lệ dự báo đúng hướng hay đúng tỷ giá.

Chi tiết: results/model_4_setar_tar/README.md.
File LaTeX chưa được biên dịch; repo không có PDF của chương này.

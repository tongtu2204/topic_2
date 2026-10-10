BỘ NOTEBOOK PHÂN TÍCH USD/VND
=============================

Dữ liệu gốc:
- data/raw/USD_VND_history.csv
- Khoảng thời gian: 04/01/2016 đến 31/08/2026

Thứ tự chạy notebook:
1. code/01_tien_xu_li_usd_vnd.ipynb
2. code/02_chia_du_lieu_huan_luyen_usd_vnd.ipynb
3. code/03_truc_quan_hoa_du_lieu_usd_vnd.ipynb
4. code/04_kiem_dinh_tinh_dung_usd_vnd.ipynb
5. code/07_mo_hinh_3_state_space_kalman_usd_vnd.ipynb
6. code/08_mo_hinh_4_setar_tar_usd_vnd.ipynb

Thiết kế chia dữ liệu:
- Train: 04/01/2016 đến 31/12/2024
- Validation: 01/01/2025 đến 31/12/2025
- Test: 01/01/2026 đến 31/08/2026

Cấu trúc thư mục:
- code/: notebook
- data/raw/: dữ liệu gốc
- data/processed/: dữ liệu đã xử lý và chia tập
- figures/: biểu đồ được notebook sinh ra
- results/: bảng kết quả và mô hình đã lưu

Ghi chú:
- Cột dữ liệu gốc tiếng Việt được chuẩn hóa thành Date, Close, Open, High, Low.
- Giá dạng chuỗi có dấu phẩy được chuyển sang số.
- Close được dùng làm biến giá chính cho mô hình.
- Notebook tiền xử lý chỉ gắn cờ các dòng không thỏa logic OHLC để kiểm tra nguồn, không tự động xóa.
- Tất cả 5 notebook đã được chạy thử thành công theo đúng thứ tự trên.

MÔ HÌNH SỐ 4 — SETAR/TAR
- Cài thư viện: pip install -r requirements.txt
- Chạy từ bất kỳ thư mục nào: python code/run_setar_tar_usd_vnd.py (đường dẫn tới script phải đúng).
- Input: log_return (%), cùng Train/Validation/Test như mô hình 3.
- Kết quả: results/model_4_setar_tar/README.md, CSV và cấu hình.
- Biểu đồ: figures/model_4_setar_tar/ (12 biểu đồ).
- Các chỉ số giá/return, dự báo hướng, khoảng dự báo, diagnostics và CI chênh lệch loss đều được lưu.
- MAPE/sMAPE chỉ tính trên giá; không áp dụng cho return có zero.
- Notebook 08 chạy đầy đủ; kiểm tra căn chỉnh lag và chống dùng trước dữ liệu bằng test_setar_tar_usd_vnd.py.

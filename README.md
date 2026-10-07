# QH25 — Chuyên đề 2: Chuỗi thời gian giá bạc XAG/USD

Dữ liệu ngày gồm 4.301 phiên từ 04/01/2010 đến 31/08/2026. Chuỗi return dùng đơn vị phần trăm: `100 * log(Close[t] / Close[t-1])`.

| Notebook | Nội dung |
|---|---|
| `code/01_tien_xu_li.ipynb` | Kiểm tra dữ liệu, tạo log-price và log-return |
| `code/02_chia_du_lieu_huan_luyen.ipynb` | Chia dữ liệu theo thời gian |
| `code/03_truc_quan_hoa_du_lieu.ipynb` | Thống kê mô tả và trực quan hóa train |
| `code/04_kiem_dinh_tinh_dung.ipynb` | ADF, ACF/PACF, Ljung–Box và ARCH-LM |
| `code/05_mo_hinh_setar_tar.ipynb` | Mục 4: SETAR / TAR, chọn cấu hình và đánh giá dự báo |

## Chạy mục 4

Từ thư mục gốc repo:

```powershell
python -m pip install -r requirements.txt
python code/run_setar_tar.py
```

Hoặc mở `code/05_mo_hinh_setar_tar.ipynb` rồi Run All. Các hàm mô hình nằm trong `code/setar_tar.py`; phần tạo đồ thị và ghi nhận xét nằm trong `code/setar_figures.py`.

Train kết thúc năm 2024; validation là năm 2025; test gồm 172 phiên năm 2026. Ngưỡng học trên train, cấu hình chọn bằng RMSE validation; sau đó refit train+validation và dự báo một bước trên test với dữ liệu đã biết đến phiên trước. Đối chứng: AR tuyến tính, trung bình return và return bằng 0.

Kết quả và nhận xét chi tiết: [results/setar_tar/README.md](results/setar_tar/README.md). Đồ thị: [figures/setar_tar/](figures/setar_tar/).

Các notebook 01/02 ban đầu vẫn dùng đường dẫn `D:/Chuyên đề 2/...`; khi chạy lại hai notebook này cần sửa đường dẫn cho máy hiện tại. Notebook 05 và script mục 4 dùng đường dẫn theo repo, đọc các CSV đã có sẵn.

## Kiểm tra cách dựng lag và dự báo

```powershell
python code/test_setar_tar.py
```

Kiểm tra độ trễ ngưỡng, trường hợp bằng ngưỡng, ước lượng hai chế độ và việc không dùng return hiện tại/tương lai trong dự báo hiện tại.

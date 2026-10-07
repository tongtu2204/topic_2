# Mục 4 — SETAR / TAR cho log-return của bạc XAG/USD

## Mô hình

TAR là họ mô hình tự hồi quy theo ngưỡng. SETAR là trường hợp biến ngưỡng lấy từ chính chuỗi trễ. Ở đây sử dụng SETAR hai chế độ, cùng bậc p và có hệ số chặn ở mỗi chế độ:

$$r_t=\begin{cases}c_L+\sum_{j=1}^{p}\phi_{L,j}r_{t-j}+\varepsilon_t,&r_{t-d}\le\gamma,\\c_H+\sum_{j=1}^{p}\phi_{H,j}r_{t-j}+\varepsilon_t,&r_{t-d}>\gamma.\end{cases}$$

Return tính theo phần trăm: `100 * log(Close[t] / Close[t-1])`. Low/High chỉ thể hiện giá trị return trễ ở dưới/trên ngưỡng; không phải nhãn thị trường tăng/giảm hay biến động thấp/cao.

## Giao thức thực nghiệm

- Train: 2010–2024; validation: 2025; test: 02/01/2026–31/08/2026.
- Dò `p=1..10`, `d=1..5`; lưới ngưỡng gồm 71 phân vị từ 15% đến 85% của biến ngưỡng trong tập ước lượng.
- Với từng `(p,d)`, chọn ngưỡng tối thiểu hóa SSR trên train; ước lượng hai phương trình bằng bình phương tối thiểu có điều kiện.
- Mỗi chế độ có ít nhất 15% quan sát và ít nhất `5*(p+1)` quan sát. Tất cả cấu hình dùng chung mốc khởi đầu lag 10.
- Chọn `(p,d)` bằng RMSE validation. AR tuyến tính cũng chọn p bằng cùng RMSE validation.
- Sau khi khóa p,d, ước lượng lại ngưỡng và hệ số trên train+validation; giữ cố định tham số trong test.
- Dự báo từng phiên một bước trước. Tại phiên t chỉ dùng dữ liệu đã biết đến t−1; return thực tế của phiên trước được cập nhật vào lag.
- Mean dự báo bằng trung bình tập ước lượng; Zero dự báo return bằng 0. Đây là các đối chứng riêng, không tham gia chọn cấu hình SETAR.
- RMSE/MAE tính bằng điểm phần trăm. Không dùng MAPE cho return vì return có thể bằng/gần 0.

## Cấu hình chọn được

- SETAR chọn trên validation: p=4, d=4, ngưỡng train = 1.072667%.
- Sau refit: ngưỡng = 1.050277%; số quan sát fit: 4,118; Low=3,212, High=906.
- AR đối chứng: p=3.

## Kết quả dự báo

| Tập | Mô hình | n | RMSE | MAE | Đúng dấu |
|---|---|---:|---:|---:|---:|
| validation | Zero | 258 | 2.014185 | 1.437107 | Không áp dụng |
| validation | Mean | 258 | 2.011941 | 1.435058 | 58.14% |
| validation | AR | 258 | 2.009686 | 1.436502 | 48.45% |
| validation | SETAR | 258 | 1.983621 | 1.433983 | 51.55% |
| test | Zero | 172 | 4.636073 | 3.059508 | Không áp dụng |
| test | Mean | 172 | 4.636515 | 3.055924 | 55.23% |
| test | AR | 172 | 4.662725 | 3.078517 | 46.51% |
| test | SETAR | 172 | 4.786064 | 3.140517 | 44.19% |

Zero không đưa ra dự báo hướng tăng/giảm, nên Direction_accuracy để trống trong CSV và không áp dụng trong bảng trên.

## Nhận xét

- RMSE test của SETAR cao hơn AR 2.645%.
- RMSE test của SETAR cao hơn Mean 3.225%.
- RMSE test của SETAR cao hơn Zero 3.235%.

Các chênh lệch trên là kết quả mô tả của một giai đoạn test, chưa phải chứng minh lợi thế ổn định hay khác biệt có ý nghĩa thống kê. Mô hình ngưỡng linh hoạt hơn không bảo đảm dự báo tốt hơn.

Hiệu ứng ARCH trong dữ liệu ban đầu chỉ liên quan tới phương sai có điều kiện; không phải kiểm định chứng minh phương trình trung bình có ngưỡng. Phần này chưa thực hiện kiểm định tuyến tính đối lập SETAR với phân phối null/ bootstrap phù hợp cho ngưỡng chưa xác định dưới H0.

## Chẩn đoán và giới hạn

`residual_diagnostics.csv` chứa Ljung–Box cho phần dư, bình phương phần dư và ARCH-LM trên train+validation. Ljung–Box dùng `model_df=0`; đây là chẩn đoán thăm dò sau chọn mô hình, không phải kiểm định chính thức đã điều chỉnh bậc tự do và quá trình tìm ngưỡng. ARCH-LM dùng số hệ số hồi quy để điều chỉnh ddof.

`train_bic` trong lưới chỉ là tiêu chí tham khảo Gaussian với phương sai chung, đếm cả hệ số, ngưỡng và phương sai. Việc lựa chọn cuối cùng dựa vào validation RMSE. Nếu phần dư còn ARCH, SETAR ở đây chưa mô hình hóa phương sai động.

Dự báo giá trong `forecasts.csv` là phép biến đổi `Close[t-1]*exp(predicted_return/100)`, không phải kỳ vọng có điều kiện của giá vì chưa hiệu chỉnh Jensen. Chỉ dùng return để xếp hạng mô hình.

Lưới ngưỡng hữu hạn có thể bỏ qua cực tiểu giữa các điểm; phạm vi p,d được đặt trước. Chưa xét bậc khác nhau giữa hai chế độ, ba chế độ hoặc dự báo nhiều bước. Dữ liệu cực đoan được giữ nguyên. Chẩn đoán trên từng chế độ không tự chứng minh tính dừng toàn cục của SETAR.

## Chạy lại

Từ thư mục gốc repo:

```powershell
python -m pip install -r requirements.txt
python code/run_setar_tar.py
```

Hoặc mở `code/05_mo_hinh_setar_tar.ipynb` và Run All. Các notebook 01/02 cũ vẫn có đường dẫn cứng D:/Chuyên đề 2; notebook 05 đọc trực tiếp dữ liệu đã chia trong repo.

## Tài liệu

1. Cryer, J. D., & Chan, K.-S. (2008). *Time Series Analysis: With Applications in R*, 2nd ed., Chapter 15: Threshold Models, pp. 383–422. Springer. https://doi.org/10.1007/978-0-387-75959-3.
2. tsDyn documentation: Self Threshold Autoregressive model. https://search.r-project.org/CRAN/refmans/tsDyn/html/setar.html.
3. tsDyn documentation: Automatic selection of SETAR hyper-parameters. https://search.r-project.org/CRAN/refmans/tsDyn/html/selectSETAR.html.

Triển khai Python dùng mô hình hai chế độ và CLS theo tài liệu; không tuyên bố tái hiện chính xác phần mềm tsDyn/TSA hay thí nghiệm gốc trong sách. Lưới, tiêu chí validation và đối chứng là thiết kế của dự án này.

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
- Trên return: báo cáo RMSE, MAE, MASE, đúng dấu và Relative RMSE so với Zero. Không dùng MAPE cho return vì return có thể bằng/gần 0.
- Trên giá: đổi dự báo return thành điểm giá bằng `Close[t-1]*exp(predicted_return/100)`, rồi báo cáo MAE, RMSE, MAPE, sMAPE, MASE, đúng chiều và Relative RMSE so với cùng một Naive (`Close[t-1]`).
- Khoảng dự báo 95% của SETAR dùng phân vị 2,5% và 97,5% của phần dư trong từng chế độ trên tập ước lượng; không dùng test để đặt độ rộng.

## Cấu hình chọn được

- SETAR chọn trên validation: p=4, d=4, ngưỡng train = 1.072667%.
- Sau refit: ngưỡng = 1.050277%; số quan sát fit: 4,118; Low=3,212, High=906.
- AR đối chứng: p=3.

## Kết quả dự báo log-return

| Tập | Mô hình | n | RMSE | MAE | MASE | Relative RMSE | Đúng dấu |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | Zero | 258 | 2.014185 | 1.437107 | 0.776975 | 1.000000 | Không áp dụng |
| validation | Mean | 258 | 2.011941 | 1.435058 | 0.775867 | 0.998886 | 58.14% |
| validation | AR | 258 | 2.009686 | 1.436502 | 0.776648 | 0.997766 | 48.45% |
| validation | SETAR | 258 | 1.983621 | 1.433983 | 0.775286 | 0.984826 | 51.55% |
| test | Zero | 172 | 4.636073 | 3.059508 | 1.639315 | 1.000000 | Không áp dụng |
| test | Mean | 172 | 4.636515 | 3.055924 | 1.637394 | 1.000095 | 55.23% |
| test | AR | 172 | 4.662725 | 3.078517 | 1.649500 | 1.005749 | 46.51% |
| test | SETAR | 172 | 4.786064 | 3.140517 | 1.682720 | 1.032353 | 44.19% |

## Kết quả trên thang giá dùng để so sánh các thuật toán

| Tập | Mô hình | n | MAE | RMSE | MAPE | sMAPE | MASE | Relative RMSE | Đúng chiều |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| validation | Zero | 258 | 0.639919 | 1.060690 | 1.433572% | 1.436958% | 2.214024 | 1.000000 | Không áp dụng |
| validation | Mean | 258 | 0.639036 | 1.059746 | 1.431710% | 1.434909% | 2.210970 | 0.999110 | 58.14% |
| validation | AR | 258 | 0.639187 | 1.059354 | 1.433297% | 1.436354% | 2.211494 | 0.998740 | 48.45% |
| validation | SETAR | 258 | 0.635695 | 1.035188 | 1.430900% | 1.433844% | 2.199411 | 0.975957 | 51.55% |
| test | Zero | 172 | 2.374392 | 3.936722 | 3.105538% | 3.056695% | 7.635549 | 1.000000 | Không áp dụng |
| test | Mean | 172 | 2.372262 | 3.937658 | 3.103017% | 3.053104% | 7.628699 | 1.000238 | 55.23% |
| test | AR | 172 | 2.388479 | 3.956405 | 3.125448% | 3.075664% | 7.680849 | 1.005000 | 46.51% |
| test | SETAR | 172 | 2.438514 | 4.064494 | 3.189635% | 3.137457% | 7.841750 | 1.032456 | 44.19% |

## Khoảng dự báo SETAR

| Tập | Thang đo | Mức danh nghĩa | Bao phủ thực tế | Độ rộng trung bình |
|---|---|---:|---:|---:|
| validation | return_percent | 95.00% | 93.41% | 7.250875 |
| validation | price_USD_per_ounce | 95.00% | 93.41% | 2.908114 |
| test | return_percent | 95.00% | 70.93% | 7.328310 |
| test | price_USD_per_ounce | 95.00% | 70.93% | 5.446824 |

Zero không đưa ra dự báo hướng tăng/giảm, nên Direction_accuracy để trống trong CSV và không áp dụng trong bảng trên.

## Nhận xét

- RMSE test của SETAR cao hơn AR 2.645%.
- RMSE test của SETAR cao hơn Mean 3.225%.
- RMSE test của SETAR cao hơn Zero 3.235%.

Các chênh lệch trên là kết quả mô tả của một giai đoạn test, chưa phải chứng minh lợi thế ổn định hay khác biệt có ý nghĩa thống kê. Mô hình ngưỡng linh hoạt hơn không bảo đảm dự báo tốt hơn.

Hiệu ứng ARCH trong dữ liệu ban đầu chỉ liên quan tới phương sai có điều kiện; không phải kiểm định chứng minh phương trình trung bình có ngưỡng. Phần này chưa thực hiện kiểm định tuyến tính đối lập SETAR với phân phối null/ bootstrap phù hợp cho ngưỡng chưa xác định dưới H0.

## Chẩn đoán và giới hạn

`residual_diagnostics.csv` chứa Ljung–Box cho phần dư, bình phương phần dư, ARCH-LM và Jarque–Bera trên train+validation. Ljung–Box dùng `model_df=0`; đây là chẩn đoán thăm dò sau chọn mô hình, không phải kiểm định chính thức đã điều chỉnh bậc tự do và quá trình tìm ngưỡng. ARCH-LM dùng số hệ số hồi quy để điều chỉnh ddof.

`train_bic` trong lưới chỉ là tiêu chí tham khảo Gaussian với phương sai chung, đếm cả hệ số, ngưỡng và phương sai. Việc lựa chọn cuối cùng dựa vào validation RMSE. Nếu phần dư còn ARCH, SETAR ở đây chưa mô hình hóa phương sai động.

Dự báo giá trong `forecasts.csv` là phép biến đổi `Close[t-1]*exp(predicted_return/100)`, không phải kỳ vọng có điều kiện của giá vì chưa hiệu chỉnh Jensen. Cấu hình SETAR vẫn được chọn bằng RMSE return trên validation; bảng giá dùng để so sánh sau khi đã khóa mô hình.

Khoảng dự báo thực nghiệm chỉ phản ánh phân phối phần dư lịch sử theo chế độ, chưa cộng bất định tham số và chưa mô hình hóa ARCH. Do đó coverage test là kết quả cần đánh giá, không phải đảm bảo sẽ bằng 95%.

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

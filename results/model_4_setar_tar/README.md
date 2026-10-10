# Mô hình 4 — SETAR/TAR cho tỷ giá USD/VND

SETAR là TAR tự kích hoạt: ngưỡng được xác định bởi return quá khứ của chính chuỗi, không phải biến ngoại sinh.

## Mô hình và giao thức

r_t = 100 log(P_t / P_(t-1)). Nếu r_(t-d) <= gamma, dùng AR(p) của chế độ Low; ngược lại dùng AR(p) của chế độ High. Mỗi chế độ có intercept và hệ số riêng.

- Train 2016–2024: fit bằng bình phương tối thiểu có điều kiện; tìm gamma bằng SSR nhỏ nhất.
- Thử p=1..10, d=1..5; gamma tại tối đa 71 phân vị trong khoảng 15%–85%; mỗi chế độ cần ít nhất 15% số quan sát và 5(p+1) quan sát.
- Chọn p,d bằng RMSE return trên Validation 2025; hòa điểm thì ưu tiên p,d nhỏ hơn. AIC/BIC không dùng để chọn trên Test.
- Refit hệ số và gamma trên Train+Validation, giữ p,d đã chọn. Tất cả cấu hình cố định trước khi đánh giá Test.
- Test 01/01–31/08/2026: 175 dự báo rolling one-step. Chỉ cập nhật lag bằng quan sát đã xảy ra, không fit lại tham số.
- Low/High nói về biến ngưỡng quá khứ, không phải khẳng định phiên kế tiếp giảm/tăng.

Cấu hình chọn bằng Validation, ngưỡng fit trên Train: p=6, d=4, gamma=-0.03939218 điểm phần trăm.
Mô hình cuối: p=6, d=4, gamma=-0.03880847; Low=624, High=1974 quan sát fit.

## Đánh giá return

| Split      | Model   |   N |      MAE |     RMSE |     MASE |   R2_out_of_sample |   DA_all_percent |   DA_nonzero_actual_percent |   Relative_RMSE_vs_Naive |
|:-----------|:--------|----:|---------:|---------:|---------:|-------------------:|-----------------:|----------------------------:|-------------------------:|
| validation | Zero    | 262 | 0.097492 | 0.168590 | 0.915013 |          -0.005169 |         8.396947 |                    0.000000 |                 1.000000 |
| validation | Mean    | 262 | 0.097876 | 0.168287 | 0.918620 |          -0.001561 |        47.328244 |                   51.666667 |                 0.998204 |
| validation | AR      | 262 | 0.100433 | 0.167174 | 0.942614 |           0.011655 |        42.366412 |                   46.250000 |                 0.991596 |
| validation | SETAR   | 262 | 0.100953 | 0.165605 | 0.947493 |           0.030115 |        43.893130 |                   47.916667 |                 0.982292 |
| test       | Zero    | 175 | 0.070908 | 0.115090 | 0.642222 |          -0.001864 |        10.285714 |                    0.000000 |                 1.000000 |
| test       | Mean    | 175 | 0.072214 | 0.115516 | 0.654058 |          -0.009282 |        40.000000 |                   44.585987 |                 1.003695 |
| test       | AR      | 175 | 0.072180 | 0.116444 | 0.653748 |          -0.025580 |        49.714286 |                   55.414013 |                 1.011767 |
| test       | SETAR   | 175 | 0.074411 | 0.118224 | 0.673957 |          -0.057158 |        48.571429 |                   54.140127 |                 1.027225 |

Return MAE/RMSE có đơn vị điểm phần trăm. Relative RMSE so với Zero return. MASE return dùng sai số persistence return trong mẫu huấn luyện, nên MASE<1 không đồng nghĩa thắng Zero trên Test.
MAPE/sMAPE return để trống vì target có 0, gần 0 và giá trị âm; không thay 0 bằng epsilon để tạo chỉ số đẹp.

## Đánh giá tỷ giá

Dự báo giá chính = P_(t-1) exp(r_hat/100) × mean(exp(residual/100)) của chế độ tương ứng. Hệ số smearing chỉ dùng residual của mẫu fit, nhằm hiệu chỉnh phép đổi thang phi tuyến. Forecast plug-in chưa hiệu chỉnh vẫn được lưu riêng trong forecasts.csv.

| Split      | Model   |       MAE |      RMSE |   MAPE_percent |   sMAPE_percent |     MASE |   DA_all_percent |   Relative_RMSE_vs_Naive |
|:-----------|:--------|----------:|----------:|---------------:|----------------:|---------:|-----------------:|-------------------------:|
| validation | Zero    | 25.187023 | 43.252829 |       0.097463 |        0.097492 | 1.439890 |         8.396947 |                 1.000000 |
| validation | Mean    | 25.289077 | 43.174200 |       0.097861 |        0.097884 | 1.445724 |        47.328244 |                 0.998182 |
| validation | AR      | 25.941585 | 42.895409 |       0.100420 |        0.100433 | 1.483027 |        42.748092 |                 0.991736 |
| validation | SETAR   | 26.071270 | 42.473124 |       0.100942 |        0.100956 | 1.490440 |        43.893130 |                 0.981973 |
| test       | Zero    | 18.562857 | 30.053405 |       0.070899 |        0.070908 | 1.016289 |        10.285714 |                 1.000000 |
| test       | Mean    | 18.911548 | 30.168630 |       0.072234 |        0.072238 | 1.035379 |        40.000000 |                 1.003834 |
| test       | AR      | 18.898896 | 30.406055 |       0.072188 |        0.072196 | 1.034686 |        49.714286 |                 1.011734 |
| test       | SETAR   | 19.483811 | 30.868815 |       0.074419 |        0.074430 | 1.066710 |        48.571429 |                 1.027132 |

RMSE SETAR cải thiện so với Naive trên Test: -2.7132% (âm nghĩa là kém hơn). Đây là kết quả ngoài mẫu của cấu hình đã chọn, không tiếp tục chỉnh theo Test.
MASE giá dùng mean(|P_t−P_(t-1)|) trên toàn bộ mẫu giá fit; Relative RMSE dùng Naive trên đúng Test. Hai chuẩn so sánh khác nhau.
Đối chứng gồm Zero/Naive, Mean và AR tuyến tính, đều thuộc giao thức đánh giá của thí nghiệm SETAR/TAR.

## Khoảng dự báo 95%

| Split      | Model   | Scale             |   Nominal_coverage_percent |   PI_coverage_percent |   Average_PI_width |   Mean_interval_score |   Below_lower_N |   Above_upper_N |
|:-----------|:--------|:------------------|---------------------------:|----------------------:|-------------------:|----------------------:|----------------:|----------------:|
| validation | SETAR   | return_percent    |                  95.000000 |             93.129771 |           0.545206 |              1.133663 |               7 |              11 |
| validation | SETAR   | price_VND_per_USD |                  95.000000 |             93.129771 |         141.758029 |            291.973111 |               7 |              11 |
| test       | SETAR   | return_percent    |                  95.000000 |             96.000000 |           0.566168 |              0.761781 |               3 |               4 |
| test       | SETAR   | price_VND_per_USD |                  95.000000 |             96.000000 |         148.611350 |            199.629748 |               3 |               4 |

Khoảng lấy phân vị 2,5% và 97,5% của residual mẫu fit theo từng chế độ. Đánh giá coverage, độ rộng và interval score. Khoảng xấp xỉ, chưa bao gồm bất định tham số và chưa mô hình hóa ARCH; không hiệu chỉnh bằng Test.

## Chẩn đoán và độ chắc chắn

- Có Ljung–Box ở lag 5/10/20 cho residual và residual bình phương, ARCH-LM lag 10, Jarque–Bera trên mẫu fit và forecast error Test.
- P-value chẩn đoán được dùng mô tả; chưa hiệu chỉnh kiểm định nhiều lần. Không dùng F-test thông thường để kết luận có hiệu ứng ngưỡng.
- paired_loss_comparisons.csv: chênh lệch squared/absolute loss với Zero, Mean, AR; CI95% paired circular moving-block bootstrap (2.000 mẫu, block=5, seed=20261010). Dương nghĩa SETAR kém hơn; CI chứa 0 nghĩa chưa có bằng chứng rõ về chênh lệch theo cách đánh giá này.
- Có thêm thống kê HAC với bandwidth=5 và p-value chuẩn xấp xỉ, chỉ tham khảo; mô hình AR/SETAR có thể lồng nhau, kiểm định và p-value này không thay thế kiểm định phi tuyến chuyên biệt.
- Directional Accuracy gồm down/flat/up; lưu thêm accuracy chỉ trên phiên thực tế đổi giá và balanced class recall. Zero dự báo flat nên không được diễn giải là tín hiệu dự báo hướng.
- Giữ nguyên các dòng OHLC bị gắn cờ; mô hình sử dụng Close. data_audit.csv và SHA256 đầu vào nằm trong selected_models.json.

## Chạy lại

```bash
pip install -r requirements.txt
python code/run_setar_tar_usd_vnd.py
python -m unittest discover -s code -p 'test_setar_tar_usd_vnd.py'
```

Hoặc Run All trong code/08_mo_hinh_4_setar_tar_usd_vnd.ipynb. Chạy từ thư mục gốc repo hoặc thư mục code. Kết quả nằm ở results/model_4_setar_tar và figures/model_4_setar_tar.

## Tài liệu

- Cryer & Chan, Time Series Analysis: With Applications in R, 2nd ed., Springer, 2008 (mô hình ngưỡng/phi tuyến).
- Hansen (1997), Inference in TAR Models, Studies in Nonlinear Dynamics and Econometrics, 2(1). https://users.ssc.wisc.edu/~behansen/papers/snde_97.html (phân phối suy luận ngưỡng không chuẩn).

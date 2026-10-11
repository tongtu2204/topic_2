"""Run model 4 on USD/VND from any working directory."""
from pathlib import Path
import json
import platform
import importlib.metadata

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import probplot

from setar_tar_usd_vnd import load_splits, select_models, evaluate, save_tables
from setar_evaluation_usd_vnd import audit_inputs, enrich_evaluation, save_extra
from setar_figures_usd_vnd import make_figures


def extra_figures(base, evaluation):
    out = base/"figures"/"model_4_setar_tar"
    scores = evaluation["price_metrics"].query("Split == 'test'")
    labels = scores.Model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("DA_all_percent", "DA_nonzero_actual_percent"), ("Down / flat / up", "Actual nonzero moves only")):
        bars = ax.bar(labels, scores[metric], color="#2563eb")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set(title=title, ylabel="Directional accuracy (%)", ylim=(0, 100))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Direction metrics: a flat forecast is not a directional signal")
    fig.tight_layout(); fig.savefig(out/"11_directional_accuracy.png", dpi=180); plt.close(fig)
    test = evaluation["predictions"].query("Split == 'test'")
    residual = test.actual_return-test.SETAR
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(test.Date, residual, color="#b45309"); axes[0].axhline(0, color="black", lw=.7)
    axes[0].set(title="Test one-step SETAR return errors", ylabel="Percentage points")
    probplot(residual, dist="norm", plot=axes[1]); axes[1].set_title("Q-Q plot: test forecast errors")
    fig.tight_layout(); fig.savefig(out/"12_test_errors_qq.png", dpi=180); plt.close(fig)


def write_note(base, selected, evaluation, config):
    out = base/"results"/"model_4_setar_tar"/"README.md"
    model = evaluation["SETAR"]
    train = selected["SETAR"]
    scores = evaluation["price_metrics"].query("Split == 'test'").set_index("Model")
    delta = 100*(1-scores.loc["SETAR", "RMSE"]/scores.loc["Zero", "RMSE"])
    lines = ["# Mô hình 4 — SETAR/TAR cho tỷ giá USD/VND", "",
             "SETAR là TAR tự kích hoạt: ngưỡng được xác định bởi return quá khứ của chính chuỗi, không phải biến ngoại sinh.", "",
             "## Mô hình và giao thức", "",
             "r_t = 100 log(P_t / P_(t-1)). Nếu r_(t-d) <= gamma, dùng AR(p) của chế độ Low; ngược lại dùng AR(p) của chế độ High. Mỗi chế độ có intercept và hệ số riêng.", "",
             "- Train 2016–2024: fit bằng bình phương tối thiểu có điều kiện; tìm gamma bằng SSR nhỏ nhất.",
             "- Thử p=1..10, d=1..5; gamma tại tối đa 71 phân vị trong khoảng 15%–85%; mỗi chế độ cần ít nhất 15% số quan sát và 5(p+1) quan sát.",
             "- Chọn p,d bằng RMSE return trên Validation 2025; hòa điểm thì ưu tiên p,d nhỏ hơn. AIC/BIC không dùng để chọn trên Test.",
             "- Refit hệ số và gamma trên Train+Validation, giữ p,d đã chọn. Tất cả cấu hình cố định trước khi đánh giá Test.",
             "- Test 01/01–31/08/2026: 175 dự báo rolling one-step. Chỉ cập nhật lag bằng quan sát đã xảy ra, không fit lại tham số.",
             "- Low/High nói về biến ngưỡng quá khứ, không phải khẳng định phiên kế tiếp giảm/tăng.", "",
             f"Cấu hình chọn bằng Validation, ngưỡng fit trên Train: p={train.p}, d={train.d}, gamma={train.threshold:.8f} điểm phần trăm.",
             f"Mô hình cuối: p={model.p}, d={model.d}, gamma={model.threshold:.8f}; Low={model.n_low}, High={model.n_high} quan sát fit.", "",
             "## Đánh giá return", "",
             evaluation["metrics"][["Split", "Model", "N", "MAE", "RMSE", "MASE", "R2_out_of_sample", "DA_all_percent", "DA_nonzero_actual_percent", "Relative_RMSE_vs_Naive"]].to_markdown(index=False, floatfmt=".6f"), "",
             "Return MAE/RMSE có đơn vị điểm phần trăm. Relative RMSE so với Zero return. MASE return dùng sai số persistence return trong mẫu huấn luyện, nên MASE<1 không đồng nghĩa thắng Zero trên Test.",
             "MAPE/sMAPE return để trống vì target có 0, gần 0 và giá trị âm; không thay 0 bằng epsilon để tạo chỉ số đẹp.", "",
             "## Đánh giá tỷ giá", "",
             "Dự báo giá chính = P_(t-1) exp(r_hat/100) × mean(exp(residual/100)) của chế độ tương ứng. Hệ số smearing chỉ dùng residual của mẫu fit, nhằm hiệu chỉnh phép đổi thang phi tuyến. Forecast plug-in chưa hiệu chỉnh vẫn được lưu riêng trong forecasts.csv.", "",
             evaluation["price_metrics"][["Split", "Model", "MAE", "RMSE", "MAPE_percent", "sMAPE_percent", "MASE", "DA_all_percent", "Relative_RMSE_vs_Naive"]].to_markdown(index=False, floatfmt=".6f"), "",
             f"RMSE SETAR cải thiện so với Naive trên Test: {delta:.4f}% (âm nghĩa là kém hơn). Đây là kết quả ngoài mẫu của cấu hình đã chọn, không tiếp tục chỉnh theo Test.",
             "MASE giá dùng mean(|P_t−P_(t-1)|) trên toàn bộ mẫu giá fit; Relative RMSE dùng Naive trên đúng Test. Hai chuẩn so sánh khác nhau.",
             "Đối chứng gồm Zero/Naive, Mean và AR tuyến tính, đều thuộc giao thức đánh giá của thí nghiệm SETAR/TAR.", "",
             "## Khoảng dự báo 95%", "",
             evaluation["interval_metrics"].to_markdown(index=False, floatfmt=".6f"), "",
             "Khoảng lấy phân vị 2,5% và 97,5% của residual mẫu fit theo từng chế độ. Đánh giá coverage, độ rộng và interval score. Khoảng xấp xỉ, chưa bao gồm bất định tham số và chưa mô hình hóa ARCH; không hiệu chỉnh bằng Test.", "",
             "## Chẩn đoán và độ chắc chắn", "",
             "- Có Ljung–Box ở lag 5/10/20 cho residual và residual bình phương, ARCH-LM lag 10, Jarque–Bera trên mẫu fit và forecast error Test.",
             "- P-value chẩn đoán được dùng mô tả; chưa hiệu chỉnh kiểm định nhiều lần. Không dùng F-test thông thường để kết luận có hiệu ứng ngưỡng.",
             "- paired_loss_comparisons.csv: chênh lệch squared/absolute loss với Zero, Mean, AR; CI95% paired circular moving-block bootstrap (2.000 mẫu, block=5, seed=20261010). Dương nghĩa SETAR kém hơn; CI chứa 0 nghĩa chưa có bằng chứng rõ về chênh lệch theo cách đánh giá này.",
             "- Có thêm thống kê HAC với bandwidth=5 và p-value chuẩn xấp xỉ, chỉ tham khảo; mô hình AR/SETAR có thể lồng nhau, kiểm định và p-value này không thay thế kiểm định phi tuyến chuyên biệt.",
             "- Directional Accuracy gồm down/flat/up; lưu thêm accuracy chỉ trên phiên thực tế đổi giá và balanced class recall. Zero dự báo flat nên không được diễn giải là tín hiệu dự báo hướng.",
             "- Giữ nguyên các dòng OHLC bị gắn cờ; mô hình sử dụng Close. data_audit.csv và SHA256 đầu vào nằm trong selected_models.json.", "",
             "## Chạy lại", "", "```bash", "pip install -r requirements.txt", "python code/run_setar_tar_usd_vnd.py", "python -m unittest discover -s code -p 'test_setar_tar_usd_vnd.py'", "```", "",
             "Hoặc Run All trong code/08_mo_hinh_4_setar_tar_usd_vnd.ipynb. Chạy từ thư mục gốc repo hoặc thư mục code. Kết quả nằm ở results/model_4_setar_tar và figures/model_4_setar_tar.", "",
             "## Tài liệu", "",
             "- Cryer & Chan, Time Series Analysis: With Applications in R, 2nd ed., Springer, 2008 (mô hình ngưỡng/phi tuyến).", 
             "- Hansen (1997), Inference in TAR Models, Studies in Nonlinear Dynamics and Econometrics, 2(1). https://users.ssc.wisc.edu/~behansen/papers/snde_97.html (phân phối suy luận ngưỡng không chuẩn).", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def run_experiment(base=None, verbose=True):
    base = Path(base) if base is not None else Path(__file__).resolve().parent.parent
    splits = load_splits(base)
    audit = audit_inputs(base, splits)
    selected, grid, profiles = select_models(splits)
    evaluation = enrich_evaluation(base, splits, selected, evaluate(splits, selected))
    tables, config = save_tables(base, splits, selected, grid, profiles, evaluation)
    extras = save_extra(base, splits, audit, evaluation, config)
    for fig in make_figures(base, splits, selected, grid, profiles, evaluation, tables):
        plt.close(fig)
    extra_figures(base, evaluation)
    report = write_note(base, selected, evaluation, config)
    environment = {"python": platform.python_version(), "packages": {name: importlib.metadata.version(name)
                    for name in ("numpy", "pandas", "scipy", "statsmodels", "matplotlib", "nbformat", "nbclient")}}
    (base/"results"/"model_4_setar_tar"/"run_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    if verbose:
        print(audit.to_string(index=False))
        print("\nTEST RETURN"); print(evaluation["metrics"].query("Split == 'test'").to_string(index=False))
        print("\nTEST PRICE"); print(evaluation["price_metrics"].query("Split == 'test'").to_string(index=False))
        print("\nINTERVALS"); print(evaluation["interval_metrics"].to_string(index=False))
        print(f"\nResults: {report}")
    return splits, selected, grid, profiles, evaluation, tables, config, extras


if __name__ == "__main__":
    run_experiment()

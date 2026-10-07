"""Figures and a Vietnamese results note for the SETAR experiment."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf


COLORS = {"SETAR": "#b45309", "AR": "#2563eb", "Mean": "#0f766e", "Zero": "#64748b"}


def make_figures(base, splits, selected, grid, profiles, evaluation, tables):
    output = Path(base)/"figures"/"setar_tar"
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.titleweight": "bold"})
    figures = []

    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(output/filename, dpi=180, bbox_inches="tight")
        figures.append(fig)

    pivot = grid[grid.Model=="SETAR"].pivot(index="p", columns="d", values="RMSE")
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns)
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i,j]:.3f}", ha="center", va="center", color="white", fontsize=9,
                    bbox={"facecolor": "black", "alpha": .22, "edgecolor": "none", "pad": 1})
    winner = selected["SETAR"]
    ax.scatter(list(pivot.columns).index(winner.d), list(pivot.index).index(winner.p),
               s=440, marker="s", facecolors="none", edgecolors="#f59e0b", linewidths=2.5)
    ax.set(xlabel="Threshold delay d (trading observations)", ylabel="AR order p in each regime",
           title="SETAR selection: validation RMSE (2025)")
    fig.colorbar(image, ax=ax, label="RMSE (percentage points)")
    save(fig, "01_validation_grid.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    train_profile = profiles[(profiles.p==winner.p)&(profiles.d==winner.d)]
    for ax, profile, model, label in zip(axes, [train_profile, evaluation["refit_profile"]],
                                        [winner, evaluation["SETAR"]], ["Train (2010-2024)", "Train + validation (2010-2025)"]):
        ax.plot(profile.threshold, profile.train_rss, color=COLORS["SETAR"])
        ax.axvline(model.threshold, color="#334155", ls="--", label=f"Threshold = {model.threshold:.3f}%")
        ax.set(title=label, xlabel="Threshold gamma (%)", ylabel="Conditional residual sum of squares")
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(f"Threshold estimation at selected p={winner.p}, d={winner.d}", fontweight="bold")
    save(fig, "02_threshold_profile.png")

    forecast = evaluation["predictions"]
    test = forecast[forecast.Split=="test"].copy()
    model = evaluation["SETAR"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for regime, color in (("Low", "#2563eb"), ("High", "#b45309")):
        frame = test[test.SETAR_regime==regime]
        axes[0].scatter(frame.Date, frame.threshold_variable, s=16, color=color, label=regime)
        axes[1].scatter(frame.Date, frame.actual_return, s=16, color=color, label=regime)
    axes[0].axhline(model.threshold, color="#334155", ls="--", label=f"Threshold = {model.threshold:.3f}%")
    axes[0].set(ylabel="Lagged return r[t-d] (%)", title="Test regimes chosen using past return")
    axes[1].set(ylabel="Realized return r[t] (%)", xlabel="Date")
    axes[1].axhline(0, color="#64748b", lw=.7)
    for ax in axes:
        ax.legend(loc="upper left", ncol=3)
        ax.grid(alpha=.2)
    save(fig, "03_test_regimes.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(test.Date, test.actual_return, color="#334155", lw=1, label="Actual return")
    for kind in ("SETAR", "AR", "Mean", "Zero"):
        for ax in axes:
            ax.plot(test.Date, test[kind], color=COLORS[kind], lw=1.3 if kind=="SETAR" else .9,
                    alpha=.9, label=kind)
    axes[0].set(title="Sequential one-step forecasts: test (2026)", ylabel="Return (%)")
    axes[1].set(title="Forecasts enlarged to show differences", ylabel="Forecast return (%)", xlabel="Date")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncol=5)
    save(fig, "04_test_forecasts.png")

    scores = evaluation["metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    kinds = ["Zero", "Mean", "AR", "SETAR"]
    for row, split in enumerate(("validation", "test")):
        subset = scores[scores.Split==split].set_index("Model").loc[kinds]
        for col, metric in enumerate(("RMSE", "MAE")):
            ax = axes[row, col]
            bars = ax.bar(kinds, subset[metric], color=[COLORS[k] for k in kinds], width=.65)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
            ax.set(title=f"{split.capitalize()} - {metric}", ylabel="Percentage points")
            ax.set_ylim(0, subset[metric].max()*1.18)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle("Forecast accuracy (lower is better)", fontweight="bold")
    save(fig, "05_forecast_metrics.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, loss, label in ((axes[0], lambda e: e**2, "Squared error"),
                            (axes[1], np.abs, "Absolute error")):
        error_setar = test.actual_return.to_numpy()-test.SETAR.to_numpy()
        for kind in ("AR", "Mean", "Zero"):
            error_base = test.actual_return.to_numpy()-test[kind].to_numpy()
            difference = np.cumsum(loss(error_setar)-loss(error_base))
            ax.plot(test.Date, difference, color=COLORS[kind], label=f"SETAR minus {kind}")
        ax.axhline(0, color="#334155", lw=.8)
        ax.set(title=label, ylabel="Cumulative loss difference", xlabel="Date")
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Test: positive difference means SETAR has higher error", fontweight="bold")
    save(fig, "06_cumulative_loss.png")

    residuals = tables["residuals.csv"]
    residual = residuals.loc[residuals.Model=="SETAR", "residual"].to_numpy()
    dates = pd.concat([splits["train"], splits["validation"]]).Date.to_numpy()[model.start:]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].plot(dates, residual, color="#334155", lw=.65)
    axes[0,0].set(title="SETAR residuals: estimation sample", ylabel="Residual (%)", xlabel="Date")
    axes[0,1].hist(residual, bins=60, density=True, color=COLORS["SETAR"], alpha=.85)
    axes[0,1].set(title="Residual distribution", xlabel="Residual (%)", ylabel="Density")
    plot_acf(residual, lags=40, ax=axes[1,0], zero=False)
    axes[1,0].set_title("ACF of residuals (descriptive)")
    plot_acf(residual**2, lags=40, ax=axes[1,1], zero=False)
    axes[1,1].set_title("ACF of squared residuals (descriptive)")
    save(fig, "07_residual_diagnostics.png")
    return figures


def write_results_note(base, config, evaluation, tables):
    output = Path(base)/"results"/"setar_tar"/"README.md"
    scores = evaluation["metrics"]
    selected = config["models"]["train"]["SETAR"]
    final = config["models"]["train_validation"]["SETAR"]
    ar = config["models"]["train_validation"]["AR"]
    lines = ["# Mục 4 — SETAR / TAR cho log-return của bạc XAG/USD", "",
             "## Mô hình", "",
             "TAR là họ mô hình tự hồi quy theo ngưỡng. SETAR là trường hợp biến ngưỡng lấy từ chính chuỗi trễ. "
             "Ở đây sử dụng SETAR hai chế độ, cùng bậc p và có hệ số chặn ở mỗi chế độ:", "",
             r"$$r_t=\begin{cases}c_L+\sum_{j=1}^{p}\phi_{L,j}r_{t-j}+\varepsilon_t,&r_{t-d}\le\gamma,\\c_H+\sum_{j=1}^{p}\phi_{H,j}r_{t-j}+\varepsilon_t,&r_{t-d}>\gamma.\end{cases}$$", "",
             "Return tính theo phần trăm: `100 * log(Close[t] / Close[t-1])`. Low/High chỉ thể hiện "
             "giá trị return trễ ở dưới/trên ngưỡng; không phải nhãn thị trường tăng/giảm hay biến động thấp/cao.", "",
             "## Giao thức thực nghiệm", "",
             "- Train: 2010–2024; validation: 2025; test: 02/01/2026–31/08/2026.",
             "- Dò `p=1..10`, `d=1..5`; lưới ngưỡng gồm 71 phân vị từ 15% đến 85% của biến ngưỡng trong tập ước lượng.",
             "- Với từng `(p,d)`, chọn ngưỡng tối thiểu hóa SSR trên train; ước lượng hai phương trình bằng bình phương tối thiểu có điều kiện.",
             "- Mỗi chế độ có ít nhất 15% quan sát và ít nhất `5*(p+1)` quan sát. Tất cả cấu hình dùng chung mốc khởi đầu lag 10.",
             "- Chọn `(p,d)` bằng RMSE validation. AR tuyến tính cũng chọn p bằng cùng RMSE validation.",
             "- Sau khi khóa p,d, ước lượng lại ngưỡng và hệ số trên train+validation; giữ cố định tham số trong test.",
             "- Dự báo từng phiên một bước trước. Tại phiên t chỉ dùng dữ liệu đã biết đến t−1; return thực tế của phiên trước được cập nhật vào lag.",
             "- Mean dự báo bằng trung bình tập ước lượng; Zero dự báo return bằng 0. Đây là các đối chứng riêng, không tham gia chọn cấu hình SETAR.",
             "- RMSE/MAE tính bằng điểm phần trăm. Không dùng MAPE cho return vì return có thể bằng/gần 0.", "",
             "## Cấu hình chọn được", "",
             f"- SETAR chọn trên validation: p={selected['p']}, d={selected['d']}, ngưỡng train = {selected['threshold']:.6f}%.",
             f"- Sau refit: ngưỡng = {final['threshold']:.6f}%; số quan sát fit: {final['nobs']:,}; "
             f"Low={final['n_low']:,}, High={final['n_high']:,}.",
             f"- AR đối chứng: p={ar['p']}.", "",
             "## Kết quả dự báo", "", "| Tập | Mô hình | n | RMSE | MAE | Đúng dấu |", "|---|---|---:|---:|---:|---:|"]
    for row in scores.itertuples():
        direction = "Không áp dụng" if row.Model=="Zero" else f"{100*row.Direction_accuracy:.2f}%"
        lines.append(f"| {row.Split} | {row.Model} | {row.n} | {row.RMSE:.6f} | {row.MAE:.6f} | {direction} |")
    lines += ["", "Zero không đưa ra dự báo hướng tăng/giảm, nên Direction_accuracy để trống trong CSV "
              "và không áp dụng trong bảng trên.", "", "## Nhận xét", ""]
    test = scores[scores.Split=="test"].set_index("Model")
    for kind in ("AR", "Mean", "Zero"):
        change = 100*(test.loc["SETAR", "RMSE"]/test.loc[kind, "RMSE"]-1)
        lines.append(f"- RMSE test của SETAR {'cao hơn' if change>=0 else 'thấp hơn'} {kind} {abs(change):.3f}%.")
    lines += ["", "Các chênh lệch trên là kết quả mô tả của một giai đoạn test, chưa phải chứng minh lợi thế "
              "ổn định hay khác biệt có ý nghĩa thống kê. Mô hình ngưỡng linh hoạt hơn không bảo đảm dự báo tốt hơn.", "",
              "Hiệu ứng ARCH trong dữ liệu ban đầu chỉ liên quan tới phương sai có điều kiện; không phải kiểm định "
              "chứng minh phương trình trung bình có ngưỡng. Phần này chưa thực hiện kiểm định tuyến tính "
              "đối lập SETAR với phân phối null/ bootstrap phù hợp cho ngưỡng chưa xác định dưới H0.", "",
              "## Chẩn đoán và giới hạn", "",
              "`residual_diagnostics.csv` chứa Ljung–Box cho phần dư, bình phương phần dư và ARCH-LM trên train+validation. "
              "Ljung–Box dùng `model_df=0`; đây là chẩn đoán thăm dò sau chọn mô hình, không phải kiểm định chính thức "
              "đã điều chỉnh bậc tự do và quá trình tìm ngưỡng. ARCH-LM dùng số hệ số hồi quy để điều chỉnh ddof.", "",
              "`train_bic` trong lưới chỉ là tiêu chí tham khảo Gaussian với phương sai chung, đếm cả hệ số, ngưỡng và phương sai. "
              "Việc lựa chọn cuối cùng dựa vào validation RMSE. Nếu phần dư còn ARCH, SETAR ở đây chưa mô hình hóa phương sai động.", "",
              "Dự báo giá trong `forecasts.csv` là phép biến đổi `Close[t-1]*exp(predicted_return/100)`, "
              "không phải kỳ vọng có điều kiện của giá vì chưa hiệu chỉnh Jensen. Chỉ dùng return để xếp hạng mô hình.", "",
              "Lưới ngưỡng hữu hạn có thể bỏ qua cực tiểu giữa các điểm; phạm vi p,d được đặt trước. "
              "Chưa xét bậc khác nhau giữa hai chế độ, ba chế độ hoặc dự báo nhiều bước. "
              "Dữ liệu cực đoan được giữ nguyên. Chẩn đoán trên từng chế độ không tự chứng minh tính dừng toàn cục của SETAR.", "",
              "## Chạy lại", "", "Từ thư mục gốc repo:", "", "```powershell", "python -m pip install -r requirements.txt",
              "python code/run_setar_tar.py", "```", "",
              "Hoặc mở `code/05_mo_hinh_setar_tar.ipynb` và Run All. Các notebook 01/02 cũ vẫn có đường dẫn cứng "
              "D:/Chuyên đề 2; notebook 05 đọc trực tiếp dữ liệu đã chia trong repo.", "",
              "## Tài liệu", "",
              "1. Cryer, J. D., & Chan, K.-S. (2008). *Time Series Analysis: With Applications in R*, "
              "2nd ed., Chapter 15: Threshold Models, pp. 383–422. Springer. https://doi.org/10.1007/978-0-387-75959-3.",
              "2. tsDyn documentation: Self Threshold Autoregressive model. https://search.r-project.org/CRAN/refmans/tsDyn/html/setar.html.",
              "3. tsDyn documentation: Automatic selection of SETAR hyper-parameters. https://search.r-project.org/CRAN/refmans/tsDyn/html/selectSETAR.html.", "",
              "Triển khai Python dùng mô hình hai chế độ và CLS theo tài liệu; không tuyên bố tái hiện chính xác phần mềm "
              "tsDyn/TSA hay thí nghiệm gốc trong sách. Lưới, tiêu chí validation và đối chứng là thiết kế của dự án này.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output

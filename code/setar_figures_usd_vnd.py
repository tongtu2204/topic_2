"""Figures and a Vietnamese results note for the SETAR experiment."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf


COLORS = {"SETAR": "#b45309", "AR": "#2563eb", "Mean": "#0f766e", "Zero": "#64748b"}


def make_figures(base, splits, selected, grid, profiles, evaluation, tables):
    output = Path(base)/"figures"/"model_4_setar_tar"
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
                                        [winner, evaluation["SETAR"]], ["Train (2016-2024)", "Train + validation (2016-2025)"]):
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

    price_scores = evaluation["price_metrics"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, metric, label in zip(axes.flat,
            ("RMSE", "MAE", "MAPE_percent", "sMAPE_percent"),
            ("RMSE (VND/USD)", "MAE (VND/USD)", "MAPE (%)", "sMAPE (%)")):
        subset = price_scores[price_scores.Split=="test"].set_index("Model").loc[kinds]
        bars = ax.bar(kinds, subset[metric], color=[COLORS[k] for k in kinds], width=.65)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set(title=f"Test - {label}", ylabel=label)
        ax.set_ylim(0, subset[metric].max()*1.18)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle("Common price-scale metrics for cross-model comparison", fontweight="bold")
    save(fig, "08_common_price_metrics.png")

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.fill_between(test.Date, test.SETAR_price_pi_lower, test.SETAR_price_pi_upper,
                    color=COLORS["SETAR"], alpha=.18, label="SETAR empirical 95% PI")
    ax.plot(test.Date, test.actual_close, color="#334155", lw=1.2, label="Actual close")
    ax.plot(test.Date, test.SETAR_price_point, color=COLORS["SETAR"], lw=1.0,
            label="SETAR one-step price point")
    coverage = evaluation["interval_metrics"].query(
        "Split == 'test' and Scale == 'price_VND_per_USD'").PI_coverage_percent.iloc[0]
    ax.set(title=f"SETAR test price forecast and empirical 95% PI (coverage {coverage:.2f}%)",
           xlabel="Date", ylabel="VND per USD")
    ax.legend(loc="upper left")
    ax.grid(alpha=.2)
    save(fig, "09_test_price_prediction_interval.png")
    return figures


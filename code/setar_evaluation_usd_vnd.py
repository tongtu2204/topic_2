"""Complete USD/VND evaluation; selection remains in setar_tar_usd_vnd.

Test observations update lags, never fitted parameters or interval quantiles.
Empirical residual smearing converts a return conditional mean to a price
conditional mean without assuming Gaussian innovations.
"""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera

from setar_tar_usd_vnd import forecast_one_step


def audit_inputs(base, splits):
    rows = []
    for name, frame in splits.items():
        prices = pd.read_csv(Path(base)/"data"/"processed"/f"price_{name}.csv", parse_dates=["Date"])
        for data in (prices, frame):
            if data.empty or data.Date.isna().any() or data.Date.duplicated().any():
                raise ValueError(f"Invalid dates/empty split: {name}")
            if not data.Date.is_monotonic_increasing or not np.isfinite(data.Close).all() or (data.Close <= 0).any():
                raise ValueError(f"Invalid price/order: {name}")
        joined = frame.merge(prices[["Date", "Close"]], on="Date", validate="one_to_one", suffixes=("", "_price"))
        if len(joined) != len(frame) or not np.array_equal(joined.Close, joined.Close_price):
            raise ValueError(f"Return/price alignment differs: {name}")
        invalid_ohlc = ((prices.High < prices.Open) | (prices.High < prices.Close)
                        | (prices.High < prices.Low) | (prices.Low > prices.Open) | (prices.Low > prices.Close))
        rows.append({"Split": name, "Price_N": len(prices), "Return_N": len(frame),
                     "Start": prices.Date.min(), "End": prices.Date.max(),
                     "OHLC_flagged_retained": int(invalid_ohlc.sum()), "Close_missing": int(prices.Close.isna().sum())})
    all_price = pd.concat([pd.read_csv(Path(base)/"data"/"processed"/f"price_{s}.csv")
                           for s in ("train", "validation", "test")], ignore_index=True)
    all_return = pd.concat(list(splits.values()), ignore_index=True)
    expected = pd.DataFrame({"Date": pd.to_datetime(all_price.Date.iloc[1:]).to_numpy(),
                             "computed": 100*np.log(all_price.Close.to_numpy()[1:]/all_price.Close.to_numpy()[:-1])})
    checked = all_return.merge(expected, on="Date", validate="one_to_one")
    if len(checked) != len(all_return) or not np.allclose(checked.log_return, checked.computed, atol=1e-10, rtol=1e-10):
        raise ValueError("Returns are not 100*log(Close[t]/Close[t-1]), including split boundaries")
    return pd.DataFrame(rows)


def full_metrics(actual, predicted, history, naive, previous=None, is_price=False):
    actual, predicted, history, naive = [np.asarray(x, dtype=float) for x in (actual, predicted, history, naive)]
    if actual.shape != predicted.shape or actual.shape != naive.shape or not all(np.isfinite(x).all() for x in (actual, predicted, history, naive)):
        raise ValueError("Nonfinite/misaligned metric inputs")
    error = actual-predicted
    mse, mae = np.mean(error**2), np.mean(np.abs(error))
    scale = np.mean(np.abs(np.diff(history)))
    naive_rmse = np.sqrt(np.mean((actual-naive)**2))
    ss_total = np.sum((actual-actual.mean())**2)
    a_direction = actual if previous is None else actual-np.asarray(previous)
    p_direction = predicted if previous is None else predicted-np.asarray(previous)
    a_sign, p_sign = np.sign(np.round(a_direction, 10)), np.sign(np.round(p_direction, 10))
    nonzero = a_sign != 0
    recalls = [np.mean(p_sign[a_sign == s] == s) for s in (-1, 0, 1) if np.any(a_sign == s)]
    scores = {"N": len(actual), "MAE": float(mae), "MSE": float(mse), "RMSE": float(np.sqrt(mse)),
              "Median_AE": float(np.median(np.abs(error))), "Max_AE": float(np.max(np.abs(error))),
              "Bias_actual_minus_forecast": float(error.mean()),
              "R2_out_of_sample": float(1-np.sum(error**2)/ss_total) if ss_total > 0 else np.nan,
              "MASE": float(mae/scale) if scale > 0 else np.nan,
              "MASE_training_scale": float(scale), "Naive_RMSE": float(naive_rmse),
              "Relative_RMSE_vs_Naive": float(np.sqrt(mse)/naive_rmse) if naive_rmse > 0 else np.nan,
              "RMSE_improvement_vs_Naive_percent": float(100*(1-np.sqrt(mse)/naive_rmse)) if naive_rmse > 0 else np.nan,
              "DA_all_percent": float(100*np.mean(a_sign == p_sign)),
              "DA_nonzero_actual_percent": float(100*np.mean(a_sign[nonzero] == p_sign[nonzero])) if nonzero.any() else np.nan,
              "Balanced_direction_accuracy_percent": float(100*np.mean(recalls)),
              "Actual_down_N": int(np.sum(a_sign < 0)), "Actual_flat_N": int(np.sum(a_sign == 0)), "Actual_up_N": int(np.sum(a_sign > 0)),
              "MAPE_percent": float(100*np.mean(np.abs(error/actual))) if is_price else np.nan,
              "sMAPE_percent": float(100*np.mean(2*np.abs(error)/(np.abs(actual)+np.abs(predicted)))) if is_price else np.nan,
              "WAPE_percent": float(100*np.sum(np.abs(error))/np.sum(np.abs(actual))) if np.abs(actual).sum() else np.nan}
    # Compatibility with plots, which expect direction as a fraction.
    scores["Direction_accuracy"] = scores["DA_all_percent"]/100
    return scores


def interval_scores(actual, lower, upper, alpha=.05):
    actual, lower, upper = [np.asarray(x, dtype=float) for x in (actual, lower, upper)]
    if np.any(lower > upper):
        raise ValueError("Reversed prediction interval")
    width = upper-lower
    score = width + (2/alpha)*(lower-actual)*(actual < lower) + (2/alpha)*(actual-upper)*(actual > upper)
    return {"Nominal_coverage_percent": 100*(1-alpha),
            "PI_coverage_percent": float(100*np.mean((actual >= lower)&(actual <= upper))),
            "Average_PI_width": float(width.mean()), "Mean_interval_score": float(score.mean()),
            "Below_lower_N": int(np.sum(actual < lower)), "Above_upper_N": int(np.sum(actual > upper))}


def loss_comparisons(frame, seed=20261010, draws=2000, block=5):
    """Paired circular moving-block bootstrap and descriptive HAC loss test.

Positive loss difference means SETAR is worse. No threshold-effect test is
claimed: conventional linear-vs-TAR F tests have nonstandard nulls.
"""
    rng = np.random.default_rng(seed)
    n = len(frame)
    starts = rng.integers(0, n, size=(draws, int(np.ceil(n/block))))
    indices = ((starts[..., None] + np.arange(block)) % n).reshape(draws, -1)[:, :n]
    rows = []
    for scale, actual_col, suffix in (("return_percent", "actual_return", ""), ("price_VND_per_USD", "actual_close", "_price_point")):
        actual = frame[actual_col].to_numpy()
        setar_error = actual-frame["SETAR"+suffix].to_numpy()
        for benchmark in ("Zero", "Mean", "AR"):
            base_error = actual-frame[benchmark+suffix].to_numpy()
            for loss in ("Squared", "Absolute"):
                delta = setar_error**2-base_error**2 if loss == "Squared" else np.abs(setar_error)-np.abs(base_error)
                centered = delta-delta.mean()
                bandwidth = min(5, n-1)
                long_variance = np.dot(centered, centered)/n
                for lag in range(1, bandwidth+1):
                    long_variance += 2*(1-lag/(bandwidth+1))*np.dot(centered[lag:], centered[:-lag])/n
                se = np.sqrt(max(long_variance, 0)/n)
                statistic = delta.mean()/se if se > 0 else np.nan
                boot = delta[indices].mean(axis=1)
                ci = np.quantile(boot, [.025, .975])
                rows.append({"Scale": scale, "Benchmark": benchmark, "Loss": loss,
                             "Mean_loss_difference_SETAR_minus_benchmark": float(delta.mean()),
                             "Paired_block_CI95_lower": float(ci[0]), "Paired_block_CI95_upper": float(ci[1]),
                             "HAC_statistic": statistic, "HAC_normal_two_sided_p_descriptive": float(2*norm.sf(abs(statistic))),
                             "Bootstrap_draws": draws, "Block_length": block, "HAC_bandwidth": bandwidth, "Seed": seed})
    return pd.DataFrame(rows)


def enrich_evaluation(base, splits, selected, evaluation):
    return_rows, price_rows, interval_rows, diag_rows = [], [], [], []
    predictions = evaluation["predictions"].copy()
    for split in ("validation", "test"):
        history = splits["train"].log_return.to_numpy()
        if split == "test":
            history = np.r_[history, splits["validation"].log_return.to_numpy()]
        estimation_names = ("train",) if split == "validation" else ("train", "validation")
        history_price = np.concatenate([pd.read_csv(Path(base)/"data"/"processed"/f"price_{s}.csv").Close.to_numpy() for s in estimation_names])
        models = selected if split == "validation" else evaluation
        mask = predictions.Split == split
        frame = predictions.loc[mask].copy()
        for kind in ("Zero", "Mean", "AR", "SETAR"):
            frame[kind+"_price_plugin"] = frame[kind+"_price_point"]
            if kind == "Mean":
                residual = history-history.mean()
                smear = np.full(len(frame), np.mean(np.exp(residual/100)))
            elif kind in ("AR", "SETAR"):
                model = models[kind]
                fitted, fit_regimes, _ = forecast_one_step(model, history[:model.start], history[model.start:])
                residual = history[model.start:]-fitted
                if kind == "SETAR":
                    factors = {r: float(np.mean(np.exp(residual[fit_regimes == r]/100))) for r in ("Low", "High")}
                    smear = np.array([factors[r] for r in frame.SETAR_regime])
                else:
                    smear = np.full(len(frame), np.mean(np.exp(residual/100)))
            else:
                smear = np.ones(len(frame))
            frame[kind+"_smearing_factor"] = smear
            frame[kind+"_price_point"] = frame[kind+"_price_plugin"]*smear
            return_rows.append({"Split": split, "Model": kind, **full_metrics(frame.actual_return, frame[kind], history, frame.Zero)})
            price_rows.append({"Split": split, "Model": kind, "Point_forecast": "empirical_mean" if kind != "Zero" else "previous_close",
                               **full_metrics(frame.actual_close, frame[kind+"_price_point"], history_price, frame.previous_close,
                                              previous=frame.previous_close, is_price=True)})
        for scale, actual_col, lower, upper in (("return_percent", "actual_return", "SETAR_return_pi_lower", "SETAR_return_pi_upper"),
                                                ("price_VND_per_USD", "actual_close", "SETAR_price_pi_lower", "SETAR_price_pi_upper")):
            interval_rows.append({"Split": split, "Model": "SETAR", "Scale": scale, **interval_scores(frame[actual_col], frame[lower], frame[upper])})
        for col in frame.columns:
            predictions.loc[mask, col] = frame[col].to_numpy()
        if split == "test":
            for kind in ("AR", "SETAR"):
                residual = frame.actual_return.to_numpy()-frame[kind].to_numpy()
                for square, label in ((False, "Ljung-Box forecast error"), (True, "Ljung-Box squared forecast error")):
                    result = acorr_ljungbox(residual**2 if square else residual, lags=[5, 10, 20], return_df=True)
                    for lag, row in result.iterrows():
                        diag_rows.append({"Model": kind, "Sample": "test", "Diagnostic": label, "Lag": lag,
                                          "Statistic": row.lb_stat, "p_value": row.lb_pvalue})
                lm, p, _, _ = het_arch(residual, nlags=10)
                jb, jp, _, _ = jarque_bera(residual)
                diag_rows.extend([{"Model": kind, "Sample": "test", "Diagnostic": "ARCH-LM", "Lag": 10, "Statistic": lm, "p_value": p},
                                  {"Model": kind, "Sample": "test", "Diagnostic": "Jarque-Bera", "Lag": np.nan, "Statistic": jb, "p_value": jp}])
    evaluation.update(metrics=pd.DataFrame(return_rows), price_metrics=pd.DataFrame(price_rows),
                      interval_metrics=pd.DataFrame(interval_rows), predictions=predictions,
                      test_diagnostics=pd.DataFrame(diag_rows),
                      loss_comparisons=loss_comparisons(predictions[predictions.Split == "test"]))
    return evaluation


def save_extra(base, splits, audit, evaluation, config):
    output = Path(base)/"results"/"model_4_setar_tar"
    extras = {"data_audit.csv": audit, "test_forecast_error_diagnostics.csv": evaluation["test_diagnostics"],
              "paired_loss_comparisons.csv": evaluation["loss_comparisons"]}
    for filename, frame in extras.items():
        frame.to_csv(output/filename, index=False)
    hashes = {str(p.relative_to(base)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((Path(base)/"data"/"processed").glob("*.csv"))}
    config.update(asset="USD/VND", price_unit="VND per USD", data_sha256=hashes,
                  price_point_forecast="previous_close * exp(predicted_return/100) * empirical regime-specific residual smearing",
                  intervals="95% empirical estimation-sample residual quantiles by regime; no parameter uncertainty; no test calibration",
                  MASE_price="MAE / mean absolute price differences on complete estimation price split",
                  MASE_return="MAE / mean absolute return differences on estimation split (persistence return scale)",
                  return_Relative_RMSE_benchmark="Zero return", price_Relative_RMSE_benchmark="Previous close",
                  direction="3 classes: down/flat/up; also report nonzero-actual and balanced class recall",
                  percentage_metrics="Return MAPE/sMAPE left blank: zero/near-zero signed target; calculated on price only",
                  diagnostics="Descriptive: Ljung-Box/ARCH/JB are not a formal linearity or threshold significance test",
                  bootstrap={"draws":2000, "block_length":5, "seed":20261010}, python=platform.python_version())
    (output/"selected_models.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return extras

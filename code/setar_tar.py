"""Two-regime SETAR estimated by conditional least squares.

The threshold variable is y[t-d], so this is a self-exciting TAR.
All forecast inputs are strictly earlier than the target observation.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.stats.stattools import jarque_bera


@dataclass
class ARModel:
    p: int
    beta: np.ndarray
    start: int
    rss: float
    nobs: int


@dataclass
class SETARModel:
    p: int
    d: int
    threshold: float
    beta_low: np.ndarray
    beta_high: np.ndarray
    start: int
    rss: float
    nobs: int
    n_low: int
    n_high: int


def lag_design(values, p, start=None):
    y = np.asarray(values, dtype=float)
    start = p if start is None else int(start)
    if p < 1 or start < p or len(y) <= start or not np.isfinite(y).all():
        raise ValueError("Invalid order, starting row or nonfinite series.")
    t = np.arange(start, len(y))
    X = np.column_stack([np.ones(len(t))] + [y[t-j] for j in range(1, p+1)])
    return X, y[t], t


def fit_ar(values, p, start=10):
    X, target, _ = lag_design(values, p, start)
    beta, _, rank, _ = np.linalg.lstsq(X, target, rcond=None)
    if rank != X.shape[1]:
        raise ValueError("Rank-deficient AR design.")
    rss = float(np.sum((target - X @ beta)**2))
    return ARModel(p, beta, start, rss, len(target))


def fit_setar(values, p, d, threshold, start=10, trim=0.15):
    if not 1 <= d <= start:
        raise ValueError("Threshold delay must be positive and at most start.")
    values = np.asarray(values, dtype=float)
    X, target, t = lag_design(values, p, start)
    low = values[t-d] <= threshold
    minimum = max(int(np.ceil(trim * len(target))), 5*(p+1))
    if low.sum() < minimum or (~low).sum() < minimum:
        raise ValueError("Too few observations in a regime.")
    betas = []
    for mask in (low, ~low):
        beta, _, rank, _ = np.linalg.lstsq(X[mask], target[mask], rcond=None)
        if rank != p+1:
            raise ValueError("Rank-deficient regime design.")
        betas.append(beta)
    fitted = np.where(low, X @ betas[0], X @ betas[1])
    return SETARModel(p, d, float(threshold), *betas, start,
                      float(np.sum((target-fitted)**2)), len(target),
                      int(low.sum()), int((~low).sum()))


def profile_threshold(values, p, d, start=10, quantiles=None, trim=0.15):
    """Learn threshold on estimation data only, never on validation/test."""
    values = np.asarray(values, dtype=float)
    quantiles = np.linspace(trim, 1-trim, 71) if quantiles is None else quantiles
    z = values[np.arange(start, len(values))-d]
    thresholds = np.unique(np.quantile(z, quantiles))
    records, models = [], []
    for gamma in thresholds:
        try:
            model = fit_setar(values, p, d, gamma, start, trim)
        except ValueError:
            continue
        # Gaussian common-variance information criteria, used descriptively.
        # Count all regression coefficients, one threshold and one variance.
        k = 2*(p+1)+2
        bic = model.nobs*np.log(model.rss/model.nobs) + k*np.log(model.nobs)
        records.append(dict(p=p, d=d, threshold=gamma, train_rss=model.rss,
                            train_bic=bic, n_low=model.n_low, n_high=model.n_high))
        models.append(model)
    if not models:
        raise ValueError("No admissible threshold.")
    winner = min(models, key=lambda model: model.rss)
    return winner, pd.DataFrame(records)


def forecast_one_step(model, history, observations):
    """Fixed parameters; update lags with each realized past observation.

    Forecast j uses history and observations[:j], never observations[j:].
    This is sequential one-step evaluation, not a forecast of an entire block
    from its initial date.
    """
    past = np.asarray(history, dtype=float)
    observed = np.asarray(observations, dtype=float)
    p = model.p
    d = model.d if isinstance(model, SETARModel) else 1
    if len(past) < max(p, d) or not np.isfinite(past).all():
        raise ValueError("Insufficient or nonfinite history.")
    combined = np.concatenate([past, observed])
    t = len(past) + np.arange(len(observed))
    X = np.column_stack([np.ones(len(t))] + [combined[t-j] for j in range(1, p+1)])
    if isinstance(model, SETARModel):
        z = combined[t-d]
        low = z <= model.threshold
        pred = np.where(low, X @ model.beta_low, X @ model.beta_high)
        return pred, np.where(low, "Low", "High"), z
    return X @ model.beta, None, None


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError("Invalid forecast shape or nonfinite prediction.")
    err = actual - predicted
    return {"RMSE": float(np.sqrt(np.mean(err**2))),
            "MAE": float(np.mean(np.abs(err))),
            "Bias_actual_minus_forecast": float(np.mean(err)),
            "Direction_accuracy": float(np.mean(np.sign(actual)==np.sign(predicted)))}


def common_metrics(actual, predicted, mase_scale, naive_rmse, previous=None,
                   percentage_metrics=False):
    """Metrics shared across models on one target scale.

    MASE uses the mean absolute one-step naive error from the estimation
    sample. Relative RMSE uses the RMSE of the same-split naive forecast.
    For prices, ``previous`` supplies the last observed price for direction.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if mase_scale <= 0 or naive_rmse <= 0:
        raise ValueError("MASE scale and naive RMSE must be positive.")
    result = metrics(actual, predicted)
    result["MASE"] = result["MAE"] / mase_scale
    result["Relative_RMSE_vs_Naive"] = result["RMSE"] / naive_rmse
    if previous is not None:
        previous = np.asarray(previous, dtype=float)
        result["Direction_accuracy"] = float(np.mean(
            np.sign(actual-previous) == np.sign(predicted-previous)))
    if percentage_metrics:
        if np.any(actual == 0):
            raise ValueError("MAPE is undefined when actual contains zero.")
        result["MAPE_percent"] = float(100*np.mean(np.abs((actual-predicted)/actual)))
        denominator = np.abs(actual)+np.abs(predicted)
        result["sMAPE_percent"] = float(100*np.mean(
            np.divide(2*np.abs(actual-predicted), denominator,
                      out=np.zeros_like(actual), where=denominator != 0)))
    return result


def setar_residual_quantiles(model, values, probability=0.95):
    """Empirical innovation quantiles per fitted SETAR regime."""
    values = np.asarray(values, dtype=float)
    fitted, regimes, _ = forecast_one_step(model, values[:model.start], values[model.start:])
    residual = values[model.start:] - fitted
    alpha = (1-probability)/2
    quantiles = {}
    for regime in ("Low", "High"):
        sample = residual[regimes == regime]
        if len(sample) < 30:
            raise ValueError("Too few residuals for regime interval.")
        quantiles[regime] = tuple(np.quantile(sample, [alpha, 1-alpha]))
    return quantiles


def load_splits(base):
    splits = {}
    for name in ("train", "validation", "test"):
        frame = pd.read_csv(Path(base)/"data"/"processed"/f"return_{name}.csv",
                            parse_dates=["Date"])
        if not frame.Date.is_monotonic_increasing or frame.Date.duplicated().any():
            raise ValueError("Dates must be sorted and unique.")
        if not np.isfinite(frame.log_return).all():
            raise ValueError("Nonfinite log return.")
        splits[name] = frame
    if not (splits["train"].Date.max() < splits["validation"].Date.min()
            <= splits["validation"].Date.max() < splits["test"].Date.min()):
        raise ValueError("Chronological split overlap.")
    return splits


def select_models(splits, orders=range(1, 11), delays=range(1, 6)):
    """Train threshold per (p,d), then select p,d by validation RMSE."""
    orders, delays = list(orders), list(delays)
    start = max(max(orders), max(delays))
    train = splits["train"].log_return.to_numpy()
    validation = splits["validation"].log_return.to_numpy()
    profiles, rows, models = [], [], {}
    for p in orders:
        ar = fit_ar(train, p, start)
        pred, _, _ = forecast_one_step(ar, train, validation)
        k = p+2
        rows.append({"Model": "AR", "p": p, "d": np.nan, "threshold": np.nan,
                     "train_rss": ar.rss,
                     "train_bic": ar.nobs*np.log(ar.rss/ar.nobs)+k*np.log(ar.nobs),
                     **metrics(validation, pred)})
        models[("AR", p, 0)] = ar
        for d in delays:
            setar, profile = profile_threshold(train, p, d, start)
            profiles.append(profile)
            pred, _, _ = forecast_one_step(setar, train, validation)
            k = 2*(p+1)+2
            rows.append({"Model": "SETAR", "p": p, "d": d,
                         "threshold": setar.threshold, "train_rss": setar.rss,
                         "train_bic": setar.nobs*np.log(setar.rss/setar.nobs)+k*np.log(setar.nobs),
                         **metrics(validation, pred)})
            models[("SETAR", p, d)] = setar
    grid = pd.DataFrame(rows)
    selected = {}
    for kind in ("AR", "SETAR"):
        row = grid[grid.Model==kind].sort_values(["RMSE", "p", "d"], na_position="last").iloc[0]
        selected[kind] = models[(kind, int(row.p), 0 if kind=="AR" else int(row.d))]
    return selected, grid, pd.concat(profiles, ignore_index=True)


def evaluate(splits, selected):
    train = splits["train"].log_return.to_numpy()
    val = splits["validation"].log_return.to_numpy()
    test = splits["test"].log_return.to_numpy()
    train_val = np.concatenate([train, val])
    ar0, setar0 = selected["AR"], selected["SETAR"]
    ar = fit_ar(train_val, ar0.p, ar0.start)
    setar, refit_profile = profile_threshold(train_val, setar0.p, setar0.d, setar0.start)
    all_metrics, price_metrics, interval_metrics, predictions = [], [], [], []
    for name, history, actual, models in (
            ("validation", train, val, selected),
            ("test", train_val, test, {"AR": ar, "SETAR": setar})):
        frame = pd.DataFrame({"Date": splits[name].Date.to_numpy(),
                              "actual_return": actual, "actual_close": splits[name].Close.to_numpy()})
        for kind in ("Zero", "Mean", "AR", "SETAR"):
            if kind=="Zero":
                predicted = np.zeros(len(actual))
            elif kind=="Mean":
                predicted = np.full(len(actual), history.mean())
            else:
                predicted, regimes, z = forecast_one_step(models[kind], history, actual)
                if kind=="SETAR":
                    frame["threshold_variable"] = z
                    frame["SETAR_regime"] = regimes
            frame[kind] = predicted
        # Back-transform to a price point forecast using the observed prior close.
        # This is NOT the conditional mean of price (no Jensen correction).
        previous_close = np.r_[splits["train" if name=="validation" else "validation"].Close.iloc[-1],
                               splits[name].Close.to_numpy()[:-1]]
        frame["previous_close"] = previous_close
        for kind in ("Zero", "Mean", "AR", "SETAR"):
            frame[f"{kind}_price_point"] = previous_close*np.exp(frame[kind]/100)

        history_close = splits["train"].Close.to_numpy() if name == "validation" else np.concatenate([
            splits["train"].Close.to_numpy(), splits["validation"].Close.to_numpy()])
        return_mase_scale = float(np.mean(np.abs(np.diff(history))))
        price_mase_scale = float(np.mean(np.abs(np.diff(history_close))))
        naive_return_rmse = metrics(actual, frame["Zero"])["RMSE"]
        naive_price_rmse = metrics(frame.actual_close, frame.Zero_price_point)["RMSE"]
        for kind in ("Zero", "Mean", "AR", "SETAR"):
            return_scores = common_metrics(actual, frame[kind], return_mase_scale,
                                           naive_return_rmse)
            price_scores = common_metrics(frame.actual_close, frame[f"{kind}_price_point"],
                                          price_mase_scale, naive_price_rmse,
                                          previous=previous_close, percentage_metrics=True)
            if kind == "Zero":
                return_scores["Direction_accuracy"] = np.nan
                price_scores["Direction_accuracy"] = np.nan
            all_metrics.append({"Split": name, "Model": kind, "n": len(actual), **return_scores})
            price_metrics.append({"Split": name, "Model": kind, "n": len(actual), **price_scores})

        # Approximate 95% one-step intervals from empirical in-sample residual
        # quantiles within each SETAR regime. These intervals use estimation
        # data only and include innovation uncertainty, not parameter uncertainty.
        interval_model = models["SETAR"]
        quantiles = setar_residual_quantiles(interval_model, history)
        q_low = np.array([quantiles[r][0] for r in frame.SETAR_regime])
        q_high = np.array([quantiles[r][1] for r in frame.SETAR_regime])
        frame["SETAR_return_pi_lower"] = frame.SETAR + q_low
        frame["SETAR_return_pi_upper"] = frame.SETAR + q_high
        frame["SETAR_price_pi_lower"] = previous_close*np.exp(frame.SETAR_return_pi_lower/100)
        frame["SETAR_price_pi_upper"] = previous_close*np.exp(frame.SETAR_return_pi_upper/100)
        for scale, actual_col, lower_col, upper_col in (
                ("return_percent", "actual_return", "SETAR_return_pi_lower", "SETAR_return_pi_upper"),
                ("price_USD_per_ounce", "actual_close", "SETAR_price_pi_lower", "SETAR_price_pi_upper")):
            covered = ((frame[actual_col] >= frame[lower_col]) &
                       (frame[actual_col] <= frame[upper_col]))
            interval_metrics.append({"Split": name, "Model": "SETAR", "Scale": scale,
                                     "Nominal_coverage_percent": 95.0,
                                     "PI_coverage_percent": float(100*covered.mean()),
                                     "Average_PI_width": float((frame[upper_col]-frame[lower_col]).mean())})
        predictions.append(frame.assign(Split=name))
    return {"metrics": pd.DataFrame(all_metrics),
            "price_metrics": pd.DataFrame(price_metrics),
            "interval_metrics": pd.DataFrame(interval_metrics),
            "predictions": pd.concat(predictions, ignore_index=True),
            "AR": ar, "SETAR": setar, "refit_profile": refit_profile,
            "train_val": train_val}


def coefficient_table(selected, evaluation):
    rows = []
    for stage, models in (("train", selected), ("train_validation", evaluation)):
        for kind in ("AR", "SETAR"):
            model = models[kind]
            regimes = [("All", model.beta)] if kind=="AR" else [
                ("Low", model.beta_low), ("High", model.beta_high)]
            for regime, beta in regimes:
                for j, value in enumerate(beta):
                    rows.append({"Fit_data": stage, "Model": kind, "Regime": regime,
                                 "Term": "Intercept" if j==0 else f"lag_{j}", "Estimate": value})
    return pd.DataFrame(rows)


def residual_diagnostics(evaluation):
    rows, residual_frames = [], []
    y = evaluation["train_val"]
    for kind in ("AR", "SETAR"):
        model = evaluation[kind]
        fitted, _, _ = forecast_one_step(model, y[:model.start], y[model.start:])
        residual = y[model.start:] - fitted
        lb = acorr_ljungbox(residual, lags=[5, 10, 20], model_df=0, return_df=True)
        lb2 = acorr_ljungbox(residual**2, lags=[5, 10, 20], model_df=0, return_df=True)
        for lag in (5, 10, 20):
            rows.append({"Model": kind, "Test": "Ljung-Box residual", "Lag": lag,
                         "Statistic": lb.loc[lag, "lb_stat"], "p_value": lb.loc[lag, "lb_pvalue"]})
            rows.append({"Model": kind, "Test": "Ljung-Box squared residual", "Lag": lag,
                         "Statistic": lb2.loc[lag, "lb_stat"], "p_value": lb2.loc[lag, "lb_pvalue"]})
        ncoeff = model.p+1 if kind=="AR" else 2*(model.p+1)
        lm, pvalue, _, _ = het_arch(residual, nlags=10, ddof=ncoeff)
        rows.append({"Model": kind, "Test": "ARCH-LM residual", "Lag": 10,
                     "Statistic": lm, "p_value": pvalue})
        jb, jb_pvalue, skew, kurtosis = jarque_bera(residual)
        rows.append({"Model": kind, "Test": "Jarque-Bera residual", "Lag": np.nan,
                     "Statistic": jb, "p_value": jb_pvalue,
                     "Skewness": skew, "Kurtosis": kurtosis})
        residual_frames.append(pd.DataFrame({"Model": kind, "position": np.arange(model.start, len(y)),
                                            "residual": residual}))
    return pd.DataFrame(rows), pd.concat(residual_frames, ignore_index=True)


def save_tables(base, splits, selected, grid, profiles, evaluation):
    output = Path(base)/"results"/"setar_tar"
    output.mkdir(parents=True, exist_ok=True)
    diagnostics, residuals = residual_diagnostics(evaluation)
    tables = {"validation_grid.csv": grid, "threshold_search_train.csv": profiles,
              "threshold_search_refit.csv": evaluation["refit_profile"],
              "forecast_metrics.csv": evaluation["metrics"],
              "price_forecast_metrics.csv": evaluation["price_metrics"],
              "setar_interval_metrics.csv": evaluation["interval_metrics"],
              "forecasts.csv": evaluation["predictions"],
              "coefficients.csv": coefficient_table(selected, evaluation),
              "residual_diagnostics.csv": diagnostics, "residuals.csv": residuals}
    regime_rows = []
    for (split, regime), frame in evaluation["predictions"].groupby(["Split", "SETAR_regime"]):
        regime_rows.append({"Split": split, "Regime": regime, "n": len(frame),
                            "Actual_mean": frame.actual_return.mean(),
                            "Actual_std": frame.actual_return.std(),
                            **metrics(frame.actual_return, frame.SETAR)})
    tables["regime_summary.csv"] = pd.DataFrame(regime_rows)
    for filename, frame in tables.items():
        frame.to_csv(output/filename, index=False)
    config = {"protocol": "fixed parameters, sequential one-step forecast using realized past returns",
              "selection": "p,d by validation RMSE; threshold minimizes estimation-sample SSR",
              "orders": list(range(1, 11)), "delays": list(range(1, 6)),
              "threshold_quantiles": [0.15, 0.85], "threshold_grid_points": 71,
              "minimum_regime_fraction": 0.15, "common_fit_start": selected["AR"].start,
              "unit": "log_return = 100 * log(Close[t]/Close[t-1])",
              "splits": {name: {"n": len(frame), "start": str(frame.Date.min().date()),
                                 "end": str(frame.Date.max().date())} for name, frame in splits.items()},
              "models": {}}
    for stage, models in (("train", selected), ("train_validation", evaluation)):
        config["models"][stage] = {}
        for kind in ("AR", "SETAR"):
            model = models[kind]
            description = {"p": model.p, "nobs": model.nobs, "rss": model.rss}
            if kind=="SETAR":
                description.update(d=model.d, threshold=model.threshold,
                                   n_low=model.n_low, n_high=model.n_high,
                                   beta_low=model.beta_low.tolist(), beta_high=model.beta_high.tolist())
            else:
                description["beta"] = model.beta.tolist()
            config["models"][stage][kind] = description
    (output/"selected_models.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return tables, config

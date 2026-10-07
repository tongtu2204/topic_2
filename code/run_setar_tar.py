"""Run the SETAR experiment from any working directory."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from setar_tar import load_splits, select_models, evaluate, save_tables
from setar_figures import make_figures, write_results_note


def main():
    base = Path(__file__).resolve().parent.parent
    splits = load_splits(base)
    selected, grid, profiles = select_models(splits)
    evaluation = evaluate(splits, selected)
    tables, config = save_tables(base, splits, selected, grid, profiles, evaluation)
    figures = make_figures(base, splits, selected, grid, profiles, evaluation, tables)
    for figure in figures:
        plt.close(figure)
    report = write_results_note(base, config, evaluation, tables)
    print("\nRETURN METRICS")
    print(evaluation["metrics"].to_string(index=False))
    print("\nPRICE METRICS FOR CROSS-MODEL COMPARISON")
    print(evaluation["price_metrics"].to_string(index=False))
    print("\nSETAR 95% PREDICTION INTERVAL")
    print(evaluation["interval_metrics"].to_string(index=False))
    print(f"SETAR: p={selected['SETAR'].p}, d={selected['SETAR'].d}")
    print(f"Results note: {report.relative_to(base)}")


if __name__ == "__main__":
    main()

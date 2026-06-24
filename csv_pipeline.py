"""
csv_pipeline.py
===============
Fractal-Driven Predictive System — Universal CSV Input
========================================================

UPGRADE: Feed ANY CSV file with a time column and a numeric column,
and the full Phase 1-5 pipeline (normalise -> train NN -> run IFS ->
project forward) runs automatically and adapts to that dataset.

No code changes needed per dataset. Only the CSV path and column name
change.

Usage
-----
# Terminal:
    python csv_pipeline.py --csv data/my_data.csv --col my_value_column

# Python / Jupyter / Colab:
    from csv_pipeline import run_csv_pipeline
    results = run_csv_pipeline("data/my_data.csv", value_col="my_value_column")
"""

import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

warnings.filterwarnings("ignore")

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "modules"))

from neural_net import NeuralNetwork
from ifs_engine import IFSPredictor

OUTPUT = _HERE / "outputs"
OUTPUT.mkdir(exist_ok=True)

PALETTE = ["#1D9E75", "#D85A30", "#3B8BD4", "#7F77DD",
           "#EF9F27", "#C0392B", "#2980B9", "#E67E22"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F7F4",
    "axes.edgecolor":   "#BDC3C7",
    "axes.grid":        True,
    "grid.color":       "#E8E7E0",
    "grid.linewidth":   0.5,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "sans-serif",
    "font.size":        11,
})


# ══════════════════════════════════════════════════════════════════════════
# STEP 1 — CSV LOADER (adapts automatically to whatever is in the file)
# ══════════════════════════════════════════════════════════════════════════
def load_csv(filepath: str, year_col: str = "year", value_col: str = None):
    """
    Load any CSV. Auto-detects the time column and, if not specified,
    auto-picks the first numeric column to predict.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}\nLooked at: {path.resolve()}")

    df = pd.read_csv(filepath)
    print(f"\n  Loaded: {path.name}")
    print(f"  Shape : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # Resolve time column
    if year_col not in df.columns:
        for candidate in ["Year", "YEAR", "date", "Date", "time", "Time", "index"]:
            if candidate in df.columns:
                year_col = candidate
                break
        else:
            df["year"] = range(len(df))
            year_col = "year"
            print("  Note: no time column found - using row index instead")

    # Resolve value column
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != year_col]
    if value_col is None:
        value_col = numeric_cols[0]
        print(f"  Auto-selected column to predict: '{value_col}'")
    elif value_col not in df.columns:
        raise ValueError(f"Column '{value_col}' not found. Available: {numeric_cols}")

    sub = df[[year_col, value_col]].dropna()
    years  = sub[year_col].astype(float).values
    values = sub[value_col].astype(float).values

    print(f"  Using column : '{value_col}'")
    print(f"  Time range   : {years[0]:.0f} to {years[-1]:.0f}  ({len(years)} points)")
    print(f"  Value range  : [{values.min():.3f}, {values.max():.3f}]")

    if len(values) < 6:
        raise ValueError(f"Only {len(values)} valid points - need at least 6.")

    return years, values, value_col, df


def normalise(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


# ══════════════════════════════════════════════════════════════════════════
# STEP 2 — TRAIN, RUN IFS, PROJECT  (identical logic regardless of dataset)
# ══════════════════════════════════════════════════════════════════════════
def train_neural_network(normed, epochs=600, lr=0.01, verbose=True):
    nn = NeuralNetwork(lr=lr, seed=42)
    print(f"\n  Training NN (2->6->1)  epochs={epochs}  lr={lr}")
    loss_hist = nn.train(normed, epochs=epochs, verbose=verbose,
                         log_every=max(1, epochs // 6))
    pred_series = nn.predict_series(normed)
    next_pred = nn.predict_next(normed)
    print(f"  Final MSE: {loss_hist[-1]:.5f}   Next-step (norm): {next_pred:.4f}")
    return nn, loss_hist, pred_series, next_pred


def run_ifs_feedback(normed, nn_pred, n_particles=250, n_steps=250,
                     noise=0.02, data_weight=0.50, nn_weight=0.35):
    ifs = IFSPredictor(n_particles=n_particles, noise=noise,
                       data_weight=data_weight, nn_weight=nn_weight, seed=7)
    ifs.set_data_anchors(normed)
    ifs.set_nn_anchor(nn_pred)
    for _ in range(n_steps):
        ifs.step()
    print(f"\n  IFS convergence : {ifs.convergence:.0%}")
    print(f"  Attractor centre: {ifs.attractor_centre.round(3)}")
    print(f"  Attractor spread: {ifs.attractor_spread:.4f}"
          + ("  (tight)" if ifs.attractor_spread < 0.15 else "  (wide - higher uncertainty)"))
    return ifs


def project_forward(nn, normed, raw, years, n_future=5):
    mn, mx = raw.min(), raw.max()
    window = list(normed[-2:])
    fut_years = np.array([years[-1] + i + 1 for i in range(n_future)])
    fut_norm, fut_lo, fut_hi = [], [], []
    for step in range(n_future):
        pr = nn.predict_next(np.array(window))
        u = 0.05 * (step + 1)
        fut_norm.append(pr)
        fut_lo.append(max(0.0, pr - u))
        fut_hi.append(min(1.0, pr + u))
        window = [window[-1], pr]
    fut_vals = np.array(fut_norm) * (mx - mn) + mn
    fut_lo   = np.array(fut_lo)   * (mx - mn) + mn
    fut_hi   = np.array(fut_hi)   * (mx - mn) + mn

    print(f"\n  {'Period':<10}{'Projected':>12}{'Low':>10}{'High':>10}")
    print(f"  {'-'*42}")
    for yr, v, lo, hi in zip(fut_years, fut_vals, fut_lo, fut_hi):
        print(f"  {yr:<10.0f}{v:>12.3f}{lo:>10.3f}{hi:>10.3f}")

    return {"years": fut_years, "values": fut_vals, "lower": fut_lo, "upper": fut_hi}


# ══════════════════════════════════════════════════════════════════════════
# STEP 3 — PLOT (auto-adapts title, axis labels, colour to the dataset)
# ══════════════════════════════════════════════════════════════════════════
def plot_results(years, raw, normed, value_col, nn, loss_hist, pred_series,
                 ifs, projection, color="#1D9E75", save=True):
    mn, mx = raw.min(), raw.max()
    pred_denorm = np.where(np.isnan(pred_series), np.nan, pred_series * (mx - mn) + mn)

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    fig.suptitle(
        f"Fractal-Driven Predictive System  -  '{value_col}'\n"
        f"Data: {years[0]:.0f}-{years[-1]:.0f}  |  Architecture: 2->6->1  |  "
        f"Final MSE: {loss_hist[-1]:.4f}",
        fontsize=13, y=1.02
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(years, raw, color=color, lw=1.8)
    ax1.set_title(f"Input signal: '{value_col}'")
    ax1.set_xlabel("Time"); ax1.set_ylabel("Value")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(years, raw, color=color, lw=1.8, label="Actual", zorder=3)
    valid = ~np.isnan(pred_denorm)
    ax2.plot(years[valid], pred_denorm[valid], color="#EF9F27",
             lw=1.5, ls="--", label="NN prediction", zorder=4)
    ax2.set_title("Neural network: prediction vs actual")
    ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(years, raw, color=color, lw=1.8, label="Historical")
    ax3.plot(projection["years"], projection["values"], "D-",
             color="#2C3E50", lw=2, ms=6, label="Projection")
    ax3.fill_between(projection["years"], projection["lower"], projection["upper"],
                     alpha=0.18, color="#2C3E50", label="Uncertainty")
    ax3.axvline(years[-1], color="#888780", lw=0.8, ls=":")
    ax3.set_title("Historical + forward projection")
    ax3.legend(fontsize=9)

    ax4 = fig.add_subplot(gs[1, 0])
    pts = ifs.particles
    conv = ifs.particle_age > 50
    ax4.scatter(pts[~conv, 0], pts[~conv, 1], s=1.5, alpha=0.4, c="#E67E22", label="Transient")
    ax4.scatter(pts[conv, 0], pts[conv, 1], s=1.5, alpha=0.55, c=color, label="Converged")
    ax4.scatter(ifs.anchors[:, 0], ifs.anchors[:, 1], s=14, color="#7F77DD",
                alpha=0.5, zorder=5, label="Data anchors")
    ax4.scatter(*ifs.nn_anchor, s=160, color="#EF9F27", zorder=10,
                edgecolors="white", lw=0.8, label=f"NN anchor")
    ax4.set_title(f"IFS attractor  (conv={ifs.convergence:.0%}, spread={ifs.attractor_spread:.3f})")
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8)

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.semilogy(loss_hist, color="#C0392B", lw=1.5)
    ax5.axhline(0.01, color="#888780", ls="--", lw=0.8)
    ax5.set_title("Training loss (log scale)")

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    next_denorm = ifs.nn_anchor[0] * (mx - mn) + mn
    rows = [
        ["Metric", "Value"],
        ["Column", value_col],
        ["Points", str(len(years))],
        ["Final MSE", f"{loss_hist[-1]:.5f}"],
        ["Next-step (raw)", f"{next_denorm:.3f}"],
        ["Convergence", f"{ifs.convergence:.0%}"],
        ["Spread", f"{ifs.attractor_spread:.4f}"],
    ]
    tbl = ax6.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.3)

    plt.tight_layout()
    if save:
        safe = value_col.replace(" ", "_").replace("/", "_")
        path = OUTPUT / f"csv_{safe}.png"
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Figure saved -> {path}")
        return str(path)
    plt.show()
    return ""


def save_projection_csv(projection, value_col, years, raw):
    safe = value_col.replace(" ", "_").replace("/", "_")
    out = OUTPUT / f"projection_{safe}.csv"
    pd.DataFrame({
        "period": projection["years"],
        "projected": projection["values"].round(4),
        "lower_bound": projection["lower"].round(4),
        "upper_bound": projection["upper"].round(4),
        "source_column": value_col,
        "last_known_period": years[-1],
        "last_known_value": raw[-1],
    }).to_csv(out, index=False)
    print(f"  Projection CSV -> {out}")
    return str(out)


# ══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT — this is what users actually call
# ══════════════════════════════════════════════════════════════════════════
def run_csv_pipeline(csv_path, value_col=None, year_col="year",
                     epochs=600, lr=0.01, n_future=5,
                     n_particles=250, n_steps=250, noise=0.02,
                     data_weight=0.50, nn_weight=0.35,
                     color=None, save_fig=True, save_proj=True):
    print("\n" + "="*60)
    print("  FRACTAL-DRIVEN PREDICTIVE SYSTEM - CSV Pipeline")
    print("="*60)

    if color is None:
        import hashlib
        idx = int(hashlib.md5(csv_path.encode()).hexdigest(), 16) % len(PALETTE)
        color = PALETTE[idx]

    years, raw, col_used, df = load_csv(csv_path, year_col, value_col)
    normed = normalise(raw)

    nn, loss_hist, pred_series, next_pred = train_neural_network(normed, epochs=epochs, lr=lr)
    ifs = run_ifs_feedback(normed, next_pred, n_particles=n_particles, n_steps=n_steps,
                           noise=noise, data_weight=data_weight, nn_weight=nn_weight)
    projection = project_forward(nn, normed, raw, years, n_future=n_future)

    fig_path = plot_results(years, raw, normed, col_used, nn, loss_hist,
                            pred_series, ifs, projection, color=color, save=save_fig)
    proj_path = save_projection_csv(projection, col_used, years, raw) if save_proj else ""

    print("\n  DONE.")
    print("="*60)

    return {"projection": projection, "nn": nn, "ifs": ifs,
            "loss_history": loss_hist, "years": years, "raw": raw, "column": col_used}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fractal-Driven Predictive System - CSV Pipeline")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--col", default=None)
    parser.add_argument("--year-col", default="year")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--future", type=int, default=5)
    parser.add_argument("--particles", type=int, default=250)
    parser.add_argument("--color", default=None)
    args = parser.parse_args()

    run_csv_pipeline(csv_path=args.csv, value_col=args.col, year_col=args.year_col,
                     epochs=args.epochs, lr=args.lr, n_future=args.future,
                     n_particles=args.particles, color=args.color)

#!/usr/bin/env python3
"""
STP plotting tool focused on the core comparisons requested:

- Bellman–Ford vs Floyd–Warshall runtimes as V changes (linear + log plots)
- Runtime distributions to show run-to-run variance
- Measured runtimes vs theoretical complexity for each algorithm (linear + log)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

_REGIME_COLORS: Dict[str, str] = {
    "sparse": "tab:blue",
    "dense": "tab:orange",
}


def _format_regime_label(regime: str) -> str:
    """Return a human-readable label for a regime."""
    if not regime:
        return "Unknown"
    txt = str(regime).strip()
    return txt.capitalize() if txt else "Unknown"


def _regime_color(regime: str) -> str | None:
    """Pick a consistent color per regime (case-insensitive)."""
    if not regime:
        return None
    return _REGIME_COLORS.get(str(regime).strip().lower())


def _regime_slug(regime: str) -> str:
    """Slugified regime name for filenames."""
    if not regime:
        return "unknown"
    slug = "".join(c if c.isalnum() else "_" for c in str(regime).strip().lower())
    slug = slug.strip("_")
    return slug or "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _save(path_no_ext: str) -> None:
    """Save the current figure as PNG+SVG and close it."""
    plt.tight_layout()
    plt.savefig(path_no_ext + ".png", dpi=180)
    plt.savefig(path_no_ext + ".svg")
    plt.close()


def _prep_axes(logx: bool = False, logy: bool = False) -> None:
    """Apply consistent axes configuration."""
    ax = plt.gca()
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="both", alpha=0.25)


def _annotate_label(label: str, log_scale: bool) -> str:
    """Append '(log scale)' when plotting on a logarithmic axis."""
    if log_scale and "log scale" not in label.lower():
        return f"{label} (log scale)"
    return label


def _format_v_label(v: float) -> str:
    """Pretty formatting for V values in labels."""
    if pd.isna(v):
        return "?"
    try:
        vf = float(v)
        if float(vf).is_integer():
            return str(int(vf))
        return f"{vf:.3g}"
    except Exception:
        return str(v)


def _mean_time_by_v(df: pd.DataFrame, time_col: str, include_E: bool = False) -> pd.DataFrame:
    """Return per-(regime,V) mean runtime (and optional mean E) for a timing column."""
    if time_col not in df.columns:
        return pd.DataFrame(columns=["regime", "V", "mean_time"])

    sub = df[df[time_col].notna()].copy()
    if sub.empty or "V" not in sub.columns:
        return pd.DataFrame(columns=["regime", "V", "mean_time"])

    sub = sub[sub["V"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=["regime", "V", "mean_time"])

    if "regime" in sub.columns:
        sub["regime"] = sub["regime"].fillna("unknown").astype(str)
    else:
        sub["regime"] = "all"

    agg_dict: Dict[str, Tuple[str, str]] = {"mean_time": (time_col, "mean")}
    if include_E and "E" in sub.columns:
        agg_dict["mean_E"] = ("E", "mean")

    grouped = (
        sub.groupby(["regime", "V"], sort=True)
        .agg(**agg_dict)
        .reset_index()
        .sort_values(["regime", "V"])
    )
    return grouped


# ---------------------------------------------------------------------------
# Plotting primitives
# ---------------------------------------------------------------------------
def plot_time_vs_v(df: pd.DataFrame, base_path: str, title_prefix: str, show_plots: bool) -> None:
    """Plot BF vs FW runtimes against V (linear + log) split by regime."""
    configs: List[Tuple[str, str, str]] = [
        ("Bellman–Ford", "bf_time_s", "o-"),
        ("Floyd–Warshall", "fw_time_s", "s--"),
    ]

    series: List[Tuple[str, str, pd.DataFrame]] = []
    for label, col, style in configs:
        stats = _mean_time_by_v(df, col)
        if stats.empty:
            continue
        series.append((label, style, stats))

    if not series:
        print("[warn] No runtime data for BF or FW; skipping time vs V plots.", file=sys.stderr)
        return

    for log_scale, suffix in [(False, "linear"), (True, "log")]:
        plt.figure()
        for label, style, stats in series:
            for regime, sub in stats.groupby("regime"):
                sub = sub.sort_values("V")
                if sub.empty:
                    continue
                if log_scale:
                    sub_plot = sub[(sub["V"] > 0) & (sub["mean_time"] > 0)]
                    if sub_plot.empty:
                        continue
                else:
                    sub_plot = sub
                color = _regime_color(regime)
                regime_label = _format_regime_label(regime)
                kwargs = {"label": f"{label} ({regime_label})"}
                if color:
                    kwargs["color"] = color
                plt.plot(
                    sub_plot["V"],
                    sub_plot["mean_time"],
                    style,
                    **kwargs,
                )

        logx = logy = log_scale
        _prep_axes(logx=logx, logy=logy)
        plt.xlabel(_annotate_label("V", logx))
        plt.ylabel(_annotate_label("Mean runtime (s)", logy))
        plt.title(f"{title_prefix}: BF vs FW — Time vs V ({suffix})")
        plt.legend()
        _save(os.path.join(base_path, f"time_vs_V_{suffix}"))
        if show_plots:
            plt.show()


def plot_runtime_distributions(df: pd.DataFrame, base_path: str, title_prefix: str, show_plots: bool) -> None:
    """Box plots showing runtime variance for median/maximum V per regime."""
    configs = [
        ("Bellman–Ford", "bf_time_s", "bf_runtime_distribution"),
        ("Floyd–Warshall", "fw_time_s", "fw_runtime_distribution"),
    ]

    for label, col, filename in configs:
        if col not in df.columns or "V" not in df.columns:
            continue

        sub = df[df[col].notna()].copy()
        if sub.empty:
            continue

        if "regime" in sub.columns:
            sub["regime"] = sub["regime"].fillna("unknown").astype(str)
        else:
            sub["regime"] = "all"

        data = []
        labels = []
        colors = []

        for regime in sorted(sub["regime"].unique()):
            sub_reg = sub[sub["regime"] == regime]
            unique_vs = np.sort(sub_reg["V"].dropna().unique())
            if unique_vs.size == 0:
                continue
            largest_v = unique_vs[-1]
            median_v = unique_vs[unique_vs.size // 2]
            for v in sorted({median_v, largest_v}):
                vals = sub_reg[sub_reg["V"] == v][col].dropna()
                if vals.empty:
                    continue
                data.append(vals)
                labels.append(f"{_format_regime_label(regime)} V={_format_v_label(v)}")
                colors.append(_regime_color(regime))

        if not data:
            continue

        plt.figure()
        bp = plt.boxplot(data, labels=labels, showmeans=True, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color or "lightgray")

        _prep_axes(logx=False, logy=False)
        plt.ylabel("Runtime (s)")
        plt.title(
            f"{title_prefix}: {label} runtime distribution (median/max V per regime)"
        )
        _save(os.path.join(base_path, filename))
        if show_plots:
            plt.show()


def plot_measured_vs_theory(df: pd.DataFrame, base_path: str, title_prefix: str, show_plots: bool) -> None:
    """Compare empirical runtimes against theoretical complexity (linear + log)."""
    configs = [
        {
            "name": "Bellman–Ford",
            "time_col": "bf_time_s",
            "needs_E": True,
            "complexity": "O(V·E)",
            "filename": "bf_vs_theory",
            "theory": lambda stats: stats["V"] * stats["mean_E"],
        },
        {
            "name": "Floyd–Warshall",
            "time_col": "fw_time_s",
            "needs_E": False,
            "complexity": "O(V³)",
            "filename": "fw_vs_theory",
            "theory": lambda stats: stats["V"] ** 3,
        },
    ]

    for cfg in configs:
        stats = _mean_time_by_v(
            df,
            cfg["time_col"],
            include_E=cfg["needs_E"],
        )
        if stats.empty:
            continue

        if cfg["needs_E"] and "mean_E" not in stats.columns:
            print(f"[warn] Missing edge data for {cfg['name']}; skipping theoretical comparison.", file=sys.stderr)
            continue

        for regime, stats_regime in stats.groupby("regime"):
            stats_regime = stats_regime.sort_values("V")
            if stats_regime.empty:
                continue

            if cfg["needs_E"] and stats_regime["mean_E"].isna().all():
                continue

            theory_raw = cfg["theory"](stats_regime).astype(float)
            stats_regime = stats_regime.assign(theory_raw=theory_raw)
            stats_regime = stats_regime.dropna(subset=["mean_time", "theory_raw", "V"])
            if stats_regime.empty:
                continue

            safe_theory = stats_regime["theory_raw"].replace(0, np.nan)
            ratio = stats_regime["mean_time"] / safe_theory
            scale = np.nanmedian(ratio.replace([np.inf, -np.inf], np.nan))
            if not np.isfinite(scale):
                scale = 1.0
            stats_regime = stats_regime.assign(theory_scaled=stats_regime["theory_raw"] * scale)

            regime_label = _format_regime_label(regime)
            regime_slug = _regime_slug(regime)

            for log_scale, suffix in [(False, "linear"), (True, "log")]:
                if log_scale:
                    mask = (
                        (stats_regime["V"] > 0)
                        & (stats_regime["mean_time"] > 0)
                        & (stats_regime["theory_scaled"] > 0)
                    )
                    stats_plot = stats_regime[mask]
                    if stats_plot.empty:
                        print(
                            f"[warn] No positive-valued data for {cfg['name']} ({regime_label}) log plot; skipping.",
                            file=sys.stderr,
                        )
                        continue
                else:
                    stats_plot = stats_regime

                plt.figure()
                color = _regime_color(regime)
                measured_kwargs = {"label": f"Measured runtime ({regime_label})"}
                theory_kwargs = {"label": f"Scaled {cfg['complexity']} ({regime_label})"}
                if color:
                    measured_kwargs["color"] = color
                    theory_kwargs["color"] = color
                plt.plot(
                    stats_plot["V"],
                    stats_plot["mean_time"],
                    "o-",
                    **measured_kwargs,
                )
                plt.plot(
                    stats_plot["V"],
                    stats_plot["theory_scaled"],
                    "--",
                    **theory_kwargs,
                )
                logx = logy = log_scale
                _prep_axes(logx=logx, logy=logy)
                plt.xlabel(_annotate_label("V", logx))
                plt.ylabel(_annotate_label("Runtime (s)", logy))
                plt.title(
                    f"{title_prefix}: {cfg['name']} vs {cfg['complexity']} — {regime_label} ({suffix})"
                )
                plt.legend()
                filename = f"{cfg['filename']}_{regime_slug}_{suffix}"
                _save(os.path.join(base_path, filename))
                if show_plots:
                    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate BF/FW comparison plots from STP benchmark CSV data.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="stp_benchmarks.csv",
        help="Input CSV file with benchmark results (default: stp_benchmarks.csv)",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="./plots",
        help="Directory to store generated plots (default: ./plots)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively.",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="STP Comparison",
        help='Prefix for plot titles (default: "STP Comparison")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[error] CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    if not _HAVE_MPL:
        print("[error] matplotlib is not available in this environment.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    missing = [c for c in ["V", "bf_time_s", "fw_time_s", "E", "regime"] if c not in df.columns]
    if missing:
        print(f"[info] Missing columns in CSV: {', '.join(missing)} (plots requiring them will be skipped).", file=sys.stderr)

    for col in ("V", "E"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("bf_time_s", "fw_time_s"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "regime" in df.columns:
        df["regime"] = df["regime"].fillna("unknown").astype(str)
    else:
        print("[warn] 'regime' column missing; treating all rows as a single regime.", file=sys.stderr)
        df["regime"] = "all"

    os.makedirs(args.plots_dir, exist_ok=True)
    show = not args.no_show

    plot_time_vs_v(df, args.plots_dir, args.title_prefix, show)
    plot_runtime_distributions(df, args.plots_dir, args.title_prefix, show)
    plot_measured_vs_theory(df, args.plots_dir, args.title_prefix, show)


if __name__ == "__main__":
    main()

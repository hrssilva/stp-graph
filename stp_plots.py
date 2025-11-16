#!/usr/bin/env python3
"""
Plotting utilities and CLI for STP benchmark results.

Typical usage:
  # 1) Run benchmarks and write CSV
  python stp_benchmark.py --csv stp_benchmarks.csv

  # 2) Generate plots from that CSV
  python stp_plots.py --csv stp_benchmarks.csv --plots-dir ./plots
"""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


def plot_results(
    df: pd.DataFrame,
    title_prefix: str = "STP",
    base_path: str = "./plots",
    show_plots: bool = False,
) -> None:
    if not _HAVE_MPL:
        print("[warn] matplotlib not available; skipping plots.", file=sys.stderr)
        return

    # 1) Bellman–Ford runtime vs V*E
    plt.figure()
    plt.scatter(df['VE'], df['bf_time_s'], s=18)
    plt.xlabel("V * E (driver for O(VE))")
    plt.ylabel("Bellman–Ford runtime (s)")
    plt.title(f"{title_prefix}: Bellman–Ford runtime ~ O(VE)")
    plt.tight_layout()
    if show_plots:
        plt.show()
    plt.savefig(os.path.join(base_path, "bf_runtime.png"))

    # 2) Sparse regime: time vs V and vs E
    df_sparse = df[df['regime'] == 'sparse'].groupby('V', as_index=False).agg(
        {'bf_time_s': 'median', 'E': 'median', 'VE': 'median'}
    )
    if not df_sparse.empty:
        plt.figure()
        plt.plot(df_sparse['V'], df_sparse['bf_time_s'], marker='o')
        plt.xlabel("V (sparse, E≈c·V)")
        plt.ylabel("Median BF runtime (s)")
        plt.title(f"{title_prefix}: Sparse → ~O(V^2)")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "bf_sparse.png"))
        if show_plots:
            plt.show()

        plt.figure()
        plt.plot(df_sparse['E'], df_sparse['bf_time_s'], marker='o')
        plt.xlabel("E (sparse)")
        plt.ylabel("Median BF runtime (s)")
        plt.title(f"{title_prefix}: Sparse runtime vs E")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "bf_sparse_vs_e.png"))
        if show_plots:
            plt.show()

    # 3) Dense regime: time vs V
    df_dense = df[df['regime'] == 'dense'].groupby('V', as_index=False).agg(
        {'bf_time_s': 'median', 'E': 'median', 'VE': 'median'}
    )
    if not df_dense.empty:
        plt.figure()
        plt.plot(df_dense['V'], df_dense['bf_time_s'], marker='o')
        plt.xlabel("V (dense, E≈Θ(V^2))")
        plt.ylabel("Median BF runtime (s)")
        plt.title(f"{title_prefix}: Dense → ~O(V^3)")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "bf_dense.png"))
        if show_plots:
            plt.show()

    # 4) BF vs FW where both ran
    df_both = df[~df['fw_time_s'].isna()].groupby(['regime', 'V'], as_index=False).agg(
        {'bf_time_s': 'median', 'fw_time_s': 'median', 'E': 'median'}
    )
    if not df_both.empty:
        for regime in ['sparse', 'dense']:
            sub = df_both[df_both['regime'] == regime]
            if sub.empty:
                continue
            plt.figure()
            plt.plot(sub['V'], sub['bf_time_s'], marker='o', label='Bellman–Ford')
            plt.plot(sub['V'], sub['fw_time_s'], marker='x', label='Floyd–Warshall')
            plt.xlabel(f"V ({regime})")
            plt.ylabel("Median runtime (s)")
            plt.title(f"{title_prefix}: {regime.capitalize()} — BF (O(VE)) vs FW (O(V^3))")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(base_path, f"bf_fw_{regime}.png"))
            if show_plots:
                plt.show()


def plot_theoretical_comparisons(
    df: pd.DataFrame,
    title_prefix: str = "STP",
    base_path: str = "./plots",
    show_plots: bool = False,
) -> None:
    if not _HAVE_MPL:
        print("[warn] matplotlib not available; skipping theoretical plots.", file=sys.stderr)
        return

    # Filter only BF (FW is always O(V^3))
    df_bf = df.copy()
    df_bf = df_bf[df_bf['bf_time_s'].notna()].copy()
    df_bf['VE_theory'] = df_bf['V'] * df_bf['E']
    df_bf['V3_theory'] = df_bf['V'] ** 3

    # Normalize to compare constants
    df_bf['bf_per_VE'] = df_bf['bf_time_s'] / df_bf['VE_theory']
    df_bf['bf_per_V3'] = df_bf['bf_time_s'] / df_bf['V3_theory']

    # ----- 1. Empirical vs. theoretical curves (scaled) -----
    for regime in ['sparse', 'dense']:
        sub = df_bf[df_bf['regime'] == regime].groupby('V', as_index=False).agg(
            {
                'bf_time_s': 'median',
                'E': 'median',
                'VE_theory': 'median',
                'V3_theory': 'median',
            }
        )
        if sub.empty:
            continue

        # Fit constants k1, k2 to overlay theoretical curves
        k_VE = np.median(sub['bf_time_s'] / sub['VE_theory'])
        k_V3 = np.median(sub['bf_time_s'] / sub['V3_theory'])

        plt.figure()
        plt.plot(sub['V'], sub['bf_time_s'], 'o-', label='Empirical BF')
        plt.plot(sub['V'], k_VE * sub['V'] * sub['E'], '--', label=f'k·V·E (k={k_VE:.2e})')
        plt.plot(sub['V'], k_V3 * (sub['V'] ** 3), ':', label=f'k·V³ (k={k_V3:.2e})')
        plt.xlabel("V")
        plt.ylabel("Runtime (s)")
        plt.title(f"{title_prefix}: {regime.capitalize()} — Empirical vs. Theoretical")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"theory_compare_{regime}.png"))
        if show_plots:
            plt.show()
        plt.close()

        plt.figure()
        plt.plot(sub['V'], sub['V'] * sub['E'], '--', label='V·E')
        plt.plot(sub['V'], (sub['V'] ** 3), ':', label='V³')
        plt.plot(sub['V'], (sub['V'] * sub['V']), 'o-', label='V²')
        plt.xlabel("V")
        plt.ylabel("Runtime (s)")
        plt.title(f"{title_prefix}: {regime.capitalize()} — Empirical vs. Theoretical")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"theory_compare_ve_v3_{regime}.png"))
        if show_plots:
            plt.show()
        plt.close()

    # ----- 2. Normalized runtime ratios -----
    plt.figure()
    for regime, style in [('sparse', 'o-'), ('dense', 's--')]:
        sub = df_bf[df_bf['regime'] == regime]
        if sub.empty:
            continue
        plt.plot(sub['V'], sub['bf_per_VE'], style, label=f'{regime}: time/(V·E)')
    plt.xlabel("V")
    plt.ylabel("time / (V·E)")
    plt.title(f"{title_prefix}: Normalized by O(VE)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "bf_per_VE.png"))
    plt.close()

    plt.figure()
    for regime, style in [('sparse', 'o-'), ('dense', 's--')]:
        sub = df_bf[df_bf['regime'] == regime]
        if sub.empty:
            continue
        plt.plot(sub['V'], sub['bf_per_V3'], style, label=f'{regime}: time/V³')
    plt.xlabel("V")
    plt.ylabel("time / V³")
    plt.title(f"{title_prefix}: Normalized by O(V³)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(base_path, "bf_per_V3.png"))
    plt.close()


# -----------------------------
# CLI for plotting module
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate plots for STP benchmarks from a CSV file.",
    )
    p.add_argument(
        '--csv', type=str, default='stp_benchmarks.csv',
        help='Input CSV file with benchmark results (default: stp_benchmarks.csv)',
    )
    p.add_argument(
        '--plots-dir', type=str, default='./plots',
        help='Directory to write plots (default: ./plots)',
    )
    p.add_argument(
        '--no-show', action='store_true',
        help='Do not display plots interactively (only save to files)',
    )
    p.add_argument(
        '--title-prefix', type=str, default='STP Consistency',
        help='Prefix for plot titles (default: "STP Consistency")',
    )
    return p.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.csv):
        print(f"[error] CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    os.makedirs(args.plots_dir, exist_ok=True)

    show = not args.no_show

    plot_results(
        df,
        title_prefix=args.title_prefix,
        base_path=args.plots_dir,
        show_plots=show,
    )
    plot_theoretical_comparisons(
        df,
        title_prefix=args.title_prefix,
        base_path=args.plots_dir,
        show_plots=show,
    )


if __name__ == '__main__':
    main()

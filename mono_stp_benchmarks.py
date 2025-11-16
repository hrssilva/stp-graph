#!/usr/bin/env python3
"""
STP Benchmark: Empirical complexity of Simple Temporal Problem (STP) consistency

Implements:
  - STP instance generation (sparse/dense, consistent or not)
  - Bellman–Ford consistency check (O(VE))
  - Floyd–Warshall consistency check (O(V^3)) for comparison
  - Benchmark harness with CSV output and plots

Usage (defaults are sensible):
  python stp_benchmark.py
  python stp_benchmark.py --repeats 5 --density-sparse 0.05 --density-dense 0.8
  python stp_benchmark.py --sparse-sizes 100 200 400 800 1200 --dense-sizes 40 60 80 100 140
  python stp_benchmark.py --csv stp_results.csv
  python stp_benchmark.py --no-plots
"""

from __future__ import annotations
import argparse
import math
import random
import sys
import os
import time
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False

try:
    import numba as _numba
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

Edge = Tuple[int, int, float]  # (u, v, w) encodes constraint: x_v - x_u <= w as edge u->v weight w


# -----------------------------
# STP generation
# -----------------------------
"""
def gen_stp_instance(
    n_vertices: int,
    *,
    density: float = 0.2,
    weight_range: Tuple[float, float] = (-10.0, 10.0),
    ensure_consistent: bool = True,
    seed: Optional[int] = None,
) -> List[Edge]:
    """""""
    Generate a random STP instance over variables 0..n-1.
    Each constraint x_v - x_u <= w is represented as directed edge u->v with weight w.

    If ensure_consistent=True, we sample a feasible potential p and relax edges so that
    w >= p[v] - p[u], guaranteeing consistency (no negative cycles).
    If ensure_consistent=False, we inject a random negative 3-cycle (if possible).
    """""""
    rng = random.Random(seed)
    edges: List[Edge] = []
    for u in range(n_vertices):
        for v in range(n_vertices):
            if u == v:
                continue
            if rng.random() < density:
                w = rng.uniform(*weight_range)
                edges.append((u, v, w))

    if ensure_consistent:
        # Enforce feasibility using random potentials p
        p = [rng.uniform(-5, 5) for _ in range(n_vertices)]
        new_edges = []
        for u, v, w in edges:
            feasible_w = max(w, p[v] - p[u])
            new_edges.append((u, v, feasible_w))
        edges = new_edges
    else:
        # Try to create a definite negative cycle (forces inconsistency)
        if n_vertices >= 3:
            K = abs(weight_range[0]) + abs(weight_range[1]) + 5.0
            a, b, c = rng.sample(range(n_vertices), 3)
            edges.append((a, b, -K))
            edges.append((b, c, -K))
            edges.append((c, a, -K))
    return edges
"""
def gen_stp_interval_instance(
    n_vertices: int,
    *,
    density: float = 0.2,                    # probability per unordered pair
    lower_range: Tuple[float, float] = (-10.0, 0.0),
    upper_range: Tuple[float, float] = (0.0, 10.0),
    ensure_consistent: bool = True,
    seed: Optional[int] = None,
) -> List[Edge]:
    rng = random.Random(seed)
    edges: List[Edge] = []
    p = []
    if ensure_consistent:
        p = [rng.uniform(-5, 5) for _ in range(n_vertices)]

    for u in range(n_vertices):
        for v in range(u + 1, n_vertices):
            if rng.random() >= density:
                continue

            # sample a raw interval (may be inconsistent/wide/narrow)
            L = rng.uniform(*lower_range)
            U = rng.uniform(*upper_range)
            if L > U:
                L, U = U, L  # ensure L <= U

            if ensure_consistent:
                delta = p[v] - p[u]
                # widen interval so that delta is feasible
                if delta < L:
                    L = delta
                elif delta > U:
                    U = delta

            # add both directed edges for the interval
            edges.append((u, v, U))    # x_v - x_u <= U
            edges.append((v, u, -L))   # x_u - x_v <= -L

    if not ensure_consistent and n_vertices >= 3:
        # force inconsistency via a negative cycle on upper bounds
        # (interval edges are fine: we only need the upper bounds to make a neg cycle)
        K = 5.0 + abs(lower_range[0]) + abs(upper_range[1])
        a, b, c = rng.sample(range(n_vertices), 3)
        edges.append((a, b, -K))
        edges.append((b, c, -K))
        edges.append((c, a, -K))

    return edges

# -----------------------------
# Consistency via shortest paths
# -----------------------------
def bellman_ford_consistency(n: int, edges: List[Edge], strict=True) -> Tuple[bool, Optional[List[float]]]:
    """
    Bellman–Ford based consistency check.
    Initialize dist[v]=0 (equivalent to super-source s->v weight 0).
    After V-1 relaxations, if any edge can still relax -> negative cycle -> inconsistent.
    If strict=True, always runs until V-1 relaxations

    Returns: (is_consistent, distances_if_consistent_else_None)
    Time: O(VE), Space: O(V)
    """
    dist = [0.0] * n  # super-source trick
    for _ in range(n - 1):
        updated = False
        for u, v, w in edges:
            nd = dist[u] + w
            if dist[v] > nd:
                dist[v] = nd
                updated = True
        if not strict and not updated:
            break
    # Detect negative cycle (allow small epsilon due to floats)
    eps = 1e-12
    for u, v, w in edges:
        if dist[v] > dist[u] + w + eps:
            return (False, None)
    return (True, dist)

def _edges_to_arrays(edges: List[Edge]):
    """Convert list[(u,v,w)] -> three contiguous NumPy arrays (int32,int32,float64)."""
    m = len(edges)
    u = np.empty(m, dtype=np.int32)
    v = np.empty(m, dtype=np.int32)
    w = np.empty(m, dtype=np.float64)
    for i, (uu, vv, ww) in enumerate(edges):
        u[i] = uu; v[i] = vv; w[i] = ww
    return u, v, w

if _HAVE_NUMBA:
    @_numba.njit(cache=True, fastmath=False)
    def _bf_numba(n: int, u: np.ndarray, v: np.ndarray, w: np.ndarray, strict: int):
        """
        Numba BF. strict=1 forces exactly V-1 passes; strict=0 allows early stop.
        Returns (ok_flag, dist).
        """
        dist = np.zeros(n, dtype=np.float64)  # super-source trick
        m = w.size
        for it in range(n - 1):
            updated = False
            for i in range(m):
                nd = dist[u[i]] + w[i]
                if nd < dist[v[i]]:
                    dist[v[i]] = nd
                    updated = True
            if (strict == 0) and (not updated):
                break

        # negative cycle check
        eps = 1e-10
        for i in range(m):
            if dist[u[i]] + w[i] + eps < dist[v[i]]:
                return False, dist
        return True, dist

def bellman_ford_consistency_fast(n: int, edges: List[Edge], strict: bool = True):
    """
    Prefer Numba BF if available, else fall back to Python BF.
    NOTE: this wrapper keeps conversion out of the timed region (call it *before* timing if you want).
    """
    if _HAVE_NUMBA:
        u, v, w = _edges_to_arrays(edges)
        ok, dist = _bf_numba(n, u, v, w, 1 if strict else 0)
        return ok, dist
    else:
        return bellman_ford_consistency(n, edges, strict=strict)

def floyd_warshall_consistency(n: int, edges: List[Edge]) -> Tuple[bool, Optional[List[List[float]]]]:
    """
    Floyd–Warshall all-pairs closure for STP (O(V^3), O(V^2) space).
    Inconsistency if any diagonal d[i][i] < 0 after closure.
 
    Robust Floyd–Warshall for STP consistency.
    Uses math.inf, guards unreachable pairs, and a scale-aware tolerance.
    """
    INF = math.inf
    d = [[0.0 if i == j else INF for j in range(n)] for i in range(n)]
    #d = np.full((n, n), INF, dtype=float)
    #np.fill_diagonal(d, 0.0)
    for u, v, w in edges:
        if w < d[u][v]:
            d[u][v] = w

    for k in range(n):
        dk = d[k]                 # row k (may be updated in-place; that's fine)
        for i in range(n):
            dik = d[i][k]
            if not math.isfinite(dik):     # skip unreachable i→k
                continue
            di = d[i]
            for j in range(n):
                dj = dk[j]
                if not math.isfinite(dj):  # skip unreachable k→j
                    continue
                alt = dik + dj             # both finite
                if alt < di[j]:
                    di[j] = alt

    # With proper INF handling, diagonals should be ≥ 0 for a consistent STP.
    # Allow a tiny tolerance for FP noise only (order 1e-10…1e-8).
    eps = 1e-10
    for i in range(n):
        if d[i][i] < -eps:
            return (False, None)
    return (True, d)

def floyd_warshall_consistency_numpy(n: int, edges: List[Edge]) -> Tuple[bool, Optional[np.ndarray]]:
    """
    NumPy-broadcasted Floyd–Warshall:
      d = min(d, d[:,k,None] + d[None,k,:]) for k in 0..n-1
    Much faster than pure Python loops for moderate n.
    """
    INF = np.inf
    d = np.full((n, n), INF, dtype=float)
    np.fill_diagonal(d, 0.0)
    # set edges
    for u, v, w in edges:
        if w < d[u, v]:
            d[u, v] = w
    # closure
    for k in range(n):
        # alt distances through k
        alt = d[:, k][:, None] + d[k, :][None, :]
        np.minimum(d, alt, out=d)
    # detect negative cycles with small tolerance
    if np.any(np.diag(d) < -1e-10):
        return (False, None)
    return (True, d)

# -----------------------------
# Benchmark harness
# -----------------------------

def _bench_job(regime: str,
               n: int,
               edges: list,
               max_fw_n: int,
               seed_base: int,
               job_seq: int) -> dict:
    

    E = len(edges)

    # ---- BF (use numba if present) ----
    t0 = time.perf_counter()
    ok_bf, _ = bellman_ford_consistency(n, edges, strict=True)
    t1 = time.perf_counter()

    # ---- FW (NumPy path if possible) ----
    fw_time = math.nan
    ok_fw = None
    if n <= max_fw_n:
        t2 = time.perf_counter()
        if np is not None:
            ok_fw, _ = floyd_warshall_consistency(n, edges)
        else:
            ok_fw, _ = floyd_warshall_consistency(n, edges)
        t3 = time.perf_counter()
        fw_time = t3 - t2

    return {
        'regime': regime,
        'V': n,
        'E': E,
        'VE': n * E,
        'bf_time_s': t1 - t0,
        'fw_time_s': fw_time,
        'bf_ok': ok_bf,
        'fw_ok': ok_fw
    }


# def benchmark_suite(
#     v_sizes_sparse: List[int],
#     v_sizes_dense: List[int],
#     *,
#     density_sparse: float = 0.05,
#     density_dense: float = 0.8,
#     repeats: int = 3,
#     seed: int = 42,
#     max_fw_n: int = 350,
# ) -> pd.DataFrame:
#     """
#     For each V in sparse/dense regimes, generate 'repeats' random consistent STPs,
#     measure Bellman–Ford and (optionally) Floyd–Warshall runtimes, and record VE.
#     """
#     rng = random.Random(seed)
#     rows = []
#     for regime, sizes, density in [('sparse', v_sizes_sparse, density_sparse),
#                                    ('dense', v_sizes_dense, density_dense)]:
#         for n in sizes:
#             edges = gen_stp_interval_instance(n, density=density, ensure_consistent=True,
#                                          seed=rng.randint(0, 10**9))
#             E = len(edges)
#             for _ in range(repeats):

#                 # Bellman–Ford
#                 t0 = time.perf_counter()
#                 ok_bf, _ = bellman_ford_consistency(n, edges)
#                 t1 = time.perf_counter()

#                 # Floyd–Warshall (for comparison; limited n)
#                 fw_time = math.nan
#                 ok_fw = None
#                 if n <= max_fw_n:
#                     t2 = time.perf_counter()
#                     ok_fw, _ = floyd_warshall_consistency(n, edges)
#                     t3 = time.perf_counter()
#                     fw_time = t3 - t2

#                 rows.append({
#                     'regime': regime,
#                     'V': n,
#                     'E': E,
#                     'VE': n * E,
#                     'bf_time_s': t1 - t0,
#                     'fw_time_s': fw_time,
#                     'bf_ok': ok_bf,
#                     'fw_ok': ok_fw
#                 })
#     return pd.DataFrame(rows)

def benchmark_suite(
    v_sizes_sparse: List[int],
    v_sizes_dense: List[int],
    *,
    density_sparse: float = 0.05,
    density_dense: float = 0.8,
    repeats: int = 3,
    seed: int = 42,
    max_fw_n: int = 350,
    max_workers: Optional[int] = None,   # NEW
) -> pd.DataFrame:
    """
    Fully parallel: each (regime, V, repetition) is an independent process job.
    Reproducible via seed; each job generates its own instance.
    """
    jobs = []
    for regime, sizes, density in [('sparse', v_sizes_sparse, density_sparse),
                                   ('dense',  v_sizes_dense,  density_dense)]:
        for n in sizes:
            rng = random.Random((hash(regime) & 0xffffffff) ^ seed ^ (n * 0x9e3779b1))
            edge_seed = rng.randint(0, 10**9)
            edges = gen_stp_interval_instance(n, density=density, ensure_consistent=True, seed=edge_seed)
            for r in range(repeats):
                # job_seq only used to vary the per-job RNG seed deterministically
                jobs.append((regime, n, edges, max_fw_n, seed, r + 1))

    # Sequential fallback if requested
    if max_workers == 1:
        rows = [_bench_job(*args) for args in jobs]
        return pd.DataFrame(rows)

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_bench_job, *args) for args in jobs]
        for fut in as_completed(futures):
            rows.append(fut.result())
    return pd.DataFrame(rows)

def plot_results(df: pd.DataFrame, title_prefix: str = "STP", base_path="./plots", show_plots=False) -> None:
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
    if show_plots: plt.show()
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
        if show_plots: plt.show()

        plt.figure()
        plt.plot(df_sparse['E'], df_sparse['bf_time_s'], marker='o')
        plt.xlabel("E (sparse)")
        plt.ylabel("Median BF runtime (s)")
        plt.title(f"{title_prefix}: Sparse runtime vs E")
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, "bf_sparse_vs_e.png"))
        if show_plots: plt.show()

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
        if show_plots: plt.show()

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
            if show_plots: plt.show()

def plot_theoretical_comparisons(df: pd.DataFrame, title_prefix: str = "STP", base_path="./plots", show_plots = False) -> None:
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
                {'bf_time_s': 'median', 'E': 'median', 'VE_theory': 'median', 'V3_theory': 'median'}
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
        if show_plots: plt.show()
        plt.close()

        plt.figure()
        plt.plot(sub['V'], sub['V'] * sub['E'], '--', label=f'V·E')
        plt.plot(sub['V'], (sub['V'] ** 3), ':', label=f'V³')
        plt.plot(sub['V'], (sub['V'] * sub['V']), 'o-', label=f'V²')
        plt.xlabel("V")
        plt.ylabel("Runtime (s)")
        plt.title(f"{title_prefix}: {regime.capitalize()} — Empirical vs. Theoretical")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, f"theory_compare_ve_v3_{regime}.png"))
        if show_plots: plt.show()
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
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark STP consistency algorithms (Bellman–Ford vs Floyd–Warshall).")
    p.add_argument('--sparse-sizes', type=int, nargs='+', default=[40, 60, 80, 100, 140, 180, 220, 260, 300, 340],
                   help='V sizes for sparse regime (default: 40 60 80 100 140 180 220 260 300 340)')
    p.add_argument('--dense-sizes', type=int, nargs='+', default=[40, 60, 80, 100, 140, 180, 220, 260, 300, 340],
                   help='V sizes for dense regime (default: 40 60 80 100 140 180 220 260 300 340)')
    p.add_argument('--density-sparse', type=float, default=0.05, help='Edge density for sparse regime (default 0.05)')
    p.add_argument('--density-dense', type=float, default=0.8, help='Edge density for dense regime (default 0.8)')
    p.add_argument('--repeats', type=int, default=3, help='Repeats per (regime,V) point (default 3)')
    p.add_argument('--max-fw-n', type=int, default=350, help='Max V for running Floyd–Warshall (default 350)')
    p.add_argument('--seed', type=int, default=42, help='RNG seed (default 42)')
    p.add_argument('--csv', type=str, default='stp_benchmarks.csv', help='Output CSV path (default stp_benchmarks.csv)')
    p.add_argument('--no-plots', action='store_true', help='Skip plotting')
    p.add_argument('--no-show', action='store_true', help='Skip displaying plots')
    p.add_argument('--workers', type=int, default=None, help='Max parallel processes (default: all cores)')
    p.add_argument('--plots-dir', type=str, default='./plots', help='Directory to write plots (default: ./plots)')
    return p.parse_args()


def main():
    args = parse_args()

    df = benchmark_suite(
        args.sparse_sizes, args.dense_sizes,
        density_sparse=args.density_sparse,
        density_dense=args.density_dense,
        repeats=args.repeats,
        seed=args.seed,
        max_fw_n=args.max_fw_n,
        max_workers=args.workers
    )

    # Write CSV
    df.to_csv(args.csv, index=False)
    print(f"[ok] wrote results to {args.csv} with {len(df)} rows")

    # Print quick summary
    for regime in ['sparse', 'dense']:
        sub = df[df['regime'] == regime]
        if sub.empty:
            continue
        Vuniq = sorted(sub['V'].unique().tolist())
        print(f"Regime={regime} V={Vuniq} — median BF time (s) by V:")
        med = sub.groupby('V')['bf_time_s'].median()
        for v in Vuniq:
            print(f"  V={v:5d}  BF median: {med.loc[v]:.6f}s")

    # Plots
    if not args.no_plots:
        # Make sure plot directory exists
        os.makedirs(args.plots_dir, exist_ok=True)
        plot_results(df, title_prefix="STP Consistency", base_path=args.plots_dir, show_plots= not args.no_show)
        plot_theoretical_comparisons(df, title_prefix="STP Consistency", base_path=args.plots_dir, show_plots= not args.no_show)


if __name__ == '__main__':
    main()


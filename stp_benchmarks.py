#!/usr/bin/env python3
"""
STP Benchmark: Empirical complexity of Simple Temporal Problem (STP) consistency

Implements:
  - STP instance generation (sparse/dense, consistent or not)
  - Bellman–Ford consistency check (O(VE))
  - Floyd–Warshall consistency check (O(V^3)) for comparison
  - Benchmark harness with CSV output

Usage (defaults are sensible):
  python stp_benchmark.py
  python stp_benchmark.py --repeats 5 --density-sparse 0.05 --density-dense 0.8
  python stp_benchmark.py --sparse-sizes 100 200 400 800 1200 --dense-sizes 40 60 80 100 140
  python stp_benchmark.py --csv stp_results.csv
"""

from __future__ import annotations
import argparse
import math
import random
import sys
import os
import time
import psutils
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed


# Single-thread some BLAS-y libs by default
#os.environ.setdefault("OMP_NUM_THREADS", "1")
#os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
#os.environ.setdefault("MKL_NUM_THREADS", "1")
#os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
#os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

Edge = Tuple[int, int, float]  # (u, v, w) encodes constraint: x_v - x_u <= w as edge u->v weight w


# -----------------------------
# STP generation
# -----------------------------
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
        u[i] = uu
        v[i] = vv
        w[i] = ww
    return u, v, w


def bellman_ford_consistency_numpy(
    n: int,
    edges,
    strict: bool = True,
) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Bellman–Ford consistency check using NumPy for the relaxation steps.

    Returns:
        (ok, dist) where:
          - ok: True if no negative cycle detected, else False
          - dist: np.ndarray of distances if ok, otherwise None
    """
    u, v, w = edges

    if len(u) == 0:
        return True, np.zeros(n, dtype=np.float64)

    dist = np.zeros(n, dtype=np.float64)  # super-source trick
    eps = 1e-12

    # V-1 relaxation rounds
    for _ in range(n - 1):
        # candidate distances via each edge
        nd = dist[u] + w  # shape (m,)

        # Find edges that would improve their target
        improved = nd + eps < dist[v]
        if not improved.any():
            if not strict:
                break
            # if strict, we still have to finish all iterations; but no updates is fine
            continue

        # We need to take the min over all incoming edges to each vertex.
        # np.minimum.at handles repeated indices in v[improved].
        # Gauss–Seidel in-place relaxation
        np.minimum.at(dist, v[improved], nd[improved])

    # Negative cycle detection: if any edge can still relax, it's inconsistent.
    if np.any(dist[u] + w + eps < dist[v]):
        return False, None

    return True, dist

def floyd_warshall_consistency(n: int, edges: List[Edge]) -> Tuple[bool, Optional[List[List[float]]]]:
    """
    Floyd–Warshall all-pairs closure for STP (O(V^3), O(V^2) space).
    Inconsistency if any diagonal d[i][i] < 0 after closure.
    """
    INF = math.inf
    d = [[0.0 if i == j else INF for j in range(n)] for i in range(n)]
    for u, v, w in edges:
        if w < d[u][v]:
            d[u][v] = w

    for k in range(n):
        dk = d[k]
        for i in range(n):
            dik = d[i][k]
            if not math.isfinite(dik):  # skip unreachable i→k
                continue
            di = d[i]
            for j in range(n):
                dj = dk[j]
                if not math.isfinite(dj):  # skip unreachable k→j
                    continue
                alt = dik + dj
                if alt < di[j]:
                    di[j] = alt

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
        alt = d[:, k][:, None] + d[k, :][None, :]
        np.minimum(d, alt, out=d)
    if np.any(np.diag(d) < -1e-10):
        return (False, None)
    return (True, d)


# -----------------------------
# Benchmark harness
# -----------------------------
def _bench_job(
    regime: str,
    n: int,
    edges: list,
    max_fw_n: int,
    seed_base: int,
    job_seq: int,
    use_numpy: bool
) -> dict:
    E = len(edges)
    proc = psutil.Process(os.getpid())

    
    # ---- BF (NumPy path if possible)  ----
    bf_edges = _edges_to_arrays(edges) if use_numpy else edges
    t0 = time.perf_counter()
    ok_bf, _ = bellman_ford_consistency_numpy(n, bf_edges, strict=True) if use_numpy else bellman_ford_consistency(n, bf_edges, strict=True)
    t1 = time.perf_counter()

    # ---- FW (NumPy path if possible) ----
    fw_time = math.nan
    ok_fw = None
    fw_alg = floyd_warshall_consistency_numpy if use_numpy else floyd_warshall_consistency
    if n <= max_fw_n:
        t2 = time.perf_counter()
        ok_fw, _ = fw_alg(n, edges)
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
        'fw_ok': ok_fw,
    }


def benchmark_suite(
    v_sizes_sparse: List[int],
    v_sizes_dense: List[int],
    *,
    density_sparse: float = 0.05,
    density_dense: float = 0.8,
    repeats: int = 3,
    seed: int = 42,
    max_fw_n: int = 350,
    max_workers: Optional[int] = None,
    use_numpy: bool = False
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
            edges = gen_stp_interval_instance(
                n, density=density, ensure_consistent=True, seed=edge_seed
            )
            for r in range(repeats):
                jobs.append((regime, n, edges, max_fw_n, seed, r + 1, use_numpy))

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


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark STP consistency algorithms (Bellman–Ford vs Floyd–Warshall).",
    )
    p.add_argument(
        '--sparse-sizes', type=int, nargs='+',
        default=[40, 60, 80, 100, 140, 180, 220, 260, 300, 340],
        help='V sizes for sparse regime (default: 40 60 80 100 140 180 220 260 300 340)',
    )
    p.add_argument(
        '--dense-sizes', type=int, nargs='+',
        default=[40, 60, 80, 100, 140, 180, 220, 260, 300, 340],
        help='V sizes for dense regime (default: 40 60 80 100 140 180 220 260 300 340)',
    )
    p.add_argument(
        '--density-sparse', type=float, default=0.05,
        help='Edge density for sparse regime (default 0.05)',
    )
    p.add_argument(
        '--density-dense', type=float, default=0.8,
        help='Edge density for dense regime (default 0.8)',
    )
    p.add_argument(
        '--repeats', type=int, default=3,
        help='Repeats per (regime,V) point (default 3)',
    )
    p.add_argument(
        '--max-fw-n', type=int, default=350,
        help='Max V for running Floyd–Warshall (default 350)',
    )
    p.add_argument('--seed', type=int, default=42, help='RNG seed (default 42)')
    p.add_argument(
        '--csv', type=str, default='stp_benchmarks.csv',
        help='Output CSV path (default stp_benchmarks.csv)',
    )
    p.add_argument(
        '--workers', type=int, default=None,
        help='Max parallel processes (default: all cores)',
    )
    p.add_argument(
    '--use-numpy', action='store_true',
    help='Use NumPy implementations of algorithms instead of pure Python.'
)

    return p.parse_args()


def main():
    args = parse_args()

    df = benchmark_suite(
        args.sparse_sizes,
        args.dense_sizes,
        density_sparse=args.density_sparse,
        density_dense=args.density_dense,
        repeats=args.repeats,
        seed=args.seed,
        max_fw_n=args.max_fw_n,
        max_workers=args.workers,
        use_numpy=args.use_numpy
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


if __name__ == '__main__':
    main()

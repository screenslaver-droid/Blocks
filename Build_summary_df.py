"""
build_n_summary.py  —  Joint ISV + LFS-Stiffness N* Recommendation for DRIFT
==============================================================================

Standalone post-processing script that reads all <id>_sweep.csv outputs from
Diagnostic_N_sweep_optimized_v3.py and produces per-class N* recommendations
using a physically grounded two-criterion optimisation.

WHY ISV ALONE IS INSUFFICIENT
──────────────────────────────
ISV (intra-superpixel variance) measures SLIC segmentation convergence only.
It tells you when the spatial partitioning stops improving — but not whether
the resulting nodes are suitable for efficient ODE integration.  For a GAT-ODE
solving the ADR equation, the graph topology also determines solver stiffness.

TWO STIFFNESS SOURCES IN THE DRIFT ADR EQUATION
──────────────────────────────────────────────────
    dh_i/dt = [Σ_j α_ij W_flux (h_j − h_i)]_diffusion
             + [s_i · MLP_src(h_i, e_i) − (1−s_i) · MLP_snk(h_i, e_i)]_reaction

    DIFFUSION STIFFNESS  (E_super / λ_max proxy)
    ────────────────────────────────────────────
    The diffusion term is discretised as D·L·h where L is the graph Laplacian.
    λ_max(L) ∝ max node degree ∝ E_super / N.  E_super grows near-quadratically
    with N, so diffusion stiffness scales as N².  For explicit / adaptive-step
    solvers (dopri5), the maximum stable step size Δt ≤ 2/(D·λ_max) shrinks
    accordingly.  E_super is also the literal GPU VRAM cost of the pre-allocated
    edge tensor — a hard feasibility ceiling.
    [Derived from: ADR_equation_modelling.pdf, slide 31 — 'E_super is a
     feasibility ceiling, not a quality metric.']

    REACTION STIFFNESS  (LFS proxy)
    ────────────────────────────────
    At higher N, each superpixel is smaller and tracks a more reactive,
    rapidly-changing physical parcel.  LFS (Lagrangian Feature Smoothness)
    measures the mean per-step feature change along each tracked node
    trajectory — i.e., it directly measures how large dh/dt is along the
    Lagrangian path.  High LFS → the reaction term R(h, e) produces rapid
    state changes → stiffer ODE system on the physics side.
    [Defined in: ADR_equation_modelling.pdf, slide 29 —
     'Low LFS means nodes track parcels with slowly varying physical
      properties — the ideal for Lagrangian graph-ODE integration.']

JOINT SCORING FRAMEWORK
─────────────────────────
For each N:
    quality(N)   = (ISV₀ − ISVₙ) / (ISV₀ − ISV_last + ε)    ∈ [0, 1]
                   Normalised ISV improvement; 0 at N_first, 1 at N_last.
    stiffness(N) = (LFSₙ − LFS₀) / (LFS_last − LFS₀ + ε)    ∈ [0, 1]
                   Normalised LFS increase; 0 at N_first, 1 at N_last.
    joint(N)     = quality(N) − α · stiffness(N)

N_joint = argmax_N joint(N)  — unconstrained joint optimum.

LFS CEILING (reaction-stiffness gate)
    lfs_ceiling = largest N where LFS(N) ≤ LFS_min × (1 + tol).
    Default tol = 10 %.  If LFS is entirely flat, ceiling = N_last.

VRAM / DIFFUSION-STIFFNESS CEILING (hard feasibility gate)
    vram_ceiling = largest N where VRAM(N, batch) ≤ vram_budget_mb.
    E_super(N) = N · min(π·R_max²/A_sp, N−1)  where R_max = v_max · T_fc.

FINAL PER-CLASS RECOMMENDATION
    voted_N       = plurality winner of {ISV_elbow, FAE_elbow, MDV_elbow, joint_N}
    recommended_N = min(voted_N, lfs_ceiling, vram_ceiling)
    snapped to nearest valid N in the sweep grid.

ALPHA GUIDANCE
    α = 1.0  Equal weight to quality and stiffness.
    α = 1.5  Stiffness-conservative (DEFAULT); appropriate for dopri5 on
             storm events with dominant reaction terms (GROWTH_DECAY,
             RAPID_GROWTH, EPISODIC).
    α = 2.0  Strongly stiffness-averse; use if ODE solver fails to converge
             at the α = 1.5 recommendation.
    α = 0.5  Quality-biased; use only if GPU VRAM budget is large and
             the ODE solver is an implicit method (e.g., Radau, SDIRK).

Usage
─────
    # Reads all *_sweep.csv from the default RESULTS_DIR:
    python build_n_summary.py

    # Custom results dir and VRAM budget:
    python build_n_summary.py \\
        --results_dir path/to/sweep \\
        --vram_budget_mb 4000 \\
        --batch_size 32 \\
        --alpha 1.5 \\
        --lfs_tol 0.10

    # Override N grid (must match what was used in the sweep):
    python build_n_summary.py --n_values 250 500 750 1000 1500 2000

    # Re-use precomputed aggregates (skip CSV reload):
    python build_n_summary.py --agg_csv sweep_aggregate.csv \\
                               --class_csv sweep_by_class.csv

Output
──────
    <RESULTS_DIR>/sweep_aggregate.csv         — mean ± CI per N  (all events)
    <RESULTS_DIR>/sweep_by_class.csv          — mean ± CI per N per class
    <RESULTS_DIR>/sweep_summary_v2.txt        — full stiffness-aware report
    <RESULTS_DIR>/class_N_recommendations.csv — per-class N* with full reasoning
"""

import argparse
import glob
import logging
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# --------------------------------------------------------------------------- #
# LOGGING
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# DEFAULTS  (mirror Diagnostic_N_sweep_optimized_v3.py)
# --------------------------------------------------------------------------- #
RESULTS_DIR        = r"C:\Users\Siddharth Nair\results\polished"
DEFAULT_N_STEP   = 50
DEFAULT_N_MIN    = 50
DEFAULT_N_MAX    = 2000
DEFAULT_N_VALUES   = list(range(DEFAULT_N_MIN, DEFAULT_N_MAX + 1, DEFAULT_N_STEP))

# Physical constants for E_super estimation
V_MAX_PX_PER_FRAME = 9
T_FORECAST_FRAMES  = 12
HW                 = 384          # SEVIR grid size (384 × 384)

# Scoring / feasibility defaults
DEFAULT_ALPHA          = 1.5     # stiffness penalty weight
DEFAULT_LFS_TOL        = 0.10    # 10 % rise tolerance for LFS ceiling
DEFAULT_VRAM_BUDGET_MB = 4000    # conservative single-GPU budget (MB)
DEFAULT_BATCH_SIZE     = 32

# Confidence interval
CI_ALPHA = 0.05   # 95 % CI via t-distribution

# Metric columns expected in per-event sweep CSVs
_METRIC_COLS = [
    "intra_sp_var_mean", "intra_sp_var_std",
    "FAE", "BDV", "GED", "MDV", "LFS",
    "boundary_recall_mean", "temporal_br_std",
    "E_super_est", "VRAM_est_MB",
    "runtime_s", "actual_N",
]


# =========================================================================== #
# CSV LOADER
# =========================================================================== #
def load_sweep_csvs(results_dir: str,
                    required_n_values: Optional[list] = None) -> pd.DataFrame:
    """
    Scan results_dir for all *_sweep.csv files and concatenate them.
    Optionally filter to events that have all required N values with
    valid ISV entries (guards against partial / stub rows).
    """
    pattern = os.path.join(results_dir, "*_sweep.csv")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No *_sweep.csv files found in: {results_dir}")

    dfs = []
    skipped = 0
    for p in paths:
        try:
            df = pd.read_csv(p)
            if "intra_sp_var_mean" not in df.columns:
                skipped += 1
                continue
            if required_n_values is not None:
                completed = set(df["N"].dropna().astype(int).unique())
                valid     = df["intra_sp_var_mean"].notna().sum()
                if completed != set(required_n_values) or valid < len(required_n_values):
                    log.warning(
                        f"  Incomplete: {os.path.basename(p)} "
                        f"({valid}/{len(required_n_values)} valid rows) — skipping.")
                    skipped += 1
                    continue
            dfs.append(df)
        except Exception as exc:
            log.warning(f"  Could not read {p}: {exc}")
            skipped += 1

    if not dfs:
        raise ValueError("No valid sweep CSVs could be loaded.")

    all_df = pd.concat(dfs, ignore_index=True)
    log.info(
        f"  Loaded {len(dfs)} events ({skipped} skipped) — "
        f"{len(all_df)} rows total.")
    return all_df


# =========================================================================== #
# E_SUPER / VRAM HELPERS
# =========================================================================== #
def estimate_e_super(n: int, H: int = HW, W: int = HW,
                     v_max: float = V_MAX_PX_PER_FRAME,
                     t_fc: int = T_FORECAST_FRAMES) -> int:
    """Pre-allocated edge count for the GAT-ODE graph at N superpixels."""
    R_max         = v_max * t_fc
    avg_sp_area   = (H * W) / max(n, 1)
    avg_neighbors = np.pi * R_max ** 2 / avg_sp_area
    return int(n * min(avg_neighbors, n - 1))


def estimate_vram_mb(n: int, batch_size: int = DEFAULT_BATCH_SIZE,
                     floats_per_edge: int = 6) -> float:
    """VRAM cost of the edge tensor: |E_super| × batch × 6 floats × 4 B."""
    return estimate_e_super(n) * batch_size * floats_per_edge * 4 / 1e6


def vram_feasible_ceiling(n_values: list, batch_size: int,
                          budget_mb: float) -> int:
    """Largest N whose edge-tensor VRAM fits within budget_mb."""
    feasible = [n for n in n_values
                if estimate_vram_mb(n, batch_size=batch_size) <= budget_mb]
    if not feasible:
        log.warning(
            f"  No N in {n_values} fits within {budget_mb} MB — "
            f"returning smallest N.")
        return n_values[0]
    return max(feasible)


# =========================================================================== #
# ELBOW DETECTION
# =========================================================================== #
def find_elbow(n_values: list, scores: list) -> int:
    """
    Normalised perpendicular-distance elbow (Kneedle core).
    Finds N where a monotone-decreasing curve has maximum curvature.
    Suitable for: ISV, FAE, MDV, TBR, (1 − BR).
    NOT suitable for LFS (which increases with N) — use find_lfs_ceiling.
    """
    if len(n_values) < 3:
        return n_values[0]
    pts   = np.array(list(zip(n_values, scores)), dtype=float)
    rng   = pts.max(axis=0) - pts.min(axis=0) + 1e-12
    pts_n = (pts - pts.min(axis=0)) / rng
    line  = pts_n[-1] - pts_n[0]
    line /= np.linalg.norm(line) + 1e-12
    dists = [
        np.linalg.norm((p - pts_n[0]) - np.dot(p - pts_n[0], line) * line)
        for p in pts_n
    ]
    return int(n_values[int(np.argmax(dists))])


def find_lfs_ceiling(n_values: list, lfs_scores: list,
                     rise_tol: float = DEFAULT_LFS_TOL) -> int:
    """
    LFS STIFFNESS CEILING — the largest N where the ODE reaction-term
    stiffness has not yet degraded significantly above its minimum.

    Standard find_elbow() is inapplicable to LFS because LFS is
    monotone-INCREASING with N (or flat). Applying find_elbow() to an
    increasing curve always returns N_first — useless as a ceiling.

    Instead: find the largest N where
        LFS(N) ≤ LFS_min × (1 + rise_tol) + rise_tol × (LFS_max − LFS_min)

    The compound threshold allows the ceiling to be non-trivial even when
    LFS_min ≈ 0.  Default rise_tol = 10 % means: accept at most a 10 %
    degradation in Lagrangian feature smoothness above the best observed
    value.

    If LFS is entirely flat across the sweep (LFS_max ≈ LFS_min), there
    is no stiffness concern and the ceiling is set to N_last.
    """
    arr  = np.asarray(lfs_scores, dtype=float)
    lmin = float(arr.min())
    lmax = float(arr.max())
    if (lmax - lmin) < 1e-9:
        return int(n_values[-1])   # flat LFS — no stiffness concern
    threshold  = lmin * (1.0 + rise_tol) + rise_tol * (lmax - lmin)
    ceiling_idx = 0
    for i, v in enumerate(arr):
        if v <= threshold:
            ceiling_idx = i
    return int(n_values[ceiling_idx])


# =========================================================================== #
# JOINT SCORING
# =========================================================================== #
def compute_joint_scores(n_values: list, isv_scores: list,
                         lfs_scores: list,
                         alpha: float = DEFAULT_ALPHA,
                         ) -> tuple[list[float], int]:
    """
    Joint quality-stiffness score:
        quality(N)   = (ISV₀ − ISVₙ)   / (ISV₀   − ISV_last  + ε)  ∈ [0,1]
        stiffness(N) = (LFSₙ − LFS₀)   / (LFS_last − LFS₀    + ε)  ∈ [0,1]
        joint(N)     = quality(N) − α · stiffness(N)

    Returns (joint_scores_list, joint_optimal_N).

    Interpretation:
    · quality measures how much of the maximum possible ISV improvement
      has been captured at each N.
    · stiffness measures how far LFS has risen above its minimum (N_first).
    · α = 1.5 means each unit of stiffness increase is penalised 1.5×
      relative to each unit of quality gain — stiffness-conservative,
      appropriate for dopri5 on convective events.
    """
    isv = np.asarray(isv_scores, dtype=float)
    lfs = np.asarray(lfs_scores, dtype=float)

    quality   = (isv[0] - isv)   / (isv[0] - isv[-1]   + 1e-12)
    stiffness = (lfs    - lfs[0]) / (lfs[-1] - lfs[0]   + 1e-12)

    joint    = quality - alpha * stiffness
    best_idx = int(np.argmax(joint))
    return list(map(float, joint)), int(n_values[best_idx])


# =========================================================================== #
# CONFIDENCE INTERVALS
# =========================================================================== #
def _ci_half(values: pd.Series, ci_alpha: float = CI_ALPHA) -> float:
    """Half-width of the t-distribution CI for a sample."""
    n = values.dropna().count()
    if n < 2:
        return float("nan")
    se = float(values.dropna().std(ddof=1)) / np.sqrt(n)
    t  = float(sp_stats.t.ppf(1 - ci_alpha / 2, df=n - 1))
    return t * se


# =========================================================================== #
# AGGREGATE STATISTICS
# =========================================================================== #
def _ci_col(grp: pd.core.groupby.GroupBy, col: str,
            ci_alpha: float = CI_ALPHA) -> pd.Series:
    """Per-group 95 % CI half-width for column `col`."""
    return grp[col].apply(lambda s: _ci_half(s, ci_alpha))


def aggregate_all(all_df: pd.DataFrame) -> pd.DataFrame:
    """Per-N mean ± 95 % CI across all events and lifecycle classes."""
    grp = all_df.groupby("N")

    agg = grp.agg(
        ISV_mean        =("intra_sp_var_mean",    "mean"),
        ISV_std         =("intra_sp_var_mean",    "std"),
        ISV_ci          =("intra_sp_var_mean",    lambda s: _ci_half(s)),
        FAE_mean        =("FAE",                  "mean"),
        FAE_ci          =("FAE",                  lambda s: _ci_half(s)),
        BDV_mean        =("BDV",                  "mean"),
        MDV_mean        =("MDV",                  "mean"),
        MDV_ci          =("MDV",                  lambda s: _ci_half(s)),
        LFS_mean        =("LFS",                  "mean"),
        LFS_std         =("LFS",                  "std"),
        LFS_ci          =("LFS",                  lambda s: _ci_half(s)),
        BR_mean         =("boundary_recall_mean", "mean"),
        BR_ci           =("boundary_recall_mean", lambda s: _ci_half(s)),
        TBR_mean        =("temporal_br_std",      "mean"),
        TBR_ci          =("temporal_br_std",      lambda s: _ci_half(s)),
        E_super_mean    =("E_super_est",          "mean"),
        VRAM_mean_MB    =("VRAM_est_MB",          "mean"),
        runtime_mean    =("runtime_s",            "mean"),
        n_events        =("event_id",             "count"),
    ).reset_index()

    # Relative CI width = (2 × CI_half) / mean — the diagnostic from plot #1
    agg["ISV_rel_ci_pct"] = (2 * agg["ISV_ci"]) / (agg["ISV_mean"] + 1e-12) * 100
    agg["LFS_rel_ci_pct"] = (2 * agg["LFS_ci"]) / (agg["LFS_mean"] + 1e-12) * 100
    return agg


def aggregate_by_class(all_df: pd.DataFrame) -> pd.DataFrame:
    """Per-(lifecycle_class, N) mean ± 95 % CI."""
    grp = all_df.groupby(["lifecycle_class", "N"])
    agg = grp.agg(
        ISV_mean        =("intra_sp_var_mean",    "mean"),
        ISV_std         =("intra_sp_var_mean",    "std"),
        ISV_ci          =("intra_sp_var_mean",    lambda s: _ci_half(s)),
        LFS_mean        =("LFS",                  "mean"),
        LFS_std         =("LFS",                  "std"),
        LFS_ci          =("LFS",                  lambda s: _ci_half(s)),
        FAE_mean        =("FAE",                  "mean"),
        MDV_mean        =("MDV",                  "mean"),
        BR_mean         =("boundary_recall_mean", "mean"),
        TBR_mean        =("temporal_br_std",      "mean"),
        E_super_mean    =("E_super_est",          "mean"),
        VRAM_mean_MB    =("VRAM_est_MB",          "mean"),
        n_events        =("event_id",             "count"),
    ).reset_index()

    agg["ISV_rel_ci_pct"] = (2 * agg["ISV_ci"]) / (agg["ISV_mean"] + 1e-12) * 100
    agg["LFS_rel_ci_pct"] = (2 * agg["LFS_ci"]) / (agg["LFS_mean"] + 1e-12) * 100
    return agg


# =========================================================================== #
# BINDING CONSTRAINT CLASSIFIER
# =========================================================================== #
def _binding_constraint(voted_n: int, joint_n: int,
                        lfs_ceiling: int, vram_ceiling: int) -> str:
    """
    Identify which constraint (if any) actually lowered the recommended N
    below the unconstrained majority vote.
    """
    if voted_n <= min(lfs_ceiling, vram_ceiling):
        return "none (quality-dominated)"
    constraints = []
    if voted_n > lfs_ceiling:
        constraints.append("LFS-stiffness")
    if voted_n > vram_ceiling:
        constraints.append("VRAM")
    return " + ".join(constraints)


# =========================================================================== #
# PER-CLASS RECOMMENDATIONS
# =========================================================================== #
def recommend_per_class(class_agg: pd.DataFrame,
                        n_values: list,
                        alpha: float = DEFAULT_ALPHA,
                        lfs_tol: float = DEFAULT_LFS_TOL,
                        batch_size: int = DEFAULT_BATCH_SIZE,
                        vram_budget_mb: float = DEFAULT_VRAM_BUDGET_MB,
                        ) -> pd.DataFrame:
    """
    For each lifecycle class compute:

      Step 1.  ISV elbow (SLIC quality convergence, primary signal)
      Step 2.  FAE / MDV elbows (cross-validation)
      Step 3.  LFS stiffness ceiling (reaction-term ODE stiffness gate)
      Step 4.  Joint optimal N = argmax J(N) = quality − α·stiffness
      Step 5.  Majority vote: {ISV_elbow, FAE_elbow, MDV_elbow, joint_N}
      Step 6.  VRAM ceiling (diffusion-term stiffness + memory hard gate)
      Step 7.  recommended_N = min(voted_N, lfs_ceiling, vram_ceiling),
               snapped to nearest valid grid point.

    Columns in output DataFrame
    ───────────────────────────
    lifecycle_class, n_events,
    isv_elbow_N, fae_elbow_N, mdv_elbow_N,   — quality elbows
    lfs_ceiling_N,                             — reaction stiffness gate
    joint_optimal_N,                           — argmax J(N)
    majority_vote_N,                           — plurality of 4 candidates
    vram_ceiling_N,                            — VRAM / diffusion gate
    recommended_N,                             — final answer (gated)
    binding_constraint,                        — which gate was active
    isv_at_rec, lfs_at_rec,                   — metric values at rec N
    quality_at_rec, stiffness_at_rec,          — normalised J components
    joint_score_at_rec,                        — J(recommended_N)
    e_super_at_rec, vram_at_rec_mb,           — VRAM cost at rec N
    isv_ci_pct_at_rec,                         — data quality warning
    vote_breakdown,                            — Counter string
    """
    vram_ceil = vram_feasible_ceiling(n_values, batch_size, vram_budget_mb)
    records   = []

    for cls in sorted(class_agg["lifecycle_class"].unique()):
        grp = (class_agg[class_agg["lifecycle_class"] == cls]
               .sort_values("N").reset_index(drop=True))
        if len(grp) < 3:
            log.warning(f"  {cls}: only {len(grp)} N values — skipping "
                        f"(need ≥ 3 for elbow detection).")
            continue

        ns       = list(grp["N"].astype(int))
        isv      = list(grp["ISV_mean"])
        lfs      = list(grp["LFS_mean"])
        fae      = list(grp["FAE_mean"])
        mdv      = list(grp["MDV_mean"])
        tbr      = list(grp["TBR_mean"])
        br       = list(grp["BR_mean"])
        isv_ci   = list(grp["ISV_rel_ci_pct"])

        # Step 1-2: quality elbows
        isv_elbow = find_elbow(ns, isv)
        fae_elbow = find_elbow(ns, fae)
        mdv_elbow = find_elbow(ns, mdv)
        tbr_elbow = find_elbow(ns, tbr)
        br_elbow  = find_elbow(ns, [1.0 - b for b in br])

        # Step 3: LFS stiffness ceiling
        lfs_ceil = find_lfs_ceiling(ns, lfs, rise_tol=lfs_tol)

        # Step 4: joint score
        joint_scores, joint_n = compute_joint_scores(ns, isv, lfs, alpha=alpha)

        # Step 5: majority vote over primary quality metrics + joint
        vote_pool    = [isv_elbow, fae_elbow, mdv_elbow, joint_n]
        vote         = Counter(vote_pool)
        voted_n      = vote.most_common(1)[0][0]

        # Step 6 & 7: apply ceilings, snap to valid grid
        recommended_raw = min(voted_n, lfs_ceil, vram_ceil)
        recommended_n   = min(ns, key=lambda x: abs(x - recommended_raw))

        # Binding constraint analysis
        binding = _binding_constraint(voted_n, joint_n, lfs_ceil, vram_ceil)

        # Metrics at recommended N
        rec_idx = ns.index(recommended_n) if recommended_n in ns else -1
        def _at(lst): return lst[rec_idx] if rec_idx >= 0 else float("nan")

        isv_range = isv[0] - isv[-1] + 1e-12
        lfs_range = lfs[-1] - lfs[0] + 1e-12
        q_at_rec  = (isv[0] - _at(isv)) / isv_range  if rec_idx >= 0 else float("nan")
        s_at_rec  = (_at(lfs) - lfs[0]) / lfs_range  if rec_idx >= 0 else float("nan")
        j_at_rec  = q_at_rec - alpha * s_at_rec       if rec_idx >= 0 else float("nan")

        records.append({
            "lifecycle_class":    cls,
            "n_events":           int(grp["n_events"].max()),
            # Quality elbows
            "isv_elbow_N":        isv_elbow,
            "fae_elbow_N":        fae_elbow,
            "mdv_elbow_N":        mdv_elbow,
            "tbr_elbow_N":        tbr_elbow,
            "br_elbow_N":         br_elbow,
            # Stiffness and VRAM gates
            "lfs_ceiling_N":      lfs_ceil,
            "vram_ceiling_N":     vram_ceil,
            # Recommendation path
            "joint_optimal_N":    joint_n,
            "majority_vote_N":    voted_n,
            "recommended_N":      recommended_n,
            "binding_constraint": binding,
            # Metric values at recommended N
            "isv_at_rec":         round(float(_at(isv)), 6),
            "lfs_at_rec":         round(float(_at(lfs)), 6),
            "quality_at_rec":     round(float(q_at_rec), 4),
            "stiffness_at_rec":   round(float(s_at_rec), 4),
            "joint_score_at_rec": round(float(j_at_rec), 4),
            "e_super_at_rec":     estimate_e_super(recommended_n),
            "vram_at_rec_mb":     round(estimate_vram_mb(recommended_n, batch_size), 1),
            # Data quality diagnostics
            "isv_ci_pct_at_rec":  round(float(_at(isv_ci)), 1),
            # Full vote audit trail
            "vote_breakdown":     str(dict(Counter(vote_pool))),
        })

    return pd.DataFrame(records)


# =========================================================================== #
# REPORT BUILDER
# =========================================================================== #
def build_summary(all_df: pd.DataFrame,
                  n_values: list,
                  output_dir: str,
                  alpha: float          = DEFAULT_ALPHA,
                  lfs_tol: float        = DEFAULT_LFS_TOL,
                  batch_size: int       = DEFAULT_BATCH_SIZE,
                  vram_budget_mb: float = DEFAULT_VRAM_BUDGET_MB,
                  ) -> tuple[int, pd.DataFrame]:
    """
    Main entry point.  Builds all outputs and returns
    (global_recommended_N, per_class_recommendations_df).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Aggregate statistics ─────────────────────────────────────────────────
    agg       = aggregate_all(all_df)
    class_agg = aggregate_by_class(all_df)

    agg.to_csv(os.path.join(output_dir, "sweep_aggregate.csv"),   index=False)
    class_agg.to_csv(os.path.join(output_dir, "sweep_by_class.csv"), index=False)
    log.info("  Saved: sweep_aggregate.csv, sweep_by_class.csv")

    ns  = list(agg["N"].astype(int))
    isv = list(agg["ISV_mean"])
    lfs = list(agg["LFS_mean"])
    fae = list(agg["FAE_mean"])
    mdv = list(agg["MDV_mean"])
    tbr = list(agg["TBR_mean"])
    br  = list(agg["BR_mean"])

    # ── Global elbows / ceilings ─────────────────────────────────────────────
    global_elbows = {
        "ISV":  find_elbow(ns, isv),
        "FAE":  find_elbow(ns, fae),
        "MDV":  find_elbow(ns, mdv),
        "TBR":  find_elbow(ns, tbr),
        "BR":   find_elbow(ns, [1.0 - b for b in br]),
    }
    lfs_ceil_global  = find_lfs_ceiling(ns, lfs, rise_tol=lfs_tol)
    vram_ceil_global = vram_feasible_ceiling(ns, batch_size, vram_budget_mb)

    joint_scores, joint_n = compute_joint_scores(ns, isv, lfs, alpha=alpha)

    # Global vote: ISV + FAE + MDV + joint (consistent with per-class logic)
    g_vote_pool = [global_elbows["ISV"], global_elbows["FAE"],
                   global_elbows["MDV"], joint_n]
    g_vote      = Counter(g_vote_pool)
    g_voted_n   = g_vote.most_common(1)[0][0]
    g_best_n    = min(g_voted_n, lfs_ceil_global, vram_ceil_global)
    g_best_n    = min(ns, key=lambda x: abs(x - g_best_n))

    # ── Per-class recommendations ────────────────────────────────────────────
    class_rec = recommend_per_class(
        class_agg, ns,
        alpha=alpha, lfs_tol=lfs_tol,
        batch_size=batch_size, vram_budget_mb=vram_budget_mb)

    class_rec.to_csv(os.path.join(output_dir, "class_N_recommendations.csv"),
                     index=False)
    log.info(f"  Saved: class_N_recommendations.csv  "
             f"({len(class_rec)} classes)")

    # ── Build text report ────────────────────────────────────────────────────
    W   = 96
    sep = "=" * W
    div = "─" * W

    n_events  = int(all_df["event_id"].nunique())
    n_classes = int(all_df["lifecycle_class"].nunique())

    lines = [
        sep,
        "  DRIFT — Diagnostic N-Sweep Summary v2  |  Joint ISV + LFS Stiffness Analysis",
        f"  {n_events} events  ·  {n_classes} lifecycle classes  ·  "
        f"N grid: {ns}",
        f"  α = {alpha}  ·  LFS rise tolerance = {lfs_tol*100:.0f}%  ·  "
        f"VRAM budget = {vram_budget_mb} MB  ·  batch = {batch_size}",
        sep,
        "",
        "  PHYSICAL RATIONALE",
        "  " + div,
        "  ISV    Intra-superpixel variance — the SLIC-native objective.",
        "         Saturates as N grows and SLIC resolves all gradient structure.",
        "         ISV elbow = N where further increase yields diminishing quality.",
        "",
        "  LFS    Lagrangian Feature Smoothness — mean per-step feature change",
        "         along each tracked node trajectory.  LFS ≈ dh/dt magnitude.",
        "         HIGH LFS ⟹ reaction term R(h,e) produces rapid state changes",
        "         ⟹ stiff ODE on the physics side (reaction stiffness).",
        "         LFS CEILING = largest N where LFS ≤ LFS_min×(1+tol).",
        "         [Slide 29: 'Low LFS = ideal for Lagrangian graph-ODE integration.']",
        "",
        "  E_super  Pre-allocated edge count: grows ∝ N².  Proxies λ_max(L),",
        "           the Laplacian eigenvalue driving DIFFUSION stiffness.",
        "           Also the literal VRAM cost of the edge tensor.",
        "           [Slide 31: 'E_super is a feasibility ceiling, not a quality metric.']",
        "",
        "  Joint score J(N) = quality(N) − α·stiffness(N)",
        "    quality(N)   = (ISV₀−ISVₙ)/(ISV₀−ISV_last)   ∈ [0,1]",
        "    stiffness(N) = (LFSₙ−LFS₀)/(LFS_last−LFS₀)   ∈ [0,1]",
        f"    α = {alpha} → stiffness-conservative "
        f"(appropriate for dopri5 on convective events)",
        "",
    ]

    # ── Global table ─────────────────────────────────────────────────────────
    lines += [
        "  GLOBAL AGGREGATE  (all events, all classes)",
        "  " + div,
        f"  {'N':>6}  {'ISV':>9}  {'±CI%':>5}  {'LFS':>8}  "
        f"{'FAE':>7}  {'MDV':>9}  {'BR':>6}  {'TBR':>6}  "
        f"{'E_super':>11}  {'VRAM':>7}  {'J(N)':>7}  {'t(s)':>5}",
        "  " + div,
    ]
    for i, (_, r) in enumerate(agg.iterrows()):
        n = int(r["N"])
        j = joint_scores[i]
        tag = ""
        if n == g_best_n:
            tag = "  ◄ GLOBAL REC"
        elif n == lfs_ceil_global and n != g_best_n:
            tag = "  ← LFS ceil"
        elif n == vram_ceil_global and n != g_best_n:
            tag = "  ← VRAM ceil"
        lines.append(
            f"  {n:>6}  {r['ISV_mean']:>9.5f}  "
            f"{r['ISV_rel_ci_pct']:>5.1f}  "
            f"{r['LFS_mean']:>8.5f}  "
            f"{r['FAE_mean']:>7.4f}  {r['MDV_mean']:>9.5f}  "
            f"{r['BR_mean']:>6.3f}  {r['TBR_mean']:>6.3f}  "
            f"{int(r['E_super_mean']):>11,d}  "
            f"{r['VRAM_mean_MB']:>7.0f}  "
            f"{j:>7.3f}  {r['runtime_mean']:>5.1f}{tag}"
        )

    # ── Global elbow/ceiling summary ─────────────────────────────────────────
    lines += [
        "  " + div, "",
        "  GLOBAL ELBOW / CEILING SUMMARY",
        "  " + div,
        f"  {'Signal':<14}  {'N':>6}  Role / Interpretation",
        "  " + div,
    ]
    for m, ne in global_elbows.items():
        agree = "  ✓" if ne == g_best_n else ""
        role  = {
            "ISV": "SLIC quality elbow             (primary quality signal)",
            "FAE": "Kinematic alignment elbow       (cross-validation)",
            "MDV": "Intra-node homogeneity elbow    (cross-validation)",
            "TBR": "Temporal boundary stability     (cross-validation)",
            "BR":  "Boundary recall elbow           (cross-validation)",
        }[m]
        lines.append(f"  {m:<14}  {ne:>6}  {role}{agree}")

    lines += [
        f"  {'LFS ceiling':<14}  {lfs_ceil_global:>6}  "
        f"Reaction-stiffness gate     (max N before ODE stiffness rises)",
        f"  {'Joint N':<14}  {joint_n:>6}  "
        f"argmax J(N) = quality − {alpha}·stiffness"
        + ("  ✓" if joint_n == g_best_n else ""),
        f"  {'VRAM ceiling':<14}  {vram_ceil_global:>6}  "
        f"Diffusion-stiffness + memory gate  "
        f"({vram_budget_mb} MB, batch={batch_size})",
        "  " + div,
        f"  Global vote pool: {dict(g_vote)}",
        f"  Binding constraint: {_binding_constraint(g_voted_n, joint_n, lfs_ceil_global, vram_ceil_global)}",
        f"  ▶  GLOBAL RECOMMENDED N = {g_best_n}",
        "",
    ]

    # ── Per-class recommendation table ───────────────────────────────────────
    lines += [
        "  PER-CLASS RECOMMENDATIONS",
        "  " + div,
        f"  {'Class':<20}  {'n':>3}  "
        f"{'ISV_e':>5}  {'LFS_c':>5}  {'Jnt':>5}  {'Vote':>5}  "
        f"{'RecN':>5}  {'quality':>7}  {'stiff':>6}  {'J':>6}  "
        f"{'VRAM':>6}  {'CI%':>5}  Binding constraint",
        "  " + div,
    ]
    for _, row in class_rec.iterrows():
        lines.append(
            f"  {row['lifecycle_class']:<20}  {row['n_events']:>3}  "
            f"{row['isv_elbow_N']:>5}  {row['lfs_ceiling_N']:>5}  "
            f"{row['joint_optimal_N']:>5}  {row['majority_vote_N']:>5}  "
            f"{row['recommended_N']:>5}  "
            f"{row['quality_at_rec']:>7.3f}  "
            f"{row['stiffness_at_rec']:>6.3f}  "
            f"{row['joint_score_at_rec']:>6.3f}  "
            f"{row['vram_at_rec_mb']:>6.0f}  "
            f"{row['isv_ci_pct_at_rec']:>5.1f}  "
            f"{row['binding_constraint']}"
        )

    lines += [
        "  " + div,
        "  Columns: ISV_e=ISV elbow, LFS_c=LFS ceiling, Jnt=joint argmax,",
        "           Vote=majority vote, RecN=final recommendation.",
        "  CI% = relative 95% CI width of ISV at RecN. "
        "> 50% signals low statistical confidence.",
        "",
    ]

    # ── Data quality warnings ─────────────────────────────────────────────────
    high_ci = class_rec[class_rec["isv_ci_pct_at_rec"] > 50.0]
    if not high_ci.empty:
        lines += [
            "  ⚠  DATA QUALITY WARNINGS (ISV CI > 50% at recommended N)",
            "  " + div,
        ]
        for _, row in high_ci.iterrows():
            lines.append(
                f"  {row['lifecycle_class']:<20}  CI = "
                f"{row['isv_ci_pct_at_rec']:.1f}%  "
                f"(n={row['n_events']} events — "
                f"recommendation has low statistical confidence)"
            )
        lines.append("")

    # ── Stiffness breakdown ───────────────────────────────────────────────────
    lines += [
        "  STIFFNESS BREAKDOWN AT RECOMMENDED N",
        "  " + div,
        f"  {'Class':<20}  {'RecN':>5}  "
        f"{'ISV':>9}  {'LFS':>8}  "
        f"{'quality':>7}  {'stiffness':>9}  {'J(N)':>7}  "
        f"{'E_super':>11}",
        "  " + div,
    ]
    for _, row in class_rec.iterrows():
        lines.append(
            f"  {row['lifecycle_class']:<20}  {row['recommended_N']:>5}  "
            f"{row['isv_at_rec']:>9.5f}  {row['lfs_at_rec']:>8.5f}  "
            f"{row['quality_at_rec']:>7.3f}  "
            f"{row['stiffness_at_rec']:>9.3f}  "
            f"{row['joint_score_at_rec']:>7.3f}  "
            f"{row['e_super_at_rec']:>11,d}"
        )

    lines += [
        "  " + div,
        f"  quality   ∈ [0,1]  fraction of max ISV improvement achieved at RecN",
        f"  stiffness ∈ [0,1]  fraction of max LFS increase incurred at RecN",
        f"  J(N)      = quality − {alpha} × stiffness",
        "",
        sep,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    out_txt = os.path.join(output_dir, "sweep_summary_v2.txt")
    Path(out_txt).write_text(report, encoding="utf-8")
    log.info(f"  Saved: sweep_summary_v2.txt")

    return g_best_n, class_rec


# =========================================================================== #
# ENTRY POINT
# =========================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="DRIFT N-sweep — Joint ISV + LFS stiffness summary (v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir", type=str, default=RESULTS_DIR,
        help=f"Directory containing *_sweep.csv files. Default: {RESULTS_DIR}")
    parser.add_argument(
        "--n_values", type=int, nargs="+", default=DEFAULT_N_VALUES,
        help="N grid used in the sweep (for completeness validation).")
    parser.add_argument(
        "--alpha", type=float, default=DEFAULT_ALPHA,
        help=f"Stiffness penalty weight α in J(N)=quality−α·stiffness. "
             f"Default: {DEFAULT_ALPHA}.")
    parser.add_argument(
        "--lfs_tol", type=float, default=DEFAULT_LFS_TOL,
        help=f"LFS ceiling rise tolerance (fraction). "
             f"Default: {DEFAULT_LFS_TOL} = 10%%.")
    parser.add_argument(
        "--vram_budget_mb", type=float, default=DEFAULT_VRAM_BUDGET_MB,
        help=f"GPU VRAM budget in MB for edge tensor. "
             f"Default: {DEFAULT_VRAM_BUDGET_MB}.")
    parser.add_argument(
        "--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Training batch size for VRAM estimation. "
             f"Default: {DEFAULT_BATCH_SIZE}.")
    parser.add_argument(
        "--no_filter", action="store_true",
        help="Load all sweep CSVs without completeness filtering.")
    # Pre-computed aggregates path (skip raw CSV reload)
    parser.add_argument(
        "--agg_csv", type=str, default=None,
        help="Path to pre-computed sweep_aggregate.csv (skips raw CSV reload).")
    parser.add_argument(
        "--class_csv", type=str, default=None,
        help="Path to pre-computed sweep_by_class.csv.")

    args = parser.parse_args()
    n_values = sorted(set(args.n_values))

    if args.agg_csv and args.class_csv:
        # Fast path: use pre-computed aggregates directly
        log.info("  Using pre-computed aggregates (--agg_csv / --class_csv).")
        agg       = pd.read_csv(args.agg_csv)
        class_agg = pd.read_csv(args.class_csv)
        # Dummy all_df for event/class counts only
        all_df = pd.concat([
            pd.DataFrame({
                "event_id":        ["_placeholder"],
                "lifecycle_class": ["_placeholder"],
            })
        ])
        # Re-build per-class recommendations directly from pre-computed agg
        os.makedirs(args.results_dir, exist_ok=True)
        agg.to_csv(os.path.join(args.results_dir, "sweep_aggregate.csv"),  index=False)
        class_agg.to_csv(os.path.join(args.results_dir, "sweep_by_class.csv"), index=False)

        ns = list(agg["N"].astype(int))
        # Ensure rel CI columns exist
        if "ISV_rel_ci_pct" not in agg.columns:
            agg["ISV_rel_ci_pct"] = float("nan")
        if "ISV_rel_ci_pct" not in class_agg.columns:
            class_agg["ISV_rel_ci_pct"] = float("nan")
        if "LFS_rel_ci_pct" not in class_agg.columns:
            class_agg["LFS_rel_ci_pct"] = float("nan")

        class_rec = recommend_per_class(
            class_agg, ns,
            alpha=args.alpha, lfs_tol=args.lfs_tol,
            batch_size=args.batch_size, vram_budget_mb=args.vram_budget_mb)
        class_rec.to_csv(
            os.path.join(args.results_dir, "class_N_recommendations.csv"),
            index=False)
        log.info("  Saved class_N_recommendations.csv")
        print(class_rec[["lifecycle_class", "recommended_N",
                          "binding_constraint"]].to_string(index=False))
        return

    # Normal path: load raw CSVs
    required = None if args.no_filter else n_values
    all_df = load_sweep_csvs(args.results_dir, required_n_values=required)

    best_N, class_rec = build_summary(
        all_df,
        n_values       = n_values,
        output_dir     = args.results_dir,
        alpha          = args.alpha,
        lfs_tol        = args.lfs_tol,
        batch_size     = args.batch_size,
        vram_budget_mb = args.vram_budget_mb,
    )

    print(f"\nGlobal recommended N = {best_N}")
    print(f"Per-class recommendations saved to: "
          f"{os.path.join(args.results_dir, 'class_N_recommendations.csv')}")


if __name__ == "__main__":
    main()
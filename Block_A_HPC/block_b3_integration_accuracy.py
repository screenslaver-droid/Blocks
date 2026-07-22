"""
block_b3_integration_accuracy.py
=================================
Block B3: Integration Accuracy as a Function of N (blocks.tex, "B3:
Integration Accuracy as a Function of N", lines ~581-619).

WHAT THIS SCRIPT DOES NOT DO: it does not re-derive the graph construction,
data pipeline, or reaction/diffusion RHS. It IMPORTS block_a_solver_
benchmark.py as a library and calls its functions directly. This is
deliberate and load-bearing, not a style choice:

  * blocks.tex's own "Shared graph construction note" (line 388) says B3
    "require[s] a fully specified Lagrangian graph -- this is *exactly* the
    graph constructed in Block A, Section A.5" and that "[t]his identity is
    required for B3 ... results to be attributable to N alone rather than
    to a different graph construction than the one whose NFE was measured
    in Block A."
  * B3's own procedure (line 589) says to use "the same graph built in
    Block A ... and the competitive gated MLP reaction with stiffness-
    calibrated weights (Section A.2)".
  * You explicitly asked for identical weight realisations between Block A
    and B3, based on make_weights().

Given that, re-implementing build_graph_topological / make_weights / the
diffusion+reaction RHS a second time in this file would be *guaranteed* to
drift from Block A the next time either script changes -- exactly the
failure mode blocks.tex line ~29 warns about for TemporalSLIC_DEM. So B3
instead:

  1. Imports block_a_solver_benchmark.py as a module (`bA`) and calls
     bA.build_graph_topological(...) with the SAME calibrated c_sigma that
     Block A used -- read from block_a_calibration.json, which Block A's
     run_benchmark() already writes out (c_sigma, D, per-class sigma_init,
     rtol/atol). B3 does NOT re-run c_sigma/D/sigma_init calibration; it
     REQUIRES Block A to have been run first (see load_block_a_calibration).

  2. Calls bA.make_weights(sigma_init, seed) with the SAME (sigma_init,
     seed) pairs Block A used for a given (event, N, lifecycle_class).
     make_weights() is a pure function of (sigma_init, seed) --
     jax.random.PRNGKey(seed) with no other hidden state -- so calling it
     from this module produces BIT-IDENTICAL weight arrays to the call
     Block A made internally in benchmark_event(). There is no separate
     "reproduce" step needed beyond calling the same function with the
     same arguments; that identity is what make_weights's determinism
     guarantees.

  3. Calls bA.make_diffusion_rhs / bA.make_reaction_rhs / bA.
     imex_strang_integrate directly -- the exact same RHS/integrator code
     Block A benchmarks NFE with -- rather than a re-implementation.

Procedure (B3, Section B3):
  For each (event, N, seed):
    Step 1 -- tight-tolerance reference:
        h_ref = IMEX_integrate(h0, diff_fn, rxn_fn, rtol=1e-6, atol=1e-7)
    Step 2 -- training-tolerance solution:
        h_train = IMEX_integrate(h0, diff_fn, rxn_fn, rtol=rtol_train,
                                  atol=atol_train)
        (rtol_train/atol_train are read from block_a_calibration.json's
        "rtol"/"atol" -- i.e. whatever Block A actually used as ITS
        training tolerance for the NFE sweep, honouring blocks.tex A.3's
        "start at rtol=1e-2" default of 1e-2/1e-3 unless Block A's own A.3
        halving-search calibration overrode it.)
    Step 3 -- relative integration error:
        eps_int = ||h_train - h_ref||_F / ||h_ref||_F
  averaged over 5 random weight seeds and all events in the class (B3,
  line 607).

  Also records NFE_ref(N) and NFE_train(N) (B3, line 619) so
  NFE_train/NFE_ref can be inspected alongside eps_int(N).

KNOWN GAP (flagged, not silently guessed -- same convention as Block A's
module docstring): the per-class LFS ceiling N_LFS^c (Table 5.6) is not
available in this context, so this script cannot itself assert "the elbow
in eps_int(N) coincides with N_LFS^c within +-100 superpixels" (B3, line
617). It DOES compute a class-relative elbow heuristic (first N whose mean
eps_int exceeds elbow_factor x the flat-region baseline) as a proxy --
replace ELBOW_TARGET_N_LFS below with Table 5.6's real values and compare
directly once available.

Output: block_b3_results.csv   (one row per event x N x seed)
        block_b3_summary.csv   (per-class x N mean/std eps_int, NFE ratio)

Usage
-----
  python block_b3_integration_accuracy.py
  python block_b3_integration_accuracy.py --n_events 5 --seeds 5
  python block_b3_integration_accuracy.py --calibration-json /path/to/block_a_calibration.json

Requires block_a_solver_benchmark.py to be importable (same directory or
on PYTHONPATH), and block_a_calibration.json to already exist -- i.e. run
Block A's run_benchmark() at least once before running this script.
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# HPC THREAD-PINNING GUARD (matches block_a_solver_benchmark.py's own guard;
# duplicated here because THIS file imports numpy directly, below, before it
# imports block_a_solver_benchmark -- so bA's guard would fire too late to
# affect numpy's BLAS backend if we relied on it alone. See that file's
# guard for the full rationale: without this, each ProcessPoolExecutor
# worker's own BLAS/XLA backend would try to claim every core on the node,
# oversubscribing whatever ncpus a PBS chunk actually grants.)
for _env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_env_var, "1")
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT BLOCK A AS A LIBRARY -- this is the whole point of this file: reuse
# its graph construction, data pipeline, and make_weights() verbatim rather
# than re-implementing (and silently drifting from) any of it.
# ─────────────────────────────────────────────────────────────────────────────
import block_a_solver_benchmark as bA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

if not bA.JAX_OK:
    raise SystemExit(
        "JAX / Diffrax not available (see block_a_solver_benchmark.py's own "
        "import error above). Install: pip install jax[cpu] diffrax"
    )

import jax.numpy as jnp  # noqa: E402  (after bA's lazy-import guard)

# ─────────────────────────────────────────────────────────────────────────────
# B3 CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# Step 1 (blocks.tex line 593): tight-tolerance reference, "4 orders of
# magnitude tighter than training". Fixed, not read from calibration.
RTOL_REF = 1e-6
ATOL_REF = 1e-7

OUTPUT_CSV  = os.path.join(bA.DATA_ROOT, "block_b3_results.csv")
SUMMARY_CSV = os.path.join(bA.DATA_ROOT, "block_b3_summary.csv")

# KNOWN GAP -- see module docstring. Elbow heuristic only; replace with
# Table 5.6's real N_LFS^c per class for the actual validation condition.
ELBOW_FACTOR = 2.0   # flag N where mean eps_int > ELBOW_FACTOR x baseline


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION LOADING -- B3 REQUIRES Block A's calibration to already exist
# ─────────────────────────────────────────────────────────────────────────────

def load_block_a_calibration(path: str = bA.CALIB_JSON) -> dict:
    """
    B3 must use the SAME c_sigma, D, and per-class sigma_init that Block A
    calibrated (A.6 / A.2) -- not re-derive its own. Re-deriving here would
    risk a slightly different c_sigma/D/sigma_init than whatever Block A
    actually used for its NFE figures, which would silently break the
    "same graph ... same weight realisation" identity blocks.tex requires
    for B3's results to be attributable to N alone (see module docstring).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. B3 requires block_a_calibration.json, "
            f"produced by block_a_solver_benchmark.py's run_benchmark(). "
            f"Run Block A first."
        )
    with open(path) as f:
        calib = json.load(f)

    required = ["c_sigma", "D", "class_sigma_init"]
    missing = [k for k in required if k not in calib]
    if missing:
        raise KeyError(f"{path} is missing required key(s): {missing}")

    # rtol/atol are always written by Block A's _save_calibration_json
    # (defaulting to bA.RTOL_DEFAULT/ATOL_DEFAULT = 1e-2/1e-3 if Block A's
    # optional A.3 tolerance calibration wasn't run) -- fall back here too,
    # defensively, in case an older calibration JSON predates that field.
    calib.setdefault("rtol", bA.RTOL_DEFAULT)
    calib.setdefault("atol", bA.ATOL_DEFAULT)
    return calib


# ─────────────────────────────────────────────────────────────────────────────
# EVENT SAMPLING + DATA LOADING -- ported from bA.run_benchmark steps 1-3
# (verbatim logic; see that function for full commentary on each step)
# ─────────────────────────────────────────────────────────────────────────────

def sample_events(catalogue_path: str, n_events_per_class: int,
                   all_events: bool = False, random_state: int = 42) -> pd.DataFrame:
    """
    Same stratified-sampling call Block A's run_benchmark() uses (A.6), or
    every event in the catalogue when all_events=True -- mirrors Block A's
    --all-events flag so a B3 cluster sweep can cover the identical event
    set Block A's --all-events sweep did (required for eps_int/NFE_ratio
    to be comparable class-by-class against Block A's own NFE figures,
    not just weight/graph-construction-identical on a smaller subsample).
    """
    catalogue = pd.read_csv(catalogue_path, low_memory=False)
    if all_events:
        sampled = catalogue.reset_index(drop=True)
        log.info(f"[all-events mode] Using all {len(sampled)} catalogue events "
                 f"across {sampled['lifecycle_class'].nunique()} classes "
                 f"(no per-class cap)")
        return sampled
    sampled = (
        catalogue
        .groupby("lifecycle_class", group_keys=False)
        .apply(lambda g: g.sample(min(n_events_per_class, len(g)), random_state=random_state))
        .reset_index(drop=True)
    )
    log.info(f"Sampled {len(sampled)} events across "
             f"{sampled['lifecycle_class'].nunique()} classes")
    for cls, cnt in sampled["lifecycle_class"].value_counts().items():
        log.info(f"  {cls:<18} {cnt}")
    return sampled


def load_all_event_data(sampled: pd.DataFrame, sevir_catalog: pd.DataFrame) -> dict:
    """
    Ported from bA.run_benchmark's steps 1-3 (channel + DEM + CAPE +
    land-type loading), calling bA's own functions directly so B3's inputs
    are identical to Block A's -- see block_a_solver_benchmark.py for full
    commentary on each of these calls.
    """
    channels_by_event: dict[str, dict] = {}
    class_by_event:    dict[str, str]  = {}
    for _, row in sampled.iterrows():
        event_id = str(row["id"])
        channels = bA.load_event_multichannel(event_id, sevir_catalog)
        if channels is None:
            log.warning(f"  {event_id}: could not load channels, skipping.")
            continue
        channels_by_event[event_id] = channels
        class_by_event[event_id]    = str(row["lifecycle_class"])

    channel_shapes = {
        eid: ch["vil"].shape[1:] for eid, ch in channels_by_event.items()
    }
    dem_by_event = bA.prefetch_dem_for_events(
        list(channels_by_event.keys()), sevir_catalog, channel_shapes,
    )
    landtype_by_event = bA.prefetch_landtype_for_events(
        list(channels_by_event.keys()), sevir_catalog, channel_shapes,
    )

    event_data: dict[str, dict] = {}
    for event_id, channels in channels_by_event.items():
        dem_norm = dem_by_event.get(
            event_id, np.zeros(channel_shapes[event_id], dtype=np.float32)
        )
        H_evt, W_evt = channel_shapes[event_id]
        cape_norm = bA.load_cape_for_event(event_id, sevir_catalog, H_evt, W_evt)
        landtype_grid = landtype_by_event.get(
            event_id, np.zeros((H_evt, W_evt, 2), dtype=np.float32)
        )
        event_data[event_id] = {
            "lifecycle_class": class_by_event[event_id],
            "channels":        channels,
            "dem_norm":        dem_norm,
            "cape_norm":       cape_norm,
            "landtype_grid":   landtype_grid,
        }
    log.info(f"Loaded channel+DEM data for {len(event_data)}/{len(sampled)} sampled events")
    return event_data


# ─────────────────────────────────────────────────────────────────────────────
# B3 CORE: per-event (N x seed) integration-accuracy sweep
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_event_b3(
        event_id:        str,
        lifecycle_class: str,
        channels:        dict,
        dem_norm:        np.ndarray,
        n_values:        list[int],
        seeds:           list[int],
        c_sigma:         float,
        D:               float,
        sigma_init:      float,
        rtol_train:      float,
        atol_train:      float,
        cape_norm:       Optional[np.ndarray] = None,
        landtype_grid:   Optional[np.ndarray] = None,
) -> list[dict]:
    """
    For a single event, sweep N and seed, returning one row per (N, seed)
    with eps_int(N), NFE_ref(N), NFE_train(N).

    The graph for each N is built with bA.build_graph_topological -- the
    IDENTICAL function Block A's benchmark_event() calls, with the SAME
    c_sigma -- so this is exactly "the same graph built in Block A" (B3,
    line 589), not a parallel re-implementation.
    """
    rows = []

    for N in n_values:
        graph = bA.build_graph_topological(
            channels, dem_norm, N=N, c_sigma=c_sigma,
            cape_norm=cape_norm, landtype_grid=landtype_grid,
        )
        if graph is None:
            log.warning(f"    {event_id} N={N}: graph build failed, skipping.")
            continue

        env_jax = jnp.array(graph["env"])
        h0_jax  = jnp.array(graph["h0"])

        for seed in seeds:
            # Bit-identical to Block A's make_weights(sigma_init, seed)
            # call for this (lifecycle_class, seed) -- same function, same
            # arguments, same jax.random.PRNGKey(seed) derivation.
            weights  = bA.make_weights(sigma_init=sigma_init, seed=seed)
            diff_rhs = bA.make_diffusion_rhs(
                graph["senders"], graph["receivers"], graph["edge_weights"], D,
            )
            rxn_rhs  = bA.make_reaction_rhs(env_jax, weights)

            # ── Step 1: tight-tolerance reference ────────────────────────
            t0 = time.perf_counter()
            h_ref, nfe_diff_ref, nfe_rxn_ref, rej_diff_ref, rej_rxn_ref, ok_ref = \
                bA.imex_strang_integrate(
                    h0_jax, diff_rhs, rxn_rhs, rtol=RTOL_REF, atol=ATOL_REF,
                    event_id=event_id, lifecycle_class=lifecycle_class,
                )
            wall_ref = time.perf_counter() - t0
            nfe_ref  = nfe_diff_ref + nfe_rxn_ref

            # ── Step 2: training-tolerance solution ──────────────────────
            t0 = time.perf_counter()
            h_train, nfe_diff_tr, nfe_rxn_tr, rej_diff_tr, rej_rxn_tr, ok_train = \
                bA.imex_strang_integrate(
                    h0_jax, diff_rhs, rxn_rhs, rtol=rtol_train, atol=atol_train,
                    event_id=event_id, lifecycle_class=lifecycle_class,
                )
            wall_train = time.perf_counter() - t0
            nfe_train  = nfe_diff_tr + nfe_rxn_tr

            # ── Step 3: relative integration error ───────────────────────
            denom   = float(jnp.linalg.norm(h_ref)) + 1e-12
            eps_int = float(jnp.linalg.norm(h_train - h_ref)) / denom

            rows.append({
                "event_id":                event_id,
                "lifecycle_class":         lifecycle_class,
                "N":                       N,
                "N_actual":                graph["N_actual"],
                "E":                       graph["E"],
                "seed":                    seed,
                "c_sigma":                 c_sigma,
                "sigma_rbf":               graph["sigma_rbf"],
                "D":                       D,
                "sigma_init":              sigma_init,
                "rtol_ref":                RTOL_REF,
                "atol_ref":                ATOL_REF,
                "rtol_train":              rtol_train,
                "atol_train":              atol_train,
                "nfe_ref":                 nfe_ref,
                "nfe_train":               nfe_train,
                "nfe_ratio_train_over_ref": round(nfe_train / (nfe_ref + 1), 6),
                "rejected_ref":            rej_diff_ref + rej_rxn_ref,
                "rejected_train":          rej_diff_tr + rej_rxn_tr,
                "ref_converged":           ok_ref,
                "train_converged":         ok_train,
                "eps_int":                 eps_int,
                "wall_ref_sec":            round(wall_ref, 4),
                "wall_train_sec":          round(wall_train, 4),
            })

    return rows


def _benchmark_event_worker_b3(payload: dict) -> dict:
    """Process-pool entry point -- mirrors bA._benchmark_event_worker."""
    rows = benchmark_event_b3(
        event_id=payload["event_id"],
        lifecycle_class=payload["cls"],
        channels=payload["channels"],
        dem_norm=payload["dem_norm"],
        n_values=payload["n_values"],
        seeds=payload["seeds"],
        c_sigma=payload["c_sigma"],
        D=payload["D"],
        sigma_init=payload["sigma_init"],
        rtol_train=payload["rtol_train"],
        atol_train=payload["atol_train"],
        cape_norm=payload.get("cape_norm"),
        landtype_grid=payload.get("landtype_grid"),
    )
    return {"event_id": payload["event_id"], "rows": rows}


def _log_event_progress(out: dict) -> None:
    if not out["rows"]:
        log.warning(f"  {out['event_id']}: no rows produced.")
        return
    df = pd.DataFrame(out["rows"])
    worst = df["eps_int"].max()
    log.info(f"  {out['event_id']}: {len(df)} rows, max eps_int={worst:.3e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN B3 SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark_b3(
        catalogue_path:      str  = bA.CATALOGUE_PATH,
        calibration_path:    str  = bA.CALIB_JSON,
        n_events_per_class:  int  = bA.N_EVENTS_PER_CLASS,
        n_seeds:             int  = bA.N_SEEDS,
        n_values:            list = bA.N_VALUES,
        n_workers:           Optional[int] = None,
        all_events:          bool = False,
) -> pd.DataFrame:
    """
    1. Load block_a_calibration.json (c_sigma, D, per-class sigma_init,
       rtol/atol) -- produced by Block A, NOT recalibrated here.
    2. Load event_catalogue.csv + SEVIR CATALOG.csv, stratified-sample
       events (same random_state=42 as Block A's A.6 step).
    3. Load channels + DEM + CAPE + land-type per sampled event (identical
       calls to Block A's own loaders).
    4. For each event x N x seed: build the Block-A graph, instantiate the
       Block-A weights, run the tight- and training-tolerance IMEX
       integrations, record eps_int(N) and NFE_ref/NFE_train.
    """
    calib = load_block_a_calibration(calibration_path)
    c_sigma:     float = calib["c_sigma"]
    D:           float = calib["D"]
    class_sigma: dict  = calib["class_sigma_init"]
    rtol_train:  float = calib["rtol"]
    atol_train:  float = calib["atol"]
    log.info(f"[B3] Using Block A calibration from {calibration_path}: "
             f"c_sigma={c_sigma}  D={D:.6g}  rtol_train={rtol_train:.2e}  "
             f"atol_train={atol_train:.2e}")

    if not os.path.exists(catalogue_path):
        raise FileNotFoundError(f"event_catalogue.csv not found at {catalogue_path}")
    sevir_catalog_path = os.path.join(bA.DATA_ROOT, "CATALOG.csv")
    if not os.path.exists(sevir_catalog_path):
        raise FileNotFoundError(f"SEVIR CATALOG.csv not found at {sevir_catalog_path}")
    sevir_catalog = pd.read_csv(sevir_catalog_path, low_memory=False)

    sampled = sample_events(catalogue_path, n_events_per_class, all_events=all_events)
    event_data = load_all_event_data(sampled, sevir_catalog)
    if not event_data:
        return pd.DataFrame()

    seeds = list(range(n_seeds))
    payloads = []
    for event_id, d in event_data.items():
        cls = d["lifecycle_class"]
        if cls not in class_sigma:
            log.warning(
                f"  {event_id}: lifecycle_class '{cls}' has no calibrated "
                f"sigma_init in {calibration_path} (class_sigma_init keys: "
                f"{list(class_sigma.keys())}) -- falling back to sigma_init=1.0. "
                f"This means B3's weights for this event will NOT match "
                f"whatever Block A actually used; re-run Block A with this "
                f"class present to fix."
            )
        payloads.append({
            "event_id":      event_id,
            "cls":           cls,
            "channels":      d["channels"],
            "dem_norm":      d["dem_norm"],
            "n_values":      n_values,
            "seeds":         seeds,
            "c_sigma":       c_sigma,
            "D":             D,
            "sigma_init":    class_sigma.get(cls, 1.0),
            "rtol_train":    rtol_train,
            "atol_train":    atol_train,
            "cape_norm":     d.get("cape_norm"),
            "landtype_grid": d.get("landtype_grid"),
        })

    resolved_workers = max(1, min(n_workers or (os.cpu_count() or 1), len(payloads)))
    outputs: dict[str, dict] = {}

    if resolved_workers <= 1:
        log.info("Running B3 sweep sequentially (n_workers=1) …")
        for payload in tqdm(payloads, desc="Events"):
            out = _benchmark_event_worker_b3(payload)
            outputs[out["event_id"]] = out
            _log_event_progress(out)
    else:
        log.info(f"Running B3 sweep across {resolved_workers} worker "
                 f"process(es) for {len(payloads)} event(s) …")
        ctx = mp.get_context("spawn")   # JAX + fork don't mix; spawn is required
        with ProcessPoolExecutor(max_workers=resolved_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_benchmark_event_worker_b3, p): p["event_id"]
                       for p in payloads}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Events"):
                event_id = futures[fut]
                try:
                    out = fut.result()
                except Exception as exc:
                    log.error(f"  [ERROR] {event_id}: worker failed: {exc}")
                    continue
                outputs[out["event_id"]] = out
                _log_event_progress(out)

    all_rows = []
    for event_id in event_data:
        if event_id in outputs:
            all_rows.extend(outputs[event_id]["rows"])

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY AGGREGATION + ELBOW DETECTION (B3 validation condition, line 617)
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_b3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-class x N: mean/std eps_int (averaged over 5 seeds and all events
    in the class, per B3 line 607), mean NFE_ref/NFE_train, and the
    NFE_train/NFE_ref ratio (B3 line 619).
    """
    if df.empty:
        return pd.DataFrame()

    agg = (
        df.groupby(["lifecycle_class", "N"])
        .agg(
            n_samples=("eps_int", "size"),
            eps_int_mean=("eps_int", "mean"),
            eps_int_std=("eps_int", "std"),
            nfe_ref_mean=("nfe_ref", "mean"),
            nfe_train_mean=("nfe_train", "mean"),
            nfe_ratio_mean=("nfe_ratio_train_over_ref", "mean"),
            frac_ref_converged=("ref_converged", "mean"),
            frac_train_converged=("train_converged", "mean"),
        )
        .reset_index()
        .sort_values(["lifecycle_class", "N"])
    )
    return agg


def detect_elbow(summary: pd.DataFrame, elbow_factor: float = ELBOW_FACTOR) -> pd.DataFrame:
    """
    Class-relative elbow heuristic (KNOWN GAP -- see module docstring):
    baseline = mean eps_int at the smallest N for that class; elbow_N =
    first N whose mean eps_int exceeds elbow_factor x baseline. This is a
    STAND-IN for "the elbow coincides with N_LFS^c from Table 5.6" (B3
    line 617) -- replace with a direct comparison once Table 5.6's
    per-class N_LFS^c values are available in this context.
    """
    rows = []
    for cls, g in summary.groupby("lifecycle_class"):
        g = g.sort_values("N")
        baseline = g["eps_int_mean"].iloc[0]
        elbow_n = None
        for _, r in g.iterrows():
            if r["eps_int_mean"] > elbow_factor * baseline:
                elbow_n = int(r["N"])
                break
        rows.append({
            "lifecycle_class": cls,
            "baseline_eps_int": baseline,
            "elbow_N_heuristic": elbow_n,
            "max_N_swept": int(g["N"].max()),
        })
    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame, elbow: pd.DataFrame) -> None:
    if summary.empty:
        log.warning("No B3 results to summarise.")
        return
    log.info("\n" + "=" * 78)
    log.info("B3 SUMMARY -- integration accuracy (eps_int) and NFE ratio vs N")
    log.info("=" * 78)
    for cls, g in summary.groupby("lifecycle_class"):
        log.info(f"\n{cls}:")
        for _, r in g.sort_values("N").iterrows():
            log.info(
                f"  N={int(r['N']):<5} eps_int={r['eps_int_mean']:.3e} "
                f"(+/-{r['eps_int_std']:.1e})  "
                f"NFE_ref={r['nfe_ref_mean']:.0f}  "
                f"NFE_train={r['nfe_train_mean']:.0f}  "
                f"ratio={r['nfe_ratio_mean']:.3f}  "
                f"conv(ref/train)={r['frac_ref_converged']:.2f}/"
                f"{r['frac_train_converged']:.2f}"
            )
    log.info("\nElbow heuristic (KNOWN GAP -- see module docstring):")
    for _, r in elbow.iterrows():
        log.info(
            f"  {r['lifecycle_class']:<15} elbow_N~{r['elbow_N_heuristic']} "
            f"(baseline eps_int={r['baseline_eps_int']:.3e}, "
            f"swept up to N={r['max_N_swept']})"
        )
    log.info("=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Block B3: integration accuracy vs N")
    parser.add_argument("--catalogue", type=str, default=bA.CATALOGUE_PATH,
                        help="Path to event_catalogue.csv")
    parser.add_argument("--calibration-json", type=str, default=bA.CALIB_JSON,
                        help="Path to block_a_calibration.json (from Block A)")
    parser.add_argument("--n_events", type=int, default=bA.N_EVENTS_PER_CLASS,
                        help=f"Events per class (default: {bA.N_EVENTS_PER_CLASS})")
    parser.add_argument("--seeds", type=int, default=bA.N_SEEDS,
                        help=f"Random weight seeds per (event, N) (default: {bA.N_SEEDS})")
    parser.add_argument("--n_values", type=int, nargs="+", default=bA.N_VALUES,
                        help=f"N sweep values (default: {bA.N_VALUES})")
    parser.add_argument("--n_workers", type=int, default=None,
                        help="Process-pool workers (default: min(cpu_count, n_events))")
    parser.add_argument("--all-events", action="store_true",
                        help="Ignore --n_events and use EVERY event in "
                             "event_catalogue.csv (all classes, no per-class "
                             "cap) -- mirrors Block A's --all-events, so a "
                             "B3 cluster sweep covers the identical event "
                             "set Block A's own --all-events sweep did.")
    parser.add_argument("--output", type=str, default=OUTPUT_CSV)
    parser.add_argument("--summary-output", type=str, default=SUMMARY_CSV)
    args = parser.parse_args()

    log.info(f"[B3] events/class={args.n_events}  seeds={args.seeds}  "
             f"N values={args.n_values}  all_events={args.all_events}")

    df = run_benchmark_b3(
        catalogue_path=args.catalogue,
        calibration_path=args.calibration_json,
        n_events_per_class=args.n_events,
        n_seeds=args.seeds,
        n_values=args.n_values,
        n_workers=args.n_workers,
        all_events=args.all_events,
    )

    if df.empty:
        log.error("B3 sweep produced no results.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info(f"Wrote {len(df)} rows -> {args.output}")

    summary = build_summary_b3(df)
    summary.to_csv(args.summary_output, index=False)
    log.info(f"Wrote summary -> {args.summary_output}")

    elbow = detect_elbow(summary)
    print_summary_table(summary, elbow)


if __name__ == "__main__":
    main()
"""
run_b1_per_class.py
====================
Block B1: single-event-per-lifecycle-class CC-vs-SLIC comparison.

Wraps b1_diagnostic.py (the fixed version of
visualize_b1_cc_vs_slic_roi-masked_precision_modified_f1.py) as a module and
runs it once per lifecycle class, mirroring the stratified-sampling pattern
in block_a_solver_benchmark.py's run_benchmark() -- but with exactly ONE
event per class instead of N_EVENTS_PER_CLASS (=5), since B1 is a per-event
diagnostic (boundary alignment of one storm against one SLIC segmentation),
not a seeded statistical benchmark like Block A's IMEX/DOPRI5 sweep.

What this script does, step by step
------------------------------------
  1. Load event_catalogue.csv (id, lifecycle_class) -- same file and same
     `.groupby("lifecycle_class").apply(lambda g: g.sample(...))` pattern
     as block_a_solver_benchmark.run_benchmark()'s "A.6: stratified sample"
     step, but with sample size 1 instead of N_EVENTS_PER_CLASS.
  2. For each sampled (class, event_id): run b1_diagnostic's Step 1-3
     pipeline across n_list, WITHOUT going through its plotting-heavy
     run()/visualize_event() entry points -- those are single-event,
     figure-producing functions. Here we call the same lower-level
     primitives directly (build_all_threshold_cells, compute_slic_t0,
     compute_boundary_f1_all_stages, compute_multi_threshold_f1,
     compute_per_cell_iou) so we get a clean per-N row per class instead of
     a PNG. Figures are still available on request via --make_figures,
     which calls the module's own visualize_event() per selected event.
  3. Per class: N_F1 = soft_argmax_n(F1_nested) [upper bound],
     N_IoU = soft_argmax_n(mIoU_weighted @ tau_high) [lower bound],
     compared against N*_class (Table 5.7 in the thesis; see
     N_STAR_BY_CLASS below -- KNOWN GAP, see note there) as the
     validation interval N_IoU <= N* <= N_F1 (spec Step B5).
  4. Save b1_per_class_results.csv (one row per class x N) and
     b1_per_class_summary.csv (one row per class), print a summary table
     in the same style as block_a_solver_benchmark.print_summary_table().

Design choices carried over verbatim from b1_diagnostic.py, per the
author's explicit instruction (not re-litigated here):
  * DEFAULT_TOLERANCE_PX = 0, not the spec's 2 px -- at delta=2 the boundary
    F1 saturates and stops discriminating between different N, which
    defeats the point of an N-sweep comparison; 0 keeps F1(N) sensitive to
    N at the cost of stricter (pixel-exact) matching.
  * SLIC features are still built from the frame-0 slice only (B1
    compares one frame of the storm against one SLIC segmentation of that
    same frame) -- normalisation *statistics* were the only part that
    needed to span all T frames (fixed in b1_diagnostic.py), not the
    feature cube itself.

KNOWN GAPS relative to the spec (flagged, not silently guessed -- same
convention as block_a_solver_benchmark.py's "KNOWN GAPS" section):
  * N_STAR_BY_CLASS below is a placeholder: every class points at the same
    global N*=1150 used in block_a_solver_benchmark.py (A.6's calibration
    anchor), since Table 5.7's actual per-class N* values were not
    available in this context. Replace with the real per-class figures
    before treating the validation-interval check as meaningful --
    right now every class is being checked against the SAME number, which
    silently assumes all seven lifecycle classes converge at one N*.
  * Single event per class is a spot-check, not a statistical estimate --
    Block A draws 5 events/class specifically so per-class NFE numbers
    have some spread; a per-class N_F1/N_IoU here is a single (event,
    class) draw and can vary a lot run to run. --seed and
    --event_for_class let you pin/replace individual draws for
    repeatability and for checking sensitivity to which event got picked.

Usage
-----
    python run_b1_per_class.py
    python run_b1_per_class.py --classes STEADY RAPID_GROWTH QUIESCENT
    python run_b1_per_class.py --n_values 250 500 750 1000 1150 1500
    python run_b1_per_class.py --event_for_class STEADY=S123456 RAPID_GROWTH=S654321
    python run_b1_per_class.py --make_figures --workers 4
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION -- adjust to match your environment
# ─────────────────────────────────────────────────────────────────────────────
# event_catalogue.csv: same file block_a_solver_benchmark.py reads for its
# "id" / "lifecycle_class" stratified sample (its CATALOGUE_PATH). Point
# this at the same file so the two Blocks are drawing from the identical
# class labels.
EVENT_CATALOGUE_PATH = r"/media/sid_nair/OS/Users/Siddharth Nair/BTP-2/GAT-ODE/SEVIR/event_catalogue.csv"

# Path to b1_diagnostic.py (the fixed module). Defaults to "next to this
# script"; override if you keep it elsewhere.
B1_MODULE_PATH = r"/media/sid_nair/OS/Users/Siddharth Nair/BTP-2/GAT-ODE/Code/visualize_b1_cc_vs_slic_roi-masked_precision_modified_f1.py"

OUTPUT_CSV  = r"/media/sid_nair/OS/Users/Siddharth Nair/BTP-2/GAT-ODE/Blocks/B/B1_Diagnostics/b1_per_class_results.csv"
SUMMARY_CSV = r"/media/sid_nair/OS/Users/Siddharth Nair/BTP-2/GAT-ODE/Blocks/B/B1_Diagnostics/b1_per_class_summary.csv"
OUT_DIR     = r"/media/sid_nair/OS/Users/Siddharth Nair/BTP-2/GAT-ODE/Blocks/B/B1_Diagnostics/b1_per_class_figures"   # only used with --make_figures

# N sweep -- matches block_a_solver_benchmark.N_VALUES by default so the
# same N grid is being compared across both blocks.
DEFAULT_N_VALUES = [250, 500, 750, 950, 1150, 1500]

# KNOWN GAP (see module docstring): placeholder until Table 5.7's real
# per-class N* values are available. All classes point at Block A's
# global anchor N*=1150 (block_a_solver_benchmark.N_STAR).
N_STAR_BY_CLASS: dict[str, int] = {
    "RAPID_GROWTH": 950,
    "GROWTH_DECAY": 1150,
    "EPISODIC":     750,
    "PLATEAU":      950,
    "RAPID_DECAY":  1150,
    "STEADY":       850,
    "QUIESCENT":    850,
}

DEFAULT_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIC IMPORT of b1_diagnostic.py as a module
# ─────────────────────────────────────────────────────────────────────────────
def _load_b1_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"b1_diagnostic.py not found at {path}. Set B1_MODULE_PATH or "
            f"--b1_module to point at the fixed module."
        )
    spec = importlib.util.spec_from_file_location("b1_diagnostic", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["b1_diagnostic"] = mod   # so it's picklable by name in worker processes
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────────────────────
# A.6-STYLE STRATIFIED SAMPLE, n=1 PER CLASS
# ─────────────────────────────────────────────────────────────────────────────
def select_one_event_per_class(
        catalogue_path: str,
        classes: Optional[list[str]] = None,
        seed: int = DEFAULT_SEED,
        overrides: Optional[dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Mirrors block_a_solver_benchmark.run_benchmark()'s "A.6: stratified
    sample" step:

        sampled = (
            catalogue
            .groupby("lifecycle_class", group_keys=False)
            .apply(lambda g: g.sample(min(n_events_per_class, len(g)),
                                       random_state=42))
            .reset_index(drop=True)
        )

    with n_events_per_class fixed at 1 (single event per class), an
    optional class subset, and per-class event overrides for
    repeatability / sensitivity checks.

    Returns a DataFrame with columns ["id", "lifecycle_class"] (plus
    whatever else event_catalogue.csv carries), one row per class.
    """
    if not os.path.exists(catalogue_path):
        raise FileNotFoundError(
            f"event_catalogue.csv not found at {catalogue_path}.\n"
            f"Run Growth_Decay_Classify.py first (same requirement as "
            f"block_a_solver_benchmark.py)."
        )
    catalogue = pd.read_csv(catalogue_path, low_memory=False)
    log.info(f"Loaded event catalogue: {len(catalogue):,} events")

    if classes:
        catalogue = catalogue[catalogue["lifecycle_class"].isin(classes)]
        missing = set(classes) - set(catalogue["lifecycle_class"].unique())
        if missing:
            log.warning(f"Classes not present in catalogue: {sorted(missing)}")

    # NOTE: intentionally NOT
    #   catalogue.groupby("lifecycle_class", group_keys=False)
    #            .apply(lambda g: g.sample(min(1, len(g)), random_state=seed))
    # -- the pattern block_a_solver_benchmark.run_benchmark() uses for its
    # own "A.6: stratified sample" step. On pandas >=2.2 (confirmed on
    # 3.0.2), groupby(...).apply() silently DROPS the grouping column from
    # the result whenever the applied function returns it unchanged, which
    # crashes the very next `sampled["lifecycle_class"]` access with a
    # KeyError. Sampling per group explicitly and concatenating sidesteps
    # that pandas version-dependent apply behaviour entirely. If Block A
    # hits a KeyError: 'lifecycle_class' at the same step, this is why --
    # worth porting this fix back there too.
    sampled = pd.concat(
        [g.sample(min(1, len(g)), random_state=seed)
         for _, g in catalogue.groupby("lifecycle_class")],
        ignore_index=True,
    )

    if overrides:
        for cls, event_id in overrides.items():
            rows = catalogue[
                (catalogue["lifecycle_class"] == cls) & (catalogue["id"] == event_id)
            ]
            if rows.empty:
                log.warning(
                    f"--event_for_class override {cls}={event_id} not found in "
                    f"catalogue for that class; keeping the sampled draw instead."
                )
                continue
            sampled = sampled[sampled["lifecycle_class"] != cls]
            sampled = pd.concat([sampled, rows.iloc[[0]]], ignore_index=True)

    log.info(f"Selected {len(sampled)} events across "
             f"{sampled['lifecycle_class'].nunique()} classes (1 event/class)")
    for _, row in sampled.sort_values("lifecycle_class").iterrows():
        log.info(f"  {row['lifecycle_class']:<18} {row['id']}")

    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# PER-EVENT, PER-N EVALUATION -- direct calls to b1_diagnostic's primitives,
# same sequence run()/visualize_event() use internally, without the
# plotting overhead.
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_event(
        b1,
        event_id: str,
        lifecycle_class: str,
        catalog: pd.DataFrame,
        n_list: list[int],
        sigma_cell: float,
        tolerance_px: int,
        margin_px: int,
        split_merged: bool,
        n_max_sweep: int,
        use_dem: bool,
        dem_norm: Optional[np.ndarray] = None,
) -> list[dict]:
    """
    Runs Steps 1-3 of b1_diagnostic.py for a single event across n_list.
    Returns one dict per N with every metric needed for the per-class
    summary (F1_nested + ablation stages, per-level F1/precision/recall,
    Scheme G/P/F composites, per-level mIoU weighted/unweighted).

    `dem_norm`: if the caller already prefetched this event's DEM tile
    (see run_per_class's up-front prefetch_dem_for_extents step, mirroring
    block_a_solver_benchmark.prefetch_dem_for_events), pass it in here so
    it isn't fetched a second time. If use_dem=True and dem_norm is None
    (e.g. called directly, outside run_per_class), this falls back to
    fetching inline -- b1.fetch_dem_norm is itself now backed by the
    extent-keyed disk cache, so an inline fetch here still only hits the
    network once per unique extent across the whole process.
    """
    data, extent = b1.load_channels(event_id, catalog)
    if "vil" not in data:
        log.warning(f"  {event_id} [{lifecycle_class}]: VIL channel not found, skipping.")
        return []

    vil_phys     = b1.decode_vil_physical(data["vil"][0])
    fused_frame0 = b1.build_fused_cube_frame0(data)   # now normalises over all T frames

    if use_dem and dem_norm is None:
        dem_norm = b1.fetch_dem_norm(extent)
    elif not use_dem:
        dem_norm = None

    vil_smooth, level_data = b1.build_all_threshold_cells(
        vil_phys, sigma_cell=sigma_cell, n_max_sweep=n_max_sweep,
        split_merged=split_merged,
    )
    active_levels = [lv for lv in b1.LEVEL_NAMES if level_data[lv]["active"]]
    log.info(f"  {event_id} [{lifecycle_class}]: active levels {active_levels}  "
             f"M=" + ", ".join(f"{lv}:{level_data[lv]['M']}" for lv in b1.LEVEL_NAMES))

    rows = []
    for n_req in n_list:
        slic_labels   = b1.compute_slic_t0(fused_frame0, n_req, dem_norm=dem_norm)
        actual_n      = len(np.unique(slic_labels))
        boundary_slic = b1.extract_boundary(slic_labels, require_positive=False)

        stages = b1.compute_boundary_f1_all_stages(
            level_data, boundary_slic,
            tolerance_px=tolerance_px, margin_px=margin_px,
        )
        mt = b1.compute_multi_threshold_f1(
            level_data, vil_smooth, boundary_slic,
            tolerance_px=tolerance_px, margin_px=margin_px,
        )
        miou_dict = b1.compute_per_cell_iou(level_data, slic_labels)

        row = {
            "event_id": event_id,
            "lifecycle_class": lifecycle_class,
            "N_requested": n_req,
            "N_actual": actual_n,
            "zero_cell": stages["zero_cell"],
            "active_levels": ",".join(stages.get("active_levels", [])),
            # Stage 0/1/2 ablation (Step 3)
            "f1_stage0": stages["f1_stage0"],
            "f1_stage1": stages["f1_stage1"],
            "f1_nested": stages["f1_nested"],
            "precision_roi": stages["precision_roi"],
            "recall": stages["recall"],
            # Scheme G/P/F composites (Step 3e)
            "f1_multi_G": mt["f1_multi_G"],
            "f1_multi_P": mt["f1_multi_P"],
            "f1_multi_F": mt["f1_multi_F"],
        }
        for lv in b1.LEVEL_NAMES:
            pl = mt["per_level"][lv]
            row[f"f1_{lv}"]        = pl["f1"]
            row[f"precision_{lv}"] = pl["precision"]
            row[f"recall_{lv}"]    = pl["recall"]
            md = miou_dict[lv]
            row[f"miou_{lv}"]          = md["miou"]
            row[f"miou_weighted_{lv}"] = md["miou_weighted"]
            row[f"n_instances_{lv}"]   = md["n_instances"]
        rows.append(row)

    return rows


def _class_event_worker(payload: dict) -> dict:
    """
    Top-level (picklable) unit of work for the ProcessPoolExecutor below --
    mirrors block_a_solver_benchmark._benchmark_event_worker's structure,
    but simpler: b1_diagnostic.py has no JAX/XLA dependency, so the default
    ("fork") multiprocessing start method is fine here (Block A has to be
    careful about that; this doesn't).
    """
    b1 = _load_b1_module(payload["b1_module_path"])
    catalog = pd.read_csv(payload["b1_catalog_path"], low_memory=False)
    rows = evaluate_event(
        b1=b1,
        event_id=payload["event_id"],
        lifecycle_class=payload["lifecycle_class"],
        catalog=catalog,
        n_list=payload["n_list"],
        sigma_cell=payload["sigma_cell"],
        tolerance_px=payload["tolerance_px"],
        margin_px=payload["margin_px"],
        split_merged=payload["split_merged"],
        n_max_sweep=payload["n_max_sweep"],
        use_dem=payload["use_dem"],
        dem_norm=payload.get("dem_norm"),
    )
    return {"event_id": payload["event_id"], "lifecycle_class": payload["lifecycle_class"],
            "rows": rows}


def _get_event_extent(catalog: pd.DataFrame, event_id: str):
    """[llcrnrlon, urcrnrlon, llcrnrlat, urcrnrlat] for an event, read
    straight from the catalog row -- same fields b1_diagnostic.load_channels
    uses, but without loading the (T,H,W) VIL/IR arrays, so the up-front
    DEM prefetch below can resolve every sampled event's extent without
    paying for a full channel load per event first."""
    rows = catalog[catalog["id"] == event_id]
    if rows.empty:
        return None
    r = rows.iloc[0]
    try:
        return [float(r["llcrnrlon"]), float(r["urcrnrlon"]),
                float(r["llcrnrlat"]), float(r["urcrnrlat"])]
    except KeyError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────
def run_per_class(
        b1,
        event_catalogue_path: str = EVENT_CATALOGUE_PATH,
        classes: Optional[list[str]] = None,
        n_list: list[int] = DEFAULT_N_VALUES,
        sigma_cell: float = 1.0,
        tolerance_px: Optional[int] = None,
        margin_px: Optional[int] = None,
        split_merged: bool = False,
        n_max_sweep: Optional[int] = None,
        use_dem: bool = False,
        seed: int = DEFAULT_SEED,
        overrides: Optional[dict[str, str]] = None,
        n_workers: Optional[int] = None,
        make_figures: bool = False,
        out_dir: str = OUT_DIR,
) -> pd.DataFrame:
    tolerance_px = b1.DEFAULT_TOLERANCE_PX if tolerance_px is None else tolerance_px
    margin_px    = b1.DEFAULT_MARGIN_PX    if margin_px    is None else margin_px
    n_max_sweep  = b1.DEFAULT_N_MAX_SWEEP  if n_max_sweep  is None else n_max_sweep

    if not os.path.exists(b1.CATALOG_PATH):
        raise FileNotFoundError(
            f"SEVIR CATALOG.csv not found at {b1.CATALOG_PATH} "
            f"(b1_diagnostic.CATALOG_PATH) -- edit that constant to match "
            f"your environment."
        )

    sampled = select_one_event_per_class(
        event_catalogue_path, classes=classes, seed=seed, overrides=overrides,
    )
    if sampled.empty:
        log.error("No events sampled -- check event_catalogue.csv / --classes.")
        return pd.DataFrame()

    # ── Up-front DEM prefetch (mirrors block_a_solver_benchmark's fix in
    #    prefetch_dem_for_events): resolve every sampled event's DEM tile
    #    ONCE, in parallel, before the main sweep starts, instead of each
    #    event fetching inline the first time it's touched -- which used
    #    to mean up to n_classes sequential Planetary Computer round-trips
    #    sitting in front of the CPU-bound part of the sweep whenever
    #    --workers=1, and n_classes concurrent but UNCOORDINATED fetches
    #    (no shared cache visibility across worker processes until each
    #    finished) when parallelised. b1.fetch_dem_norm's disk cache means
    #    this is a no-op after the first run anyway, but doing it here
    #    still collapses a cold-cache run from ~n_classes sequential
    #    round-trips down to ~1 round-trip's worth of wall time. ─────────
    dem_by_event: dict[str, np.ndarray] = {}
    if use_dem:
        b1_catalog_for_extents = pd.read_csv(b1.CATALOG_PATH, low_memory=False)
        extents_by_event = {
            str(row["id"]): _get_event_extent(b1_catalog_for_extents, str(row["id"]))
            for _, row in sampled.iterrows()
        }
        dem_by_event = b1.prefetch_dem_for_extents(
            extents_by_event, max_workers=min(3, len(extents_by_event)),
        )

    payloads = [
        {
            "event_id": str(row["id"]),
            "lifecycle_class": str(row["lifecycle_class"]),
            "b1_module_path": b1.__file__,
            "b1_catalog_path": b1.CATALOG_PATH,
            "n_list": n_list,
            "sigma_cell": sigma_cell,
            "tolerance_px": tolerance_px,
            "margin_px": margin_px,
            "split_merged": split_merged,
            "n_max_sweep": n_max_sweep,
            "use_dem": use_dem,
            "dem_norm": dem_by_event.get(str(row["id"])) if use_dem else None,
        }
        for _, row in sampled.iterrows()
    ]

    all_rows: list[dict] = []
    resolved_workers = max(1, min(n_workers or (os.cpu_count() or 1), len(payloads)))

    if resolved_workers == 1:
        log.info("Running per-class sweep sequentially (n_workers=1) ...")
        b1_catalog = pd.read_csv(b1.CATALOG_PATH, low_memory=False)
        for payload in tqdm(payloads, desc="Classes"):
            rows = evaluate_event(
                b1=b1,
                event_id=payload["event_id"],
                lifecycle_class=payload["lifecycle_class"],
                catalog=b1_catalog,
                n_list=payload["n_list"],
                sigma_cell=payload["sigma_cell"],
                tolerance_px=payload["tolerance_px"],
                margin_px=payload["margin_px"],
                split_merged=payload["split_merged"],
                n_max_sweep=payload["n_max_sweep"],
                use_dem=payload["use_dem"],
                dem_norm=payload.get("dem_norm"),
            )
            all_rows.extend(rows)
    else:
        log.info(f"Running per-class sweep across {resolved_workers} worker processes "
                 f"({len(payloads)} classes, each independent) ...")
        with ProcessPoolExecutor(max_workers=resolved_workers) as pool:
            futures = [pool.submit(_class_event_worker, p) for p in payloads]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Classes"):
                out = fut.result()
                all_rows.extend(out["rows"])
                if out["rows"]:
                    log.info(f"  done: {out['event_id']} [{out['lifecycle_class']}]  "
                             f"{len(out['rows'])} N values")

    results_df = pd.DataFrame(all_rows)

    if make_figures and not results_df.empty:
        log.info("Generating per-class diagnostic figures (--make_figures) ...")
        os.makedirs(out_dir, exist_ok=True)
        b1_catalog = pd.read_csv(b1.CATALOG_PATH, low_memory=False)
        for _, row in sampled.iterrows():
            event_id, lifecycle_class = str(row["id"]), str(row["lifecycle_class"])
            data, extent = b1.load_channels(event_id, b1_catalog)
            if "vil" not in data:
                continue
            vil_phys     = b1.decode_vil_physical(data["vil"][0])
            fused_frame0 = b1.build_fused_cube_frame0(data)
            # Reuse the up-front prefetch result if we have it; falls back
            # to a (disk-cache-backed) inline fetch only if this event
            # wasn't in the prefetched set for some reason.
            dem_norm = dem_by_event.get(event_id) if use_dem else None
            if use_dem and dem_norm is None:
                dem_norm = b1.fetch_dem_norm(extent)
            vil_smooth, level_data = b1.build_all_threshold_cells(
                vil_phys, sigma_cell=sigma_cell, n_max_sweep=n_max_sweep,
                split_merged=split_merged,
            )
            b1.visualize_event(
                event_id=f"{lifecycle_class}_{event_id}",
                vil_phys=vil_phys, vil_smooth=vil_smooth, level_data=level_data,
                fused_frame0=fused_frame0, dem_norm=dem_norm, n_list=n_list,
                sigma_cell=sigma_cell, tolerance_px=tolerance_px,
                split_merged=split_merged, margin_px=margin_px, out_dir=out_dir,
            )

    return results_df


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per class: N_F1 (soft-argmax of F1_nested, upper bound), N_IoU
    (soft-argmax of mIoU_weighted @ tau_high, lower bound), peak values,
    and the Step B5 validation-interval check against N_STAR_BY_CLASS.
    """
    if results_df.empty:
        return pd.DataFrame()

    summary_rows = []
    for cls, g in results_df.groupby("lifecycle_class"):
        g = g.sort_values("N_requested")
        n_list  = g["N_requested"].tolist()
        f1_vals = g["f1_nested"].tolist()
        iou_vals = g["miou_weighted_high"].tolist()

        # b1.soft_argmax_n is a free function, imported lazily by the caller
        # (see print_summary_table / main below) to avoid importing b1 here.
        n_f1  = _soft_argmax_n(f1_vals, n_list)
        n_iou = _soft_argmax_n(iou_vals, n_list)
        n_star = N_STAR_BY_CLASS.get(cls)

        interval_ok = (
            n_star is not None and n_f1 is not None and n_iou is not None
            and n_iou <= n_star <= n_f1
        )

        summary_rows.append({
            "lifecycle_class": cls,
            "event_id": g["event_id"].iloc[0],
            "N_F1": n_f1,
            "F1_nested_peak": max(f1_vals) if f1_vals else 0.0,
            "N_IoU": n_iou,
            "mIoU_weighted_high_peak": max(iou_vals) if iou_vals else 0.0,
            "N_star": n_star,
            "interval_ok": interval_ok,
            "any_zero_cell": bool(g["zero_cell"].any()),
        })

    return pd.DataFrame(summary_rows)


def _soft_argmax_n(values: list[float], n_list: list[int], threshold: float = 0.95):
    """Local copy of b1_diagnostic.soft_argmax_n's logic so build_summary()
    doesn't need a module reference -- kept in exact sync with the
    original; see b1_diagnostic.soft_argmax_n's docstring for the spec
    citation (N_soft = min{N : value(N) >= threshold * max value})."""
    if not values or all(v < 1e-6 for v in values):
        return None
    peak = max(values)
    cutoff = threshold * peak
    for n, v in zip(n_list, values):
        if v >= cutoff:
            return n
    return n_list[int(np.argmax(values))]


def print_summary_table(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("  B1 Per-Class Summary: single-event N_F1 / N_IoU vs N* (Step B5 validation interval)")
    print("=" * 100)
    header = (f"{'Class':<18}  {'Event':<14}  {'N_IoU':>7}  {'N*':>6}  {'N_F1':>7}  "
              f"{'ok?':>5}  {'F1_peak':>8}  {'mIoU_peak':>10}  {'0-cell':>7}")
    print(header)
    print("-" * 100)
    for _, r in summary_df.sort_values("lifecycle_class").iterrows():
        print(
            f"{r['lifecycle_class']:<18}  {str(r['event_id']):<14}  "
            f"{str(r['N_IoU']):>7}  {str(r['N_star']):>6}  {str(r['N_F1']):>7}  "
            f"{'OK' if r['interval_ok'] else 'CHECK':>5}  "
            f"{r['F1_nested_peak']:>8.3f}  {r['mIoU_weighted_high_peak']:>10.3f}  "
            f"{'yes' if r['any_zero_cell'] else 'no':>7}"
        )
    n_ok = summary_df["interval_ok"].sum()
    print("-" * 100)
    print(f"  Validation interval holds for {n_ok}/{len(summary_df)} classes.")
    print("  NOTE: N_star is currently the same placeholder value for every class "
          "(see N_STAR_BY_CLASS docstring) -- replace with Table 5.7 before treating "
          "'CHECK' rows as a real per-class failure.")
    print("=" * 100 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def _parse_event_overrides(pairs: Optional[list[str]]) -> dict[str, str]:
    overrides = {}
    for pair in pairs or []:
        if "=" not in pair:
            log.warning(f"--event_for_class expects CLASS=EVENT_ID, got: {pair!r} (skipped)")
            continue
        cls, event_id = pair.split("=", 1)
        overrides[cls] = event_id
    return overrides


def main():
    parser = argparse.ArgumentParser(
        description="Block B1: single-event-per-class CC-vs-SLIC comparison "
                     "(wraps b1_diagnostic.py; class stratification mirrors "
                     "block_a_solver_benchmark.py's A.6 sampling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--b1_module", default=B1_MODULE_PATH,
                         help=f"Path to b1_diagnostic.py (default: {B1_MODULE_PATH})")
    parser.add_argument("--event_catalogue", default=EVENT_CATALOGUE_PATH,
                         help="Path to event_catalogue.csv (id, lifecycle_class)")
    parser.add_argument("--classes", nargs="+", default=None,
                         help="Subset of lifecycle classes to run (default: all present)")
    parser.add_argument("--n_values", type=int, nargs="+", default=DEFAULT_N_VALUES,
                         help=f"N sweep values (default: {DEFAULT_N_VALUES})")
    parser.add_argument("--sigma_cell", type=float, default=1.0)
    parser.add_argument("--tolerance", type=int, default=None,
                         help="delta in px (default: b1_diagnostic.DEFAULT_TOLERANCE_PX = 0)")
    parser.add_argument("--margin_px", type=int, default=None,
                         help="delta_margin in px (default: b1_diagnostic.DEFAULT_MARGIN_PX)")
    parser.add_argument("--split_merged", action="store_true")
    parser.add_argument("--n_max_sweep", type=int, default=None)
    parser.add_argument("--use_dem", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--event_for_class", nargs="+", default=None,
                         help="Override the sampled event for specific classes, "
                              "e.g. --event_for_class STEADY=S123456 QUIESCENT=S654321")
    parser.add_argument("--workers", type=int, default=None,
                         help="Worker processes across classes (default: "
                              "min(cpu_count, n_classes); pass 1 for sequential)")
    parser.add_argument("--make_figures", action="store_true",
                         help="Also render b1_diagnostic's per-event diagnostic figures")
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--out", default=OUTPUT_CSV)
    parser.add_argument("--summary", default=SUMMARY_CSV)
    args = parser.parse_args()

    b1 = _load_b1_module(args.b1_module)
    overrides = _parse_event_overrides(args.event_for_class)

    results_df = run_per_class(
        b1=b1,
        event_catalogue_path=args.event_catalogue,
        classes=args.classes,
        n_list=args.n_values,
        sigma_cell=args.sigma_cell,
        tolerance_px=args.tolerance,
        margin_px=args.margin_px,
        split_merged=args.split_merged,
        n_max_sweep=args.n_max_sweep,
        use_dem=args.use_dem,
        seed=args.seed,
        overrides=overrides,
        n_workers=args.workers,
        make_figures=args.make_figures,
        out_dir=args.out_dir,
    )

    if results_df.empty:
        log.error("No results produced -- check catalogue paths and HDF5 files.")
        sys.exit(1)

    results_df.to_csv(args.out, index=False)
    log.info(f"Raw per-N results saved -> {args.out}  ({len(results_df):,} rows)")

    summary_df = build_summary(results_df)
    summary_df.to_csv(args.summary, index=False)
    log.info(f"Per-class summary saved -> {args.summary}")

    print_summary_table(summary_df)
    log.info("B1 per-class comparison complete.")


if __name__ == "__main__":
    main()
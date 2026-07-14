"""
test_block_a_single_event.py
=============================
Smoke test for block_a_solver_benchmark.py: runs the FULL Block A pipeline
(DEM + 4-channel SLIC graph -> topological adjacency -> c_sigma/D/sigma_init
calibration -> IMEX Strang vs DOPRI5) on ONE event per lifecycle class, and
renders diagnostic plots so you can eyeball whether each stage is doing
something sensible before committing to the full N_EVENTS_PER_CLASS x
N_VALUES x N_SEEDS sweep.

It imports block_a_solver_benchmark.py directly and calls its real
functions (build_graph_topological, calibrate_c_sigma, calibrate_D,
calibrate_class_sigma, imex_strang_integrate, dopri5_integrate, ...) —
this is an integration test of the actual pipeline, not a re-implementation.

Two modes
---------
--synthetic (default)
    Generates one synthetic SEVIR-like (VIL, IR107, IR069) event per
    lifecycle class, plus a synthetic hill-and-valley DEM, entirely
    in-memory. No HDF5/catalog files or network access needed — this is
    the mode to use to check "is the code working at all".

--real
    Loads real events from CATALOGUE_PATH / the SEVIR CATALOG.csv (one
    per lifecycle class — the first available), and uses the real
    prefetch_dem_for_events() -> Planetary Computer DEM pipeline (disk
    cache keyed by EXTENT, not event_id, so classes that happen to share
    a bounding box reuse the same fetched tile). Use this once you have
    the actual data mounted.

Usage
-----
  python test_block_a_single_event.py                    # synthetic, fast
  python test_block_a_single_event.py --n 300             # bigger test graph
  python test_block_a_single_event.py --real               # real SEVIR data
  python test_block_a_single_event.py --real --n 500

Output
------
  block_a_test/<CLASS>_diagnostics.png   — per-class 6-panel diagnostic figure
  block_a_test/summary.png               — cross-class NFE/lambda_max/convergence
  block_a_test/summary.csv               — one row per class
"""
from __future__ import annotations
import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_a_solver_benchmark as ba  # the module under test

if not ba.JAX_OK:
    sys.exit("JAX/Diffrax not available — cannot run the smoke test.")

import jax
import jax.numpy as jnp

LIFECYCLE_CLASSES = list(ba.TARGET_RHO.keys())   # the 7 classes, in TARGET_RHO order
OUT_DIR_DEFAULT   = "block_a_test"


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA  (only used in --synthetic mode; mirrors the style of
# generate_synthetic_event() in visualize_hierarchical_graph.py, but
# parameterised per lifecycle class so the 7 test events are distinguishable)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_event(lifecycle_class: str, seed: int,
                              H: int = 160, W: int = 160, T: int = 13):
    """
    Build a synthetic multi-cell VIL/IR107/IR069 event whose intensity
    trajectory loosely reflects the named lifecycle class (this only
    matters for the Preliminary normalisation range — Block A's graph is
    built from frame 0 only). Returns RAW (un-normalised) (T,H,W) arrays.
    """
    rng = np.random.default_rng(seed)

    def gauss2d(cy, cx, sigma):
        y, x = np.ogrid[:H, :W]
        return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))

    # life(t) in [0,1] intensity envelope, shaped per class
    t_frac = np.linspace(0.0, 1.0, T)
    if lifecycle_class == "RAPID_GROWTH":
        envelope = t_frac ** 0.5
    elif lifecycle_class == "GROWTH_DECAY":
        envelope = np.sin(np.pi * t_frac)
    elif lifecycle_class == "EPISODIC":
        envelope = 0.5 + 0.5 * np.sin(4 * np.pi * t_frac)
    elif lifecycle_class == "PLATEAU":
        envelope = np.clip(t_frac * 3, 0, 1) * np.clip((1 - t_frac) * 3, 0, 1) + 0.3
        envelope = np.clip(envelope, 0, 1)
    elif lifecycle_class == "RAPID_DECAY":
        envelope = (1 - t_frac) ** 0.5
    elif lifecycle_class == "STEADY":
        envelope = np.full(T, 0.6) + rng.normal(0, 0.03, T)
    else:  # QUIESCENT
        envelope = np.full(T, 0.15) + rng.normal(0, 0.02, T)
    envelope = np.clip(envelope, 0.05, 1.0)

    cells = [
        [H * 0.35, W * 0.30, H * 0.10, 220, -0.4, 0.8],
        [H * 0.55, W * 0.65, H * 0.06, 180, -0.2, 1.1],
        [H * 0.75, W * 0.40, H * 0.05, 150, -0.5, 0.5],
    ]

    vil   = np.zeros((T, H, W), dtype=np.float32)
    ir107 = np.zeros((T, H, W), dtype=np.float32)
    ir069 = np.zeros((T, H, W), dtype=np.float32)
    base_ir107, base_ir069 = 250.0, 240.0

    for t in range(T):
        for cy0, cx0, sig, peak, ddy, ddx in cells:
            cy = float(np.clip(cy0 + ddy * t, 0, H - 1))
            cx = float(np.clip(cx0 + ddx * t, 0, W - 1))
            g = gauss2d(cy, cx, sig)
            vil[t] += g * peak * envelope[t] + rng.normal(0, 3, (H, W))
            ir_off = -50 * envelope[t] * g
            ir107[t] += ir_off
            ir069[t] += ir_off * 0.85
        vil[t]   = np.clip(vil[t], 0, 255)
        ir107[t] = base_ir107 + ir107[t] + rng.normal(0, 1.5, (H, W))
        ir069[t] = base_ir069 + ir069[t] + rng.normal(0, 1.5, (H, W))

    return {"vil": vil, "ir069": ir069, "ir107": ir107}


def generate_synthetic_dem(H: int = 160, W: int = 160, seed: int = 7) -> np.ndarray:
    """Smooth hill-and-valley terrain (metres), just for a non-trivial DEM
    channel to look at — not meant to be physically realistic."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W]
    dem = np.zeros((H, W), dtype=np.float32)
    for _ in range(4):
        cy, cx = rng.uniform(0, H), rng.uniform(0, W)
        sigma  = rng.uniform(H * 0.15, H * 0.35)
        amp    = rng.uniform(80, 300) * rng.choice([-1, 1])
        dem += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2))
    dem += 400.0  # baseline elevation
    return dem.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PER-EVENT PIPELINE RUN  (calls the real block_a_solver_benchmark functions)
# ─────────────────────────────────────────────────────────────────────────────

def run_one_event(event_id: str, lifecycle_class: str, channels: dict,
                   dem_norm: np.ndarray, N: int, c_sigma: float,
                   D: float, sigma_init: float, seed: int = 0,
                   n_macro_steps: int = ba.N_MACRO_STEPS,
                   rtol: float = ba.RTOL_DEFAULT,
                   atol: float = ba.ATOL_DEFAULT,
                   cape_norm: Optional[np.ndarray] = None,
                   landtype_grid: Optional[np.ndarray] = None) -> dict:
    
    # Updated to pass environmental grids through
    graph = ba.build_graph_topological(channels, dem_norm, N=N, c_sigma=c_sigma,
                                       cape_norm=cape_norm, landtype_grid=landtype_grid)
    if graph is None:
        raise RuntimeError(f"{event_id}: graph build failed")

    env_jax = jnp.array(graph["env"])
    h0_jax  = jnp.array(graph["h0"])

    weights  = ba.make_weights(sigma_init=sigma_init, seed=seed)
    diff_rhs = ba.make_diffusion_rhs(graph["senders"], graph["receivers"],
                                      graph["edge_weights"], D)
    rxn_rhs  = ba.make_reaction_rhs(env_jax, weights)

    rho = ba.measure_reaction_spectral_radius(weights, h0_jax, env_jax)
    # DIAGNOSTIC: is rho (measured at h0 only) still representative once h
    # moves away from h0? See assess_reaction_stability_away_from_h0's
    # docstring — SiLU's non-saturating growth means it may not be.
    rho_profile = ba.assess_reaction_stability_away_from_h0(weights, h0_jax, env_jax)

    # Wall-clock timing. IMPORTANT: JAX JIT-compiles imex_strang_integrate/
    # dopri5_integrate on first call for a given array-shape signature (N_act,
    # edge count, ...), so a single raw timing conflates one-time compile
    # cost with actual solve cost -- exactly why NFE, not wall-clock, has
    # been the primary metric throughout this pipeline (NFE is compile- and
    # hardware-independent; wall-clock isn't). Record BOTH: the raw first-
    # call time (what a user actually experiences once per shape) and a
    # second "warm" re-run of the identical call (steady-state solve time,
    # with compilation already cached) so both numbers are available and
    # neither is silently conflated with the other.
    t0 = time.perf_counter()
    h_imex, nfe_diff, nfe_rxn, rej_diff, rej_rxn, ok_imex = \
        ba.imex_strang_integrate(h0_jax, diff_rhs, rxn_rhs, n_steps=n_macro_steps,
                                  rtol=rtol, atol=atol,
                                  event_id=event_id, lifecycle_class=lifecycle_class)
    jax.block_until_ready(h_imex)   # ensure the timer captures the actual compute,
    wall_imex_raw = time.perf_counter() - t0   # not just async dispatch return

    t0 = time.perf_counter()
    h_dop, nfe_dop, rej_dop, ok_dop = \
        ba.dopri5_integrate(h0_jax, diff_rhs, rxn_rhs, n_steps=n_macro_steps,
                             rtol=rtol, atol=atol,
                             event_id=event_id, lifecycle_class=lifecycle_class)
    jax.block_until_ready(h_dop)
    wall_dopri5_raw = time.perf_counter() - t0

    # Warm re-run: same shapes/args, so this hits JAX's compilation cache --
    # isolates steady-state per-solve cost from the one-time compile cost
    # baked into the numbers above.
    t0 = time.perf_counter()
    h_imex_warm, *_ = ba.imex_strang_integrate(
        h0_jax, diff_rhs, rxn_rhs, n_steps=n_macro_steps, rtol=rtol, atol=atol,
        event_id=event_id, lifecycle_class=lifecycle_class,
    )
    jax.block_until_ready(h_imex_warm)
    wall_imex_warm = time.perf_counter() - t0

    t0 = time.perf_counter()
    h_dop_warm, *_ = ba.dopri5_integrate(
        h0_jax, diff_rhs, rxn_rhs, n_steps=n_macro_steps, rtol=rtol, atol=atol,
        event_id=event_id, lifecycle_class=lifecycle_class,
    )
    jax.block_until_ready(h_dop_warm)
    wall_dopri5_warm = time.perf_counter() - t0

    l2_err = float(jnp.linalg.norm(h_imex - h_dop)) / (float(jnp.linalg.norm(h_dop)) + 1e-12)

    # DIAGNOSTIC (reviewer suggestion #1): "max_steps_reached with lots of
    # rejections" is ambiguous between genuine stiffness (Kvaerno5/DOPRI5's
    # controller correctly using tiny steps on a bounded-but-hard problem —
    # more steps or a looser tolerance fixes it) and a real finite-time
    # divergence (the state is actually blowing up — more steps just delays
    # the same failure). Only worth the extra cost when something actually
    # failed, and only on the sub-solver that failed (reaction is the classic
    # Kvaerno5-Newton culprit for IMEX; the unsplit RHS for DOPRI5).
    divergence_diag: dict = {}
    if not ok_imex:
        divergence_diag["reaction"] = ba.diagnose_stiffness_vs_divergence(
            h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol, n_steps=n_macro_steps,
            kind="reaction", event_id=event_id, lifecycle_class=lifecycle_class,
        )
    if not ok_dop:
        divergence_diag["dopri5"] = ba.diagnose_stiffness_vs_divergence(
            h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol, n_steps=n_macro_steps,
            kind="dopri5", event_id=event_id, lifecycle_class=lifecycle_class,
        )

    result = {
        "event_id": event_id, "lifecycle_class": lifecycle_class,
        "N_requested": N, "N_actual": graph["N_actual"], "E": graph["E"],
        "c_sigma": c_sigma, "sigma_rbf": graph["sigma_rbf"],
        "D": D, "sigma_init": sigma_init, "spectral_radius": rho,
        "rho_at_h0": rho_profile.get("scale_0.0", float("nan")),
        "rho_at_2x_step": rho_profile.get("scale_2.0", float("nan")),
        "lambda_max_topo": graph["lambda_max_topo"],
        "lambda_max_weighted": D * graph["lambda_max_ew"],
        "nfe_diffusion": nfe_diff, "nfe_reaction": nfe_rxn,
        "nfe_imex_total": nfe_diff + nfe_rxn,
        "rejected_diff": rej_diff, "rejected_rxn": rej_rxn,
        "imex_converged": ok_imex,
        "nfe_dopri5": nfe_dop, "rejected_dopri5": rej_dop,
        "dopri5_converged": ok_dop,
        "nfe_ratio": nfe_dop / (nfe_diff + nfe_rxn + 1),
        "solution_l2_error": l2_err,
        # Raw = includes one-time JIT compilation for this array-shape
        # signature; warm = steady-state solve time with compilation
        # already cached. wall_*_raw - wall_*_warm ~= compile overhead.
        "wall_imex_raw_sec":    round(wall_imex_raw, 4),
        "wall_imex_warm_sec":   round(wall_imex_warm, 4),
        "wall_dopri5_raw_sec":  round(wall_dopri5_raw, 4),
        "wall_dopri5_warm_sec": round(wall_dopri5_warm, 4),
        "wall_speedup_warm":    round(wall_dopri5_warm / max(wall_imex_warm, 1e-9), 4),
        "h_imex_finite": bool(np.all(np.isfinite(np.asarray(h_imex)))),
        "h_dop_finite": bool(np.all(np.isfinite(np.asarray(h_dop)))),
        "imex_failure_verdict":   divergence_diag.get("reaction", {}).get("verdict", ""),
        "dopri5_failure_verdict": divergence_diag.get("dopri5", {}).get("verdict", ""),
    }
    return result, graph, np.asarray(h0_jax), np.asarray(h_imex), np.asarray(h_dop), divergence_diag


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["VIL", "IR107", "IR069"]


def plot_event_diagnostics(event_id, cls, channels, dem_raw, dem_norm,
                            dem_weighted, graph, h0, h_imex, h_dop, result,
                            out_dir, divergence_diag: dict | None = None):
    divergence_diag = divergence_diag or {}
    fig, axes = plt.subplots(2, 4, figsize=(21, 10))
    fig.suptitle(f"Block A pipeline check — {cls}  (event {event_id})",
                 fontsize=13, fontweight="bold")

    vil0 = channels["vil"][0]
    H, W = vil0.shape

    # ── (0,0) VIL frame 0 with SLIC boundaries + topological edges ─────────
    ax = axes[0, 0]
    ax.imshow(vil0, cmap="turbo", origin="upper")
    pos = graph["positions"]
    s, r = graph["senders"], graph["receivers"]
    seen = set()
    for i in range(len(s)):
        key = tuple(sorted((int(s[i]), int(r[i]))))
        if key in seen:
            continue
        seen.add(key)
        ax.plot([pos[s[i], 0], pos[r[i], 0]], [pos[s[i], 1], pos[r[i], 1]],
                color="yellow", linewidth=0.4, alpha=0.6)
    ax.scatter(pos[:, 0], pos[:, 1], c="red", s=4, zorder=5)
    ax.set_title(f"VIL(t=0) + topological graph\n"
                 f"N_actual={graph['N_actual']}  E={graph['E']}")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

    # ── (0,1) DEM raw / normalised / weighted ───────────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(dem_raw, cmap="terrain", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046, label="metres")
    ax.set_title("DEM (raw)")

    ax = axes[0, 2]
    im = ax.imshow(dem_weighted, cmap="terrain", origin="upper")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(f"DEM weighted (dem_norm x lambda_z={ba.LAMBDA_Z})")

    # ── (1,0) node-state histograms ──────────────────────────────────────────
    ax = axes[1, 0]
    for c in range(3):
        ax.hist(h0[:, c], bins=25, alpha=0.55, label=CHANNEL_NAMES[c])
    ax.set_title("Node states h0 (per-superpixel means)")
    ax.set_xlabel("normalised value [0,1]")
    ax.legend(fontsize=8)

    # ── (1,1) Integration step diagnostic — accepted vs. rejected steps ─────
    ax = axes[1, 1]
    stage_labels = ["Tsit5\n(diffusion)", "Kvaerno5\n(reaction)", "DOPRI5\n(unsplit)"]

    accepted = [
        result["nfe_diffusion"] / ba.STAGE_COUNT_TSIT5,
        result["nfe_reaction"] / ba.STAGE_COUNT_KVAERNO5,
        result["nfe_dopri5"] / ba.STAGE_COUNT_DOPRI5,
    ]
    rejected = [
        result["rejected_diff"],
        result["rejected_rxn"],
        result["rejected_dopri5"],
    ]

    xpos  = np.arange(len(stage_labels))
    width = 0.35
    acc_plot = [max(v, 0.5) for v in accepted]
    rej_plot = [max(v, 0.5) for v in rejected]

    bars_acc = ax.bar(xpos - width / 2, acc_plot, width, color="#55A868", label="Accepted steps")
    bars_rej = ax.bar(xpos + width / 2, rej_plot, width, color="#C44E52", label="Rejected steps")
    for b, v in zip(bars_acc, accepted):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(round(v)):,}",
                ha="center", va="bottom", fontsize=7.5)
    for b, v in zip(bars_rej, rejected):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{int(round(v)):,}",
                ha="center", va="bottom", fontsize=7.5,
                color="#C44E52" if v > 0 else "black", fontweight="bold" if v > 0 else "normal")

    ax.set_xticks(xpos)
    ax.set_xticklabels(stage_labels)
    ax.set_yscale("log")
    ax.set_ylabel("step count (log scale)")

    rej_rates = []
    for a, rj in zip(accepted, rejected):
        total = a + rj
        rej_rates.append(100.0 * rj / total if total > 0 else 0.0)
    worst_i = int(np.argmax(rej_rates))
    ax.set_title(
        "Integration step diagnostic (accepted vs. rejected)\n"
        f"worst: {stage_labels[worst_i].splitlines()[0]} "
        f"{rej_rates[worst_i]:.1f}% steps rejected"
    )
    ax.legend(fontsize=8, loc="upper left")

    # ── (1,2) NFE breakdown ──────────────────────────────────────────────────
    ax = axes[1, 2]
    labels = ["NFE_diff\n(IMEX)", "NFE_rxn\n(IMEX)", "NFE_total\n(IMEX)", "NFE\n(DOPRI5)"]
    values = [result["nfe_diffusion"], result["nfe_reaction"],
              result["nfe_imex_total"], result["nfe_dopri5"]]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    bars = ax.bar(labels, values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_title(f"NFE  (ratio DOPRI5/IMEX = {result['nfe_ratio']:.2f}x)\n"
                 f"imex_ok={result['imex_converged']}  dop_ok={result['dopri5_converged']}")

    # ── (0,3) text summary of the divergence-vs-stiffness verdict ───────────
    ax = axes[0, 3]
    ax.axis("off")
    lines = [f"Divergence-vs-stiffness triage  ({cls})", ""]
    if not divergence_diag:
        lines.append("Both solvers converged — nothing to triage.")
    else:
        for kind, diag in divergence_diag.items():
            lines.append(f"[{kind}]  verdict = {diag['verdict']}")
            if np.isfinite(diag.get("h_growth", float('nan'))):
                lines.append(f"  max|h| growth:  {diag['h_growth']:.2f}x")
                lines.append(f"  accepted dt shrink: {diag['dt_shrink']:.2e}x")
                lines.append(f"  dt-vs-step trend corr: {diag['dt_trend_corr']:.2f}")
            lines.append(f"  accepted/rejected steps seen: "
                         f"{diag['n_accepted_total']}/{diag.get('n_rejected_total', '?')}")
            lines.append("")
        lines.append("DIVERGENCE -> raising max_steps won't help;")
        lines.append("  the reaction/DOPRI5 output itself needs bounding.")
        lines.append("STIFFNESS  -> a smaller step budget or tighter")
        lines.append("  tolerance is the correct, sufficient fix.")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes,
            fontsize=8.3, va="top", ha="left", family="monospace")

    # ── (1,3) max|h| and accepted dt over the failing sub-solver's steps ────
    ax = axes[1, 3]
    if not divergence_diag:
        ax.axis("off")
        ax.text(0.5, 0.5, "no failing sub-solver\nto diagnose", ha="center",
                va="center", transform=ax.transAxes, fontsize=10, color="gray")
    else:
        kind = max(divergence_diag, key=lambda k: divergence_diag[k]["n_accepted_total"])
        diag = divergence_diag[kind]
        n = diag["n_accepted_total"]
        if n >= 2:
            step_idx = np.arange(n)
            l1, = ax.plot(step_idx, diag["h_max_history"], color="#C44E52", label="max|h|")
            ax.set_yscale("log")
            ax.set_xlabel("accepted step #")
            ax.set_ylabel("max|h|  (log)", color="#C44E52")
            ax.tick_params(axis="y", labelcolor="#C44E52")

            ax2 = ax.twinx()
            l2, = ax2.plot(step_idx, diag["dt_history"], color="#4C72B0", label="accepted dt")
            ax2.set_yscale("log")
            ax2.set_ylabel("accepted dt  (log)", color="#4C72B0")
            ax2.tick_params(axis="y", labelcolor="#4C72B0")

            ax.legend(handles=[l1, l2], fontsize=7.5, loc="upper left")
            ax.set_title(f"[{kind}] {diag['verdict']}\nmax|h| vs. accepted dt per step")
        else:
            ax.axis("off")
            ax.text(0.5, 0.5, f"[{kind}]\ntoo few accepted steps\nto plot a trend "
                    f"({n})", ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(out_dir, f"{cls}_diagnostics.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL WORKER
# ─────────────────────────────────────────────────────────────────────────────

def _run_and_plot_one_class(payload: dict) -> dict:
    ba.MAX_STEPS_EXPLICIT = payload["max_steps"]
    ba.MAX_STEPS_IMPLICIT = payload["max_steps"]
    ba.MAX_STEPS_DOPRI5   = payload["max_steps_dopri5"]

    t0 = time.perf_counter()
    result, graph, h0, h_imex, h_dop, divergence_diag = run_one_event(
        payload["event_id"], payload["cls"], payload["channels"],
        payload["dem_norm"], N=payload["N_TEST"], c_sigma=payload["c_sigma"],
        D=payload["D"], sigma_init=payload["sigma_init"],
        seed=payload["seed"], n_macro_steps=payload["macro_steps"],
        rtol=payload.get("rtol", ba.RTOL_DEFAULT),
        atol=payload.get("atol", ba.ATOL_DEFAULT),
        cape_norm=payload.get("cape_norm"),             # Added
        landtype_grid=payload.get("landtype_grid"),     # Added
    )

    dem_w = payload["dem_norm"] * ba.LAMBDA_Z
    plot_path = plot_event_diagnostics(
        payload["event_id"], payload["cls"], payload["channels"],
        payload["dem_raw"], payload["dem_norm"], dem_w, graph, h0,
        h_imex, h_dop, result, payload["out_dir"], divergence_diag,
    )
    wall_s = time.perf_counter() - t0
    return {"cls": payload["cls"], "result": result, "plot_path": plot_path,
            "wall_s": wall_s}


def plot_summary(df: pd.DataFrame, out_dir: str) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    classes = df["lifecycle_class"].tolist()

    ax = axes[0]
    ax.bar(classes, df["nfe_ratio"], color="#4C72B0")
    ax.set_title("NFE ratio (DOPRI5 / IMEX) per class")
    ax.set_ylabel("x speedup of IMEX")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    ax = axes[1]
    width = 0.35
    xpos = np.arange(len(classes))
    ax.bar(xpos - width/2, df["lambda_max_topo"], width, label="lambda_max_topo")
    ax.bar(xpos + width/2, df["lambda_max_weighted"], width, label="lambda_max (D-scaled)")
    ax.set_xticks(xpos); ax.set_xticklabels(classes, rotation=45)
    ax.set_title("Diffusion stiffness diagnostics")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.bar(xpos - width/2, df["spectral_radius"], width, label="rho_measured")
    target = [ba.TARGET_RHO.get(c, np.nan) for c in classes]
    ax.bar(xpos + width/2, target, width, label="rho_target")
    ax.set_xticks(xpos); ax.set_xticklabels(classes, rotation=45)
    ax.set_yscale("log")
    ax.set_title("Reaction stiffness calibration (log scale)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "summary.png")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# SANITY CHECKS  (printed pass/fail, not asserted — this is a smoke test)
# ─────────────────────────────────────────────────────────────────────────────

def print_sanity_checks(df: pd.DataFrame, N_requested: int):
    print("\n" + "=" * 78)
    print("  SANITY CHECKS")
    print("=" * 78)
    checks = [
        ("N_actual within 20% of N_requested",
         (df["N_actual"] - N_requested).abs().max() <= 0.2 * N_requested),
        ("All events have E > 0 (nonempty topological graph)", (df["E"] > 0).all()),
        ("lambda_max_topo finite and > 0 for all events",
         np.isfinite(df["lambda_max_topo"]).all() and (df["lambda_max_topo"] > 0).all()),
        ("IMEX solutions are all finite (no NaN/Inf)", df["h_imex_finite"].all()),
        ("DOPRI5 solutions are all finite (no NaN/Inf)", df["h_dop_finite"].all()),
        ("IMEX converged for every class", df["imex_converged"].all()),
        ("Reaction spectral radius roughly tracks target (within 10x)",
         (df["spectral_radius"] / df["lifecycle_class"].map(ba.TARGET_RHO))
         .between(0.1, 10).all()),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'CHECK'}] {label}")
    print("=" * 78 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true",
                        help="Use real SEVIR data instead of synthetic events")
    parser.add_argument("--n", type=int, default=60,
                        help="Test N (superpixel count) — kept small for a fast smoke test")
    parser.add_argument("--macro-steps", type=int, default=2,
                        help="Macro steps per integration (production default is "
                             f"{ba.N_MACRO_STEPS}; kept small here to bound smoke-test runtime "
                             "— on CPU-only JAX, each additional macro step for a stiff class "
                             "can add real wall time, since Kvaerno5's Newton solves aren't free)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max internal solver steps for Tsit5/Kvaerno5 sub-steps "
                             f"(production default is {ba.MAX_STEPS_EXPLICIT}/{ba.MAX_STEPS_IMPLICIT}; "
                             "raise this — and expect much longer runtime — once you move past "
                             "the smoke test)")
    parser.add_argument("--max-steps-dopri5", type=int, default=1000,
                        help=f"Max internal DOPRI5 steps (production default is {ba.MAX_STEPS_DOPRI5})")
    parser.add_argument("--out", type=str, default=OUT_DIR_DEFAULT,
                        help="Output directory for plots/CSV")
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel worker PROCESSES for the per-class IMEX-vs-DOPRI5 "
                             "stage (the dominant cost: each class triggers its own fresh "
                             "JAX/XLA JIT compile of Tsit5/Kvaerno5/Dopri5, plus the actual "
                             "solve, plus plotting). 0 (default) = auto, "
                             "min(n_classes, os.cpu_count()). Set to 1 to force sequential "
                             "execution, e.g. for easier debugging/profiling.")
    parser.add_argument("--io-workers", type=int, default=8,
                        help="Parallel THREADS for loading real per-class HDF5 channel data "
                             "(--real mode only, ignored in --synthetic mode; this stage is "
                             "I/O-bound so threads — not processes — are the right tool).")
    args = parser.parse_args()

    # Reduced step budgets purely to keep this smoke test bounded in wall time —
    # NOT representative of the real Block A sweep. Run the full
    # block_a_solver_benchmark.py (or re-run this script with
    # --macro-steps 12 --max-steps 4000) for publishable NFE numbers.
    ba.MAX_STEPS_EXPLICIT = args.max_steps
    ba.MAX_STEPS_IMPLICIT = args.max_steps
    ba.MAX_STEPS_DOPRI5   = args.max_steps_dopri5

    os.makedirs(args.out, exist_ok=True)
    N_TEST = args.n

    event_data: dict[str, dict] = {}   # event_id -> {lifecycle_class, channels, dem_raw, dem_norm, cape, landtype}

    if args.real:
        print("Loading real SEVIR data (one event per class)…")
        catalogue = pd.read_csv(ba.CATALOGUE_PATH, low_memory=False)
        sevir_catalog = pd.read_csv(os.path.join(ba.DATA_ROOT, "CATALOG.csv"), low_memory=False)

        def _select_event_for_class(cls: str):
            rows = catalogue[catalogue["lifecycle_class"] == cls]
            for _, row in rows.iterrows():
                event_id = str(row["id"])
                channels = ba.load_event_multichannel(event_id, sevir_catalog)
                if channels is not None:
                    return cls, event_id, channels
            return cls, None, None

        selected: dict[str, tuple] = {}   # cls -> (event_id, channels)
        n_io_workers = max(1, min(args.io_workers, len(LIFECYCLE_CLASSES)))
        with ThreadPoolExecutor(max_workers=n_io_workers) as pool:
            futures = [pool.submit(_select_event_for_class, cls) for cls in LIFECYCLE_CLASSES]
            for fut in as_completed(futures):
                cls, event_id, channels = fut.result()
                if channels is None:
                    print(f"  [WARN] no usable event found for class {cls}, skipping.")
                    continue
                selected[cls] = (event_id, channels)

        # Stage 2: fetch DEM and Landtype for all selected events up front, in parallel
        channel_shapes = {
            eid: ch["vil"].shape[1:] for eid, ch in selected.values()
        }
        dem_by_event = ba.prefetch_dem_for_events(
            [eid for eid, _ in selected.values()], sevir_catalog, channel_shapes,
        )
        landtype_by_event = ba.prefetch_landtype_for_events(
            [eid for eid, _ in selected.values()], sevir_catalog, channel_shapes,
        )

        for cls in LIFECYCLE_CLASSES:
            if cls not in selected:
                continue
            event_id, channels = selected[cls]
            H_evt, W_evt = channel_shapes[event_id]
            
            dem_norm = dem_by_event.get(
                event_id, np.zeros((H_evt, W_evt), dtype=np.float32)
            )
            cape_norm = ba.load_cape_for_event(event_id, sevir_catalog, H_evt, W_evt)
            landtype_grid = landtype_by_event.get(
                event_id, np.zeros((H_evt, W_evt, 2), dtype=np.float32)
            )
            
            event_data[cls] = {
                "event_id": event_id, "channels": channels,
                "dem_raw": dem_norm, "dem_norm": dem_norm,   # raw already normalised
                "cape_norm": cape_norm,
                "landtype_grid": landtype_grid
            }
    else:
        print("Generating synthetic events (one per class)…")
        for i, cls in enumerate(LIFECYCLE_CLASSES):
            raw_channels = generate_synthetic_event(cls, seed=100 + i)
            channels = {k: ba._normalise_channel_minmax(v) for k, v in raw_channels.items()}
            dem_raw = generate_synthetic_dem(seed=7 + i)
            dem_norm = ba._normalise_channel_minmax(dem_raw)
            event_data[cls] = {
                "event_id": f"SYNTH_{cls}", "channels": channels,
                "dem_raw": dem_raw, "dem_norm": dem_norm,
                "cape_norm": None,         # Will trigger zero fallbacks gracefully in topology builder
                "landtype_grid": None,
            }

    if not event_data:
        sys.exit("No events available — aborting.")

    # ── A.6: c_sigma calibration on the STEADY event (fallback: first event) ─
    steady_key = "STEADY" if "STEADY" in event_data else next(iter(event_data))
    steady = event_data[steady_key]
    print(f"\nCalibrating c_sigma on {steady['event_id']} ({steady_key}) at N={N_TEST}…")
    c_sigma, lambda_max_topo_ref, sweep = ba.calibrate_c_sigma(
        steady["channels"], steady["dem_norm"], N=N_TEST,
    )
    print(f"  c_sigma = {c_sigma}  (lambda_max_topo = {lambda_max_topo_ref:.3f})")

    # ── A.2: D calibration on the same event ──────────────────────────────
    graph_calib = ba.build_graph_topological(steady["channels"], steady["dem_norm"],
                                              N=N_TEST, c_sigma=c_sigma)
    r_stab = ba.estimate_tsit5_stability_radius()
    D = ba.calibrate_D(graph_calib, r_stab)
    print(f"  r_stab(Tsit5) = {r_stab:.4f}   D = {D:.6g}")

    # ── A.2: per-class sigma_init (single calibration event per class here) ─
    print("\nCalibrating sigma_init per class (1 event/class — smoke test)…")
    class_sigma: dict[str, float] = {}

    def _calibrate_one_class(cls: str, d: dict):
        g = ba.build_graph_topological(d["channels"], d["dem_norm"], N=N_TEST, c_sigma=c_sigma,
                                       cape_norm=d.get("cape_norm"), landtype_grid=d.get("landtype_grid"))
        sigma, rho = ba.calibrate_class_sigma(cls, [g], max_iter=8)
        return cls, sigma, rho

    n_calib_workers = max(1, min(args.workers or (os.cpu_count() or 1), len(event_data)))
    if n_calib_workers <= 1:
        for cls, d in event_data.items():
            cls, sigma, rho = _calibrate_one_class(cls, d)
            class_sigma[cls] = sigma
            print(f"  {cls:<15} sigma_init={sigma:.4f}  rho_measured={rho:.3f}  "
                  f"(target={ba.TARGET_RHO.get(cls, float('nan')):.3f})")
    else:
        with ThreadPoolExecutor(max_workers=n_calib_workers) as pool:
            futures = {pool.submit(_calibrate_one_class, cls, d): cls
                       for cls, d in event_data.items()}
            calib_out = {}
            for fut in as_completed(futures):
                cls, sigma, rho = fut.result()
                calib_out[cls] = (sigma, rho)
        for cls in event_data:
            sigma, rho = calib_out[cls]
            class_sigma[cls] = sigma
            print(f"  {cls:<15} sigma_init={sigma:.4f}  rho_measured={rho:.3f}  "
                  f"(target={ba.TARGET_RHO.get(cls, float('nan')):.3f})")

    # ── A.3: tolerance calibration, against the STIFFEST class present ──────
    stiffest_cls = max(class_sigma, key=lambda c: ba.TARGET_RHO.get(c, 0.0))
    print(f"\nCalibrating tolerance (A.3) against {stiffest_cls} "
          f"(stiffest class present, target_rho="
          f"{ba.TARGET_RHO.get(stiffest_cls, float('nan')):.3f})…")
    
    stiff_data = event_data[stiffest_cls]
    g_stiff = ba.build_graph_topological(
        stiff_data["channels"], stiff_data["dem_norm"],
        N=N_TEST, c_sigma=c_sigma,
        cape_norm=stiff_data.get("cape_norm"),
        landtype_grid=stiff_data.get("landtype_grid")
    )
    
    w_stiff = ba.make_weights(sigma_init=class_sigma[stiffest_cls], seed=0)
    diff_rhs_stiff = ba.make_diffusion_rhs(g_stiff["senders"], g_stiff["receivers"],
                                            g_stiff["edge_weights"], D)
    rxn_rhs_stiff = ba.make_reaction_rhs(jnp.array(g_stiff["env"]), w_stiff)
    rtol, atol = ba.calibrate_tolerance(
        jnp.array(g_stiff["h0"]), diff_rhs_stiff, rxn_rhs_stiff, label=stiffest_cls,
    )
    print(f"  rtol={rtol:.2e}  atol={atol:.2e}  "
          f"(spec starting point was rtol=1e-2; module default is "
          f"rtol={ba.RTOL_DEFAULT:.1e})")

    # ── Run the full IMEX-vs-DOPRI5 pipeline for each class ────────────────
    print(f"\nRunning IMEX vs DOPRI5 at N={N_TEST} for each class…")
    n_workers = max(1, min(args.workers or (os.cpu_count() or 1), len(event_data)))

    payloads = [
        {
            "cls": cls, "event_id": d["event_id"], "channels": d["channels"],
            "dem_norm": d["dem_norm"], "dem_raw": d["dem_raw"],
            "N_TEST": N_TEST, "c_sigma": c_sigma, "D": D,
            "sigma_init": class_sigma[cls], "seed": 0,
            "rtol": rtol, "atol": atol,
            "macro_steps": args.macro_steps, "max_steps": args.max_steps,
            "max_steps_dopri5": args.max_steps_dopri5, "out_dir": args.out,
            "cape_norm": d.get("cape_norm"),            # Added
            "landtype_grid": d.get("landtype_grid"),    # Added
        }
        for cls, d in event_data.items()
    ]

    outputs: dict[str, dict] = {}
    if n_workers <= 1:
        print(f"  (workers=1 -> running sequentially)")
        for payload in payloads:
            out = _run_and_plot_one_class(payload)
            outputs[out["cls"]] = out
            print(f"  {out['cls']:<15} N_act={out['result']['N_actual']:>4}  "
                  f"E={out['result']['E']:>5}  "
                  f"NFE(IMEX)={out['result']['nfe_imex_total']:>6}  "
                  f"NFE(DOPRI5)={out['result']['nfe_dopri5']:>7}  "
                  f"ratio={out['result']['nfe_ratio']:.2f}x  "
                  f"ok_imex={out['result']['imex_converged']}  "
                  f"ok_dop={out['result']['dopri5_converged']}  "
                  f"[{out['wall_s']:.1f}s]")
            print(f"    -> {out['plot_path']}")
    else:
        print(f"  (workers={n_workers} processes)")
        ctx = mp.get_context("spawn")   
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_and_plot_one_class, payload): payload["cls"]
                       for payload in payloads}
            for fut in as_completed(futures):
                cls = futures[fut]
                try:
                    out = fut.result()
                except Exception as exc:
                    print(f"  [ERROR] {cls}: worker failed: {exc}")
                    continue
                outputs[out["cls"]] = out
                print(f"  {out['cls']:<15} N_act={out['result']['N_actual']:>4}  "
                      f"E={out['result']['E']:>5}  "
                      f"NFE(IMEX)={out['result']['nfe_imex_total']:>6}  "
                      f"NFE(DOPRI5)={out['result']['nfe_dopri5']:>7}  "
                      f"ratio={out['result']['nfe_ratio']:.2f}x  "
                      f"ok_imex={out['result']['imex_converged']}  "
                      f"ok_dop={out['result']['dopri5_converged']}  "
                      f"[{out['wall_s']:.1f}s]")
                print(f"    -> {out['plot_path']}")

    rows = [outputs[cls]["result"] for cls in event_data if cls in outputs]

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary CSV -> {csv_path}")

    summary_plot = plot_summary(df, args.out)
    print(f"Summary plot -> {summary_plot}")

    print_sanity_checks(df, N_TEST)


if __name__ == "__main__":
    main()
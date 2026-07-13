"""
triage_stiffness_vs_divergence.py
==================================
Answers ONE question, cheaply, before anything else in Block A is touched:

    When IMEX (Kvaerno5 reaction) or DOPRI5 (unsplit) hits max_steps_reached
    with a lot of rejections, is that because the problem is genuinely
    STIFF (bounded, just hard for the controller), or because the state is
    actually DIVERGING toward infinity?

This matters because the two failure modes need different fixes:
  STIFFNESS  -> a smaller step budget / tighter-then-looser tolerance sweep
                (A.3, calibrate_tolerance) is a correct and SUFFICIENT fix.
  DIVERGENCE -> calibrate_tolerance / raising max_steps will NOT fix it —
                more steps just delays the same blow-up. This is what
                triage runs originally found for every one of the 7
                lifecycle classes (dominant reaction-Jacobian eigenvalue at
                h0 was POSITIVE in all of them — a genuinely unstable
                linearisation, not stiffness). block_a_solver_benchmark.py
                now implements the A.2' fix (see _local_reaction): a
                bounded Newtonian-relaxation reaction term,
                dh/dt = k(e)*(h_eq(h,e) - h), with k > 0 (softplus) and
                h_eq bounded in (0,1)^d_m, which is dissipative everywhere
                by construction rather than only locally at h0. Re-running
                this script after that fix should show STIFFNESS (or no
                failures at all) across all 7 classes; a DIVERGENCE verdict
                post-fix would mean k(e) isn't clearing
                _gate_coupling_ceiling() for that class — see the
                margin/[LOW MARGIN] print in calibrate_shared() below.

This script does NOT run the full N x events x seeds sweep, does NOT
produce plots, and does NOT touch the tolerance-selection halving search
unless a class actually comes back STIFFNESS (see --check-tolerance). It
imports the REAL pipeline functions from block_a_solver_benchmark.py
(build_graph_topological, make_weights, imex_strang_integrate,
dopri5_integrate, diagnose_stiffness_vs_divergence, calibrate_tolerance)
and the event-loading helpers already written in
test_block_a_single_event.py — nothing here is a re-implementation.

Usage
-----
  python triage_stiffness_vs_divergence.py                     # synthetic, all 7 classes
  python triage_stiffness_vs_divergence.py --classes RAPID_GROWTH,EPISODIC
  python triage_stiffness_vs_divergence.py --real --classes RAPID_GROWTH
  python triage_stiffness_vs_divergence.py --n 60 --macro-steps 2 --max-steps 300

Output
------
  Printed verdict per class (and per failing sub-solver), plus a final
  one-line answer: STIFFNESS / DIVERGENCE / MIXED / NO FAILURES OBSERVED.
  A flat CSV of the same rows is also written to --out (default:
  block_a_triage/triage.csv) in case you want to paste it into notes.
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import block_a_solver_benchmark as ba          # the real pipeline
import test_block_a_single_event as tbe        # reuse its event-loading helpers

if not ba.JAX_OK:
    sys.exit("JAX/Diffrax not available — cannot run the triage.")

import jax.numpy as jnp

ALL_CLASSES = list(ba.TARGET_RHO.keys())


# ─────────────────────────────────────────────────────────────────────────────
# EVENT LOADING  (thin wrapper around test_block_a_single_event.py's own
# synthetic/real loading branches, restricted to the requested classes)
# ─────────────────────────────────────────────────────────────────────────────

def load_event_data(classes: list[str], use_real: bool, io_workers: int) -> dict[str, dict]:
    event_data: dict[str, dict] = {}

    if use_real:
        print("Loading real SEVIR data (one event per requested class)…")
        catalogue = pd.read_csv(ba.CATALOGUE_PATH, low_memory=False)
        sevir_catalog = pd.read_csv(os.path.join(ba.DATA_ROOT, "CATALOG.csv"), low_memory=False)

        selected: dict[str, tuple] = {}
        for cls in classes:
            rows = catalogue[catalogue["lifecycle_class"] == cls]
            for _, row in rows.iterrows():
                event_id = str(row["id"])
                channels = ba.load_event_multichannel(event_id, sevir_catalog)
                if channels is not None:
                    selected[cls] = (event_id, channels)
                    break
            if cls not in selected:
                print(f"  [WARN] no usable event found for class {cls}, skipping.")

        channel_shapes = {eid: ch["vil"].shape[1:] for eid, ch in selected.values()}
        dem_by_event = ba.prefetch_dem_for_events(
            [eid for eid, _ in selected.values()], sevir_catalog, channel_shapes,
        )
        for cls in classes:
            if cls not in selected:
                continue
            event_id, channels = selected[cls]
            dem_norm = dem_by_event.get(
                event_id, np.zeros(channel_shapes[event_id], dtype=np.float32)
            )
            event_data[cls] = {"event_id": event_id, "channels": channels, "dem_norm": dem_norm}
    else:
        print("Generating synthetic events (one per requested class)…")
        for i, cls in enumerate(classes):
            raw_channels = tbe.generate_synthetic_event(cls, seed=100 + i)
            channels = {k: ba._normalise_channel_minmax(v) for k, v in raw_channels.items()}
            dem_raw = tbe.generate_synthetic_dem(seed=7 + i)
            dem_norm = ba._normalise_channel_minmax(dem_raw)
            event_data[cls] = {"event_id": f"SYNTH_{cls}", "channels": channels, "dem_norm": dem_norm}

    if not event_data:
        sys.exit("No events available for the requested classes — aborting.")
    return event_data


# ─────────────────────────────────────────────────────────────────────────────
# SHARED CALIBRATION  (c_sigma, D, per-class sigma_init) — same procedure as
# test_block_a_single_event.py, no tolerance search here (that's the thing
# under test, not an input to it).
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_shared(event_data: dict, N_TEST: int):
    steady_key = "STEADY" if "STEADY" in event_data else next(iter(event_data))
    steady = event_data[steady_key]
    print(f"Calibrating c_sigma on {steady['event_id']} ({steady_key}) at N={N_TEST}…")
    c_sigma, lambda_max_topo_ref, _sweep = ba.calibrate_c_sigma(
        steady["channels"], steady["dem_norm"], N=N_TEST,
    )

    graph_calib = ba.build_graph_topological(steady["channels"], steady["dem_norm"],
                                              N=N_TEST, c_sigma=c_sigma)
    r_stab = ba.estimate_tsit5_stability_radius()
    D = ba.calibrate_D(graph_calib, r_stab)
    print(f"  c_sigma={c_sigma}  D={D:.6g}")

    # Explicit CFL headroom check AT THE N ACTUALLY BEING TESTED (N_TEST),
    # not just at N* where D was calibrated. Per blocks.tex Sec. A.2/A.6,
    # D is fixed once at N* and "as N grows, the diffusion CFL tightens" —
    # so if N_TEST < N* (as here, N_TEST=950 < N*=1150), this should show
    # MORE headroom than at N*, not less. If it doesn't, that's a concrete,
    # graph-specific finding worth chasing rather than an N-vs-N* mismatch.
    cfl = ba.check_diffusion_cfl(graph_calib, D, r_stab)
    print(f"  [CFL check @ N={N_TEST}] k_max={cfl['k_max']:.4f}  "
          f"lambda_max={cfl['lambda_max']:.4f}  dt_stable={cfl['dt_stable']:.4f}s  "
          f"dt_half_requested={cfl['dt_half_requested']:.1f}s  "
          f"headroom={cfl['headroom_ratio']:.2f}x"
          f"{'  [MARGINAL]' if cfl['marginal'] else ''}")

    class_sigma: dict[str, float] = {}
    print("Calibrating sigma_init per class…")
    for cls, d in event_data.items():
        g = ba.build_graph_topological(d["channels"], d["dem_norm"], N=N_TEST, c_sigma=c_sigma)
        sigma, rho = ba.calibrate_class_sigma(cls, [g], max_iter=8)
        class_sigma[cls] = sigma
        # A.2' check: k(e)'s calibrated magnitude (rho, here) vs. the fixed,
        # h-independent gate-coupling ceiling (see block_a_solver_benchmark.
        # _gate_coupling_ceiling) — QUIESCENT is the binding case since it
        # has the smallest TARGET_RHO and so the least room above zero.
        weights = ba.make_weights(sigma, seed=0)
        ceiling = ba._gate_coupling_ceiling(weights)
        margin = rho / max(ceiling, 1e-12)
        print(f"  {cls:<15} sigma_init={sigma:.4f}  rho_measured={rho:.3f}  "
              f"(target={ba.TARGET_RHO.get(cls, float('nan')):.3f})  "
              f"gate_ceiling={ceiling:.4f}  margin={margin:.1f}x"
              f"{'  [LOW MARGIN]' if margin < 2.0 else ''}")

    return c_sigma, D, class_sigma


# ─────────────────────────────────────────────────────────────────────────────
# PER-CLASS TRIAGE
# ─────────────────────────────────────────────────────────────────────────────

def triage_one_class(cls: str, d: dict, N_TEST: int, c_sigma: float, D: float,
                      sigma_init: float, rtol: float, atol: float,
                      macro_steps: int, seed: int = 0) -> dict:
    graph = ba.build_graph_topological(d["channels"], d["dem_norm"], N=N_TEST, c_sigma=c_sigma)
    if graph is None:
        return {"lifecycle_class": cls, "event_id": d["event_id"], "error": "graph build failed"}

    env_jax = jnp.array(graph["env"])
    h0_jax = jnp.array(graph["h0"])

    weights = ba.make_weights(sigma_init=sigma_init, seed=seed)
    diff_rhs = ba.make_diffusion_rhs(graph["senders"], graph["receivers"], graph["edge_weights"], D)
    rxn_rhs = ba.make_reaction_rhs(env_jax, weights)

    _h_imex, _nfe_d, _nfe_r, _rej_d, _rej_r, ok_imex = ba.imex_strang_integrate(
        h0_jax, diff_rhs, rxn_rhs, n_steps=macro_steps, rtol=rtol, atol=atol,
        event_id=d["event_id"], lifecycle_class=cls,
    )
    _h_dop, _nfe_dop, _rej_dop, ok_dop = ba.dopri5_integrate(
        h0_jax, diff_rhs, rxn_rhs, n_steps=macro_steps, rtol=rtol, atol=atol,
        event_id=d["event_id"], lifecycle_class=cls,
    )

    row = {
        "lifecycle_class": cls, "event_id": d["event_id"],
        "imex_converged": ok_imex, "dopri5_converged": ok_dop,
        "reaction_verdict": "", "dopri5_verdict": "", "diffusion_verdict": "",
        "reaction_h_growth": float("nan"), "reaction_dt_shrink": float("nan"),
        "dopri5_h_growth": float("nan"), "dopri5_dt_shrink": float("nan"),
        "diffusion_h_growth": float("nan"), "diffusion_dt_shrink": float("nan"),
    }

    # Per-EVENT CFL headroom — the shared calibrate_shared() check only
    # covers the calibration graph (STEADY, used to set c_sigma/D); each
    # class's own event has its own SLIC segmentation and therefore its
    # own k_max, which can differ event-to-event even at the same N if one
    # event happens to have an unusually short edge / dense local cluster.
    r_stab = ba.estimate_tsit5_stability_radius()
    cfl = ba.check_diffusion_cfl(graph, D, r_stab)
    row["cfl_headroom_ratio"] = cfl["headroom_ratio"]
    if cfl["marginal"]:
        print(f"  {cls:<15} [CFL] N={N_TEST}  headroom={cfl['headroom_ratio']:.2f}x  "
              f"[MARGINAL — this event's graph may be the reason, not a generic N effect]")

    if not ok_imex:
        # The failures observed are diffusion sub-step failures (macro-step
        # 2's first half-step specifically), not reaction ones — run the
        # diffusion diagnostic, not just reaction, or we're diagnosing the
        # wrong term. Kept alongside the reaction check below since a fast
        # sweep can't yet tell you WHICH sub-step failed without this.
        diag_diff = ba.diagnose_stiffness_vs_divergence(
            h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol, n_steps=macro_steps,
            kind="diffusion", event_id=d["event_id"], lifecycle_class=cls,
        )
        row["diffusion_verdict"] = diag_diff["verdict"]
        row["diffusion_h_growth"] = diag_diff["h_growth"]
        row["diffusion_dt_shrink"] = diag_diff["dt_shrink"]
        # Near-atol-floor check: if the accuracy (not stability) budget is
        # what's actually binding, a large fraction of accepted-step |h|
        # values sitting close to atol is the signature to look for —
        # rtol*|h| becomes negligible there, so atol alone sets the local
        # error budget the controller has to satisfy.
        h_hist = diag_diff["h_max_history"]
        if len(h_hist) > 0:
            frac_near_atol = float(np.mean(h_hist < 10.0 * atol))
            print(f"  {cls:<15} [diffusion diag] h_max range=[{h_hist.min():.4g}, "
                  f"{h_hist.max():.4g}]  frac_steps_with_h_max<10*atol={frac_near_atol:.2f}")

        diag = ba.diagnose_stiffness_vs_divergence(
            h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol, n_steps=macro_steps,
            kind="reaction", event_id=d["event_id"], lifecycle_class=cls,
        )
        row["reaction_verdict"] = diag["verdict"]
        row["reaction_h_growth"] = diag["h_growth"]
        row["reaction_dt_shrink"] = diag["dt_shrink"]

    if not ok_dop:
        diag = ba.diagnose_stiffness_vs_divergence(
            h0_jax, diff_rhs, rxn_rhs, rtol=rtol, atol=atol, n_steps=macro_steps,
            kind="dopri5", event_id=d["event_id"], lifecycle_class=cls,
        )
        row["dopri5_verdict"] = diag["verdict"]
        row["dopri5_h_growth"] = diag["h_growth"]
        row["dopri5_dt_shrink"] = diag["dt_shrink"]

    # Stash what's needed later if a tolerance re-check is warranted.
    row["_h0_jax"] = h0_jax
    row["_diff_rhs"] = diff_rhs
    row["_rxn_rhs"] = rxn_rhs
    return row


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true",
                        help="Use real SEVIR data instead of synthetic events")
    parser.add_argument("--classes", type=str, default=",".join(ALL_CLASSES),
                        help="Comma-separated lifecycle classes to triage "
                             f"(default: all — {','.join(ALL_CLASSES)})")
    parser.add_argument("--n", type=int, default=60,
                        help="Test N (superpixel count) — kept small; this is a triage, not the sweep")
    parser.add_argument("--macro-steps", type=int, default=2,
                        help="Macro steps per integration (production default is "
                             f"{ba.N_MACRO_STEPS}; kept small to bound triage runtime)")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max internal solver steps for Tsit5/Kvaerno5")
    parser.add_argument("--max-steps-dopri5", type=int, default=1000,
                        help="Max internal DOPRI5 steps")
    parser.add_argument("--rtol", type=float, default=ba.RTOL_DEFAULT,
                        help=f"rtol to triage AT (default: module's own {ba.RTOL_DEFAULT:.1e}, "
                             "the spec's prescribed starting point — not pre-tightened)")
    parser.add_argument("--atol", type=float, default=ba.ATOL_DEFAULT,
                        help=f"atol to triage AT (default: module's own {ba.ATOL_DEFAULT:.1e})")
    parser.add_argument("--seed", type=int, default=0, help="Weight init seed")
    parser.add_argument("--check-tolerance", dest="check_tolerance", action="store_true",
                        default=True,
                        help="For any class whose verdict is STIFFNESS, also run A.3's "
                             "calibrate_tolerance() against it to confirm a tolerance/step "
                             "change actually resolves convergence (default: on)")
    parser.add_argument("--no-check-tolerance", dest="check_tolerance", action="store_false")
    parser.add_argument("--out", type=str, default="block_a_triage",
                        help="Output directory for triage.csv")
    args = parser.parse_args()

    ba.MAX_STEPS_EXPLICIT = args.max_steps
    ba.MAX_STEPS_IMPLICIT = args.max_steps
    ba.MAX_STEPS_DOPRI5 = args.max_steps_dopri5

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in classes if c not in ALL_CLASSES]
    if unknown:
        sys.exit(f"Unknown class(es) {unknown} — valid classes are {ALL_CLASSES}")

    os.makedirs(args.out, exist_ok=True)
    N_TEST = args.n

    event_data = load_event_data(classes, args.real, io_workers=8)
    c_sigma, D, class_sigma = calibrate_shared(event_data, N_TEST)

    print(f"\nTriaging at rtol={args.rtol:.2e}  atol={args.atol:.2e}  "
          f"macro_steps={args.macro_steps}  max_steps={args.max_steps}/"
          f"{args.max_steps_dopri5}(dopri5)…\n")

    rows = []
    for cls, d in event_data.items():
        row = triage_one_class(
            cls, d, N_TEST, c_sigma, D, class_sigma[cls],
            rtol=args.rtol, atol=args.atol, macro_steps=args.macro_steps, seed=args.seed,
        )
        rows.append(row)
        if row.get("imex_converged", True) and row.get("dopri5_converged", True):
            print(f"  {cls:<15} converged (both IMEX and DOPRI5) — nothing to triage")
            continue
        if not row["imex_converged"]:
            print(f"  {cls:<15} [reaction/Kvaerno5]  verdict={row['reaction_verdict']:<12}  "
                  f"h_growth={row['reaction_h_growth']:.2f}x  dt_shrink={row['reaction_dt_shrink']:.2e}x")
        if not row["dopri5_converged"]:
            print(f"  {cls:<15} [dopri5]             verdict={row['dopri5_verdict']:<12}  "
                  f"h_growth={row['dopri5_h_growth']:.2f}x  dt_shrink={row['dopri5_dt_shrink']:.2e}x")

    # ── Optional: for any STIFFNESS verdict, confirm a tolerance change
    # actually fixes it (A.3's calibrate_tolerance, run ONLY against
    # classes/sub-solvers that came back STIFFNESS — never against
    # DIVERGENCE, since that would just waste an expensive halving search
    # on a problem tolerance can't fix). ─────────────────────────────────
    tol_results: dict[str, tuple[float, float, bool]] = {}
    if args.check_tolerance:
        stiff_classes = [
            r for r in rows
            if r.get("reaction_verdict") == "STIFFNESS" or r.get("dopri5_verdict") == "STIFFNESS"
        ]
        if stiff_classes:
            print("\nSTIFFNESS verdict found — confirming via A.3 calibrate_tolerance()…")
            for r in stiff_classes:
                cls = r["lifecycle_class"]
                rtol_c, atol_c = ba.calibrate_tolerance(
                    r["_h0_jax"], r["_diff_rhs"], r["_rxn_rhs"],
                    rtol_start=args.rtol, label=cls,
                )
                # Re-check convergence at the returned tolerance to report
                # a plain converged/not-converged, not just the halving
                # search's last-good value.
                _h, _nd, _nr, _rd, _rr, ok = ba.imex_strang_integrate(
                    r["_h0_jax"], r["_diff_rhs"], r["_rxn_rhs"],
                    n_steps=args.macro_steps, rtol=rtol_c, atol=atol_c,
                    lifecycle_class=cls,
                )
                tol_results[cls] = (rtol_c, atol_c, ok)
                print(f"  {cls:<15} calibrate_tolerance -> rtol={rtol_c:.2e}  atol={atol_c:.2e}  "
                      f"converged={ok}")
        else:
            print("\nNo STIFFNESS verdicts — skipping calibrate_tolerance (it would be wasted "
                  "effort on classes that are converged or DIVERGENCE).")

    # ── Write flat CSV (drop the non-serialisable jax/callable fields) ──────
    csv_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    for r in csv_rows:
        cls = r["lifecycle_class"]
        if cls in tol_results:
            r["calibrated_rtol"], r["calibrated_atol"], r["calibrated_converges"] = tol_results[cls]
    df = pd.DataFrame(csv_rows)
    csv_path = os.path.join(args.out, "triage.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV -> {csv_path}")

    # ── Final one-line answer ────────────────────────────────────────────────
    verdicts = set()
    for r in rows:
        for k in ("reaction_verdict", "dopri5_verdict"):
            if r.get(k):
                verdicts.add(r[k])

    print("\n" + "=" * 78)
    if not verdicts:
        print("  ANSWER: NO FAILURES OBSERVED at this (N, macro_steps, max_steps, "
              "rtol/atol) — nothing to triage here. This does NOT confirm the real "
              "sweep's failures are absent; it means this cheap config didn't "
              "reproduce them. Re-run closer to the failing (event, N, seed) "
              "triples (larger N, more macro steps) if you need to reproduce first.")
    elif verdicts == {"STIFFNESS"}:
        print("  ANSWER: STIFFNESS. calibrate_tolerance() (A.3) is a correct and "
              "SUFFICIENT fix — see per-class results above for the tolerance that "
              "actually converges.")
    elif verdicts == {"DIVERGENCE"} or verdicts == {"DIVERGENCE (low data — see note)"} or \
         verdicts <= {"DIVERGENCE", "DIVERGENCE (low data — see note)"}:
        print("  ANSWER: DIVERGENCE. Tolerance calibration and raising max_steps will "
              "NOT fix this — more steps just delays the same blow-up. The reaction "
              "(or DOPRI5) output itself needs bounding (e.g. a calibrated Lipschitz "
              "ceiling on the reaction MLPs per A.2), which is a methodology change, "
              "not a solver setting. Flag this to your advisor before spending compute "
              "on a tolerance sweep.")
    elif "AMBIGUOUS" in verdicts and len(verdicts) == 1:
        print("  ANSWER: AMBIGUOUS. Too few accepted steps to classify — rerun with "
              "more macro_steps or a larger max_steps budget purely to get enough "
              "accepted-step history for a verdict, then re-triage.")
    else:
        print(f"  ANSWER: MIXED across classes ({sorted(verdicts)}). Treat per-class, "
              "not as one global fix — see the table above/CSV for which classes "
              "need calibrate_tolerance vs. which need the reaction output bounded.")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""
python triage_stiffness_vs_divergence.py --real --classes RAPID_GROWTH,GROWTH_DECAY --n 1150 --macro-steps 12 --max-steps 4000 --max-steps-dopri5 50000
"""
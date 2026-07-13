"""
block_a_solver_benchmark.py
============================
Block A: Empirical solver benchmark — IMEX Strang-split (Tsit5 + Kvaerno5)
vs. DOPRI5 (unsplit) on the Lagrangian superpixel graph built from SEVIR events.
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
import warnings
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# THREADING LIMITS (CRITICAL FOR MULTIPROCESSING)
# We must force underlying C-libraries to single-threaded mode BEFORE importing 
# them. Otherwise, multiple Python processes will spawn hundreds of competing 
# threads, thrashing the CPU.
# ─────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_updates=false"

import h5py
import numpy as np
import pandas as pd
import scipy.spatial
from skimage.segmentation import slic as skimage_slic
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_ROOT      = r"C:\Users\Siddharth Nair\OneDrive\Desktop\BTP-2\GAT-ODE\SEVIR"
CATALOGUE_PATH = os.path.join(DATA_ROOT, "event_catalogue.csv")
OUTPUT_CSV     = os.path.join(DATA_ROOT, "block_a_results.csv")
SUMMARY_CSV    = os.path.join(DATA_ROOT, "block_a_summary.csv")

N_VALUES            = [500, 750] 
N_EVENTS_PER_CLASS  = 10    
N_SEEDS             = 3     

R_MAX_PX  = 108    
SIGMA_COEFF = 3.0  

DT_MACRO      = 300.0   
N_MACRO_STEPS = 12      
NODE_DIM      = 3       
ENV_DIM       = 4       
HIDDEN_DIM    = 64      

RTOL               = 1e-3
ATOL               = 1e-4
MAX_STEPS_EXPLICIT = 4000
MAX_STEPS_IMPLICIT = 4000
MAX_STEPS_DOPRI5   = 50_000  

STIFFNESS_SCALE: dict[str, float] = {
    "RAPID_GROWTH": 4.0,  
    "GROWTH_DECAY": 3.0,  
    "EPISODIC":     2.0,  
    "PLATEAU":      1.5,  
    "RAPID_DECAY":  1.5,  
    "STEADY":       1.0,  
    "QUIESCENT":    0.5,  
}

try:
    import jax
    import jax.numpy as jnp
    import diffrax
    import cv2
    JAX_OK = True
except ImportError as e:
    log.error(f"JAX / Diffrax not found: {e}")
    JAX_OK = False

# ─────────────────────────────────────────────────────────────────────────────
# SEVIR DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def get_local_path(catalog_filename: str) -> Optional[str]:
    p1 = os.path.join(DATA_ROOT, catalog_filename)
    if os.path.exists(p1): return p1
    parts = catalog_filename.replace("\\", "/").split("/")
    if len(parts) == 3:
        p2 = os.path.join(DATA_ROOT, parts[1], parts[0], parts[2])
        if os.path.exists(p2): return p2
    return None

def _read_channel_from_hdf5(file_path: str, img_type: str, event_id: str) -> Optional[np.ndarray]:
    try:
        with h5py.File(file_path, "r") as f:
            if img_type not in f or "id" not in f: return None
            file_ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in f["id"][:]]
            if event_id not in file_ids: return None
            idx  = file_ids.index(event_id)
            data = f[img_type][idx].astype(np.float32)
            if data.ndim == 3 and data.shape[2] < data.shape[0]:
                data = data.transpose(2, 0, 1)
            return data
    except Exception:
        return None

def _normalise_channel(data: np.ndarray, img_type: str) -> np.ndarray:
    if img_type == "vil":
        return np.clip(data / 255.0, 0.0, 1.0)
    else:
        data = np.clip(data, 0.0, None)
        v_max = data.max()
        if v_max > 0.0: data = data / v_max
        return data

def load_event_multichannel(event_id: str, catalog_df: pd.DataFrame) -> Optional[dict[str, np.ndarray]]:
    T_ref, H_ref, W_ref = None, None, None
    channels: dict[str, np.ndarray] = {}

    for img_type in ["vil", "ir069", "ir107"]:
        rows = catalog_df[(catalog_df["id"] == event_id) & (catalog_df["img_type"] == img_type)]
        if rows.empty:
            channels[img_type] = None
            continue

        file_path = get_local_path(rows.iloc[0]["file_name"])
        if not file_path:
            channels[img_type] = None
            continue

        raw = _read_channel_from_hdf5(file_path, img_type, event_id)
        if raw is None:
            channels[img_type] = None
            continue

        channels[img_type] = _normalise_channel(raw, img_type)
        if T_ref is None:
            T_ref, H_ref, W_ref = channels[img_type].shape

    if channels.get("vil") is None: return None
    T_ref, H_ref, W_ref = channels["vil"].shape

    for k in ["ir069", "ir107"]:
        if channels[k] is None:
            channels[k] = np.zeros((T_ref, H_ref, W_ref), dtype=np.float32)

    return channels

# ─────────────────────────────────────────────────────────────────────────────
# SUPERPIXEL GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_graph(channels: dict[str, np.ndarray], N: int, t_idx: int = 0) -> Optional[dict]:
    vil   = channels["vil"][t_idx]    
    ir069 = channels["ir069"][t_idx]
    ir107 = channels["ir107"][t_idx]

    H, W = vil.shape
    if ir069.shape != (H, W): ir069 = cv2.resize(ir069, (W, H), interpolation=cv2.INTER_CUBIC)
    if ir107.shape != (H, W): ir107 = cv2.resize(ir107, (W, H), interpolation=cv2.INTER_CUBIC)

    rgb = np.stack([vil, ir069, ir107], axis=-1) 

    try:
        segments = skimage_slic(rgb, n_segments=N, compactness=10, start_label=0, convert2lab=False)
    except Exception:
        return None

    unique_ids = np.unique(segments)
    N_act      = len(unique_ids)

    positions = np.zeros((N_act, 2), dtype=np.float32)
    h0        = np.zeros((N_act, 3), dtype=np.float32)

    for k, seg_id in enumerate(unique_ids):
        mask         = (segments == seg_id)
        ys, xs       = np.where(mask)
        positions[k] = [xs.mean(), ys.mean()]       
        h0[k, 0]     = vil[mask].mean()
        h0[k, 1]     = ir069[mask].mean()
        h0[k, 2]     = ir107[mask].mean()

    env = np.zeros((N_act, ENV_DIM), dtype=np.float32)
    env[:, 0] = float(h0[:, 1].mean())   
    env[:, 1] = float(h0[:, 2].mean())   
    env[:, 2] = float(h0[:, 0].std())    

    dist_mat  = scipy.spatial.distance.cdist(positions, positions)
    mask_edge = (dist_mat < R_MAX_PX) & (dist_mat > 0.0)
    senders, receivers = np.where(mask_edge)
    
    sigma        = R_MAX_PX / SIGMA_COEFF
    edge_dists   = dist_mat[senders, receivers]
    edge_weights = np.exp(-edge_dists**2 / (2.0 * sigma**2)).astype(np.float32)

    return {
        "positions": positions, "h0": h0, "env": env,
        "senders": senders.astype(np.int32), "receivers": receivers.astype(np.int32),
        "edge_weights": edge_weights, "N_actual": N_act,
    }

# ─────────────────────────────────────────────────────────────────────────────
# JAX WEIGHT INITIALISATION & ODE RHS
# ─────────────────────────────────────────────────────────────────────────────
def make_weights(stiffness_scale: float, seed: int) -> dict:
    keys  = jax.random.split(jax.random.PRNGKey(seed), 7)
    s     = stiffness_scale / float(np.sqrt(HIDDEN_DIM))

    return {
        "W_flux": jax.random.normal(keys[0], (NODE_DIM, NODE_DIM)) * 0.1,
        "gate_w": jax.random.normal(keys[1], (NODE_DIM + ENV_DIM, 1)) * 0.1,
        "gate_b": jnp.zeros(1),
        "src_w1": jax.random.normal(keys[2], (NODE_DIM + ENV_DIM, HIDDEN_DIM)) * s,
        "src_w2": jax.random.normal(keys[3], (HIDDEN_DIM, NODE_DIM)) * s,
        "snk_w1": jax.random.normal(keys[4], (NODE_DIM + ENV_DIM, HIDDEN_DIM)) * s,
        "snk_w2": jax.random.normal(keys[5], (HIDDEN_DIM, NODE_DIM)) * s,
    }

def make_diffusion_rhs(senders_np, receivers_np, edge_weights_np, W_flux_jax):
    s, r, ew, Wf = jnp.array(senders_np), jnp.array(receivers_np), jnp.array(edge_weights_np), W_flux_jax                   
    def diffusion_rhs(t, h, args):
        diff = h[r] - h[s]                          
        flux = ew[:, None] * (diff @ Wf.T)          
        return jnp.zeros_like(h).at[r].add(flux)    
    return diffusion_rhs

def make_reaction_rhs(env_jax, weights):
    gw, gb = weights["gate_w"], weights["gate_b"]   
    sw1, sw2 = weights["src_w1"], weights["src_w2"]   
    kw1, kw2 = weights["snk_w1"], weights["snk_w2"]
    def reaction_rhs(t, h, args):
        h_env = jnp.concatenate([h, env_jax], axis=-1)         
        s     = jax.nn.sigmoid(h_env @ gw + gb)            
        f_src = jnp.tanh(h_env @ sw1) @ sw2                
        f_snk = jnp.tanh(h_env @ kw1) @ kw2               
        return s * f_src - (1.0 - s) * f_snk               
    return reaction_rhs

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATORS
# ─────────────────────────────────────────────────────────────────────────────
def _run_substep(term, solver, t0, t1, dt0, y0, max_steps, pid):
    sol = diffrax.diffeqsolve(term, solver, t0=t0, t1=t1, dt0=dt0, y0=y0, args=None, stepsize_controller=pid, saveat=diffrax.SaveAt(t1=True), max_steps=max_steps, throw=False)
    n_steps = int(sol.stats["num_steps"])
    converged = bool(sol.result == diffrax.RESULTS.successful) if hasattr(sol.result, "successful") else (n_steps < max_steps - 1)
    return sol.ys[-1], n_steps, converged

def imex_strang_integrate(h0_jax, diff_rhs_fn, rxn_rhs_fn, dt=DT_MACRO, n_steps=N_MACRO_STEPS):
    pid = diffrax.PIDController(rtol=RTOL, atol=ATOL)
    h, total_nfe_diff, total_nfe_rxn, converged = h0_jax, 0, 0, True

    for k in range(n_steps):
        t_s, t_half, t_e = float(k) * dt, float(k) * dt + dt / 2.0, float(k) * dt + dt             
        h, nfe_d1, ok = _run_substep(diffrax.ODETerm(diff_rhs_fn), diffrax.Tsit5(), t_s, t_half, dt/20.0, h, MAX_STEPS_EXPLICIT, pid)
        h, nfe_r, ok2 = _run_substep(diffrax.ODETerm(rxn_rhs_fn), diffrax.Kvaerno5(), t_s, t_e, dt/2.0, h, MAX_STEPS_IMPLICIT, pid)
        h, nfe_d2, ok3 = _run_substep(diffrax.ODETerm(diff_rhs_fn), diffrax.Tsit5(), t_half, t_e, dt/20.0, h, MAX_STEPS_EXPLICIT, pid)
        total_nfe_diff += nfe_d1 + nfe_d2
        total_nfe_rxn += nfe_r
        converged = converged and ok and ok2 and ok3
    return h, total_nfe_diff, total_nfe_rxn, converged

def dopri5_integrate(h0_jax, diff_rhs_fn, rxn_rhs_fn, dt=DT_MACRO, n_steps=N_MACRO_STEPS):
    pid = diffrax.PIDController(rtol=RTOL, atol=ATOL)
    term = diffrax.ODETerm(lambda t, h, args: diff_rhs_fn(t, h, args) + rxn_rhs_fn(t, h, args))
    h, total_nfe, converged = h0_jax, 0, True

    for k in range(n_steps):
        t_s, t_e = float(k) * dt, float(k) * dt + dt
        sol = diffrax.diffeqsolve(term, diffrax.Dopri5(), t0=t_s, t1=t_e, dt0=dt/10.0, y0=h, args=None, stepsize_controller=pid, saveat=diffrax.SaveAt(t1=True), max_steps=MAX_STEPS_DOPRI5, throw=False)
        h = sol.ys[-1]
        total_nfe += int(sol.stats["num_steps"])
        converged = converged and (bool(sol.result == diffrax.RESULTS.successful) if hasattr(sol.result, "successful") else (int(sol.stats["num_steps"]) < MAX_STEPS_DOPRI5 - 1))
    return h, total_nfe, converged

def estimate_spectral_radius(rxn_rhs_fn, h_sample, n_power_iter=20):
    v = jax.random.normal(jax.random.PRNGKey(99), h_sample.shape)
    v /= (jnp.linalg.norm(v) + 1e-12)
    def jvp_fn(v): return jax.jvp(lambda h: rxn_rhs_fn(0.0, h, None), (h_sample,), (v,))[1]
    
    rho = 0.0
    for _ in range(n_power_iter):
        u = jvp_fn(v)
        rho = float(jnp.linalg.norm(u) / (jnp.linalg.norm(v) + 1e-12))
        v = u / (jnp.linalg.norm(u) + 1e-12)
    return rho

# ─────────────────────────────────────────────────────────────────────────────
# ISOLATED WORKER FUNCTION (BYPASSES GIL)
# ─────────────────────────────────────────────────────────────────────────────
def process_event_worker(task_args: dict) -> list[dict]:
    """
    This function runs entirely inside its own process. 
    It has its own Python interpreter and its own GIL, completely decoupled 
    from the main script.
    """
    event_id = task_args["event_id"]
    lc_class = task_args["lifecycle_class"]
    n_values = task_args["n_values"]
    seeds = task_args["seeds"]
    sevir_catalog = task_args["sevir_catalog"]

    channels = load_event_multichannel(event_id, sevir_catalog)
    if channels is None: return []

    rows = []
    scale = STIFFNESS_SCALE.get(lc_class, 1.0)

    for N in n_values:
        graph = build_graph(channels, N=N, t_idx=0)
        if graph is None: continue

        N_act, E = graph["N_actual"], len(graph["senders"])
        env_jax, h0_jax = jnp.array(graph["env"]), jnp.array(graph["h0"])

        for seed in seeds:
            weights = make_weights(scale, seed)
            diff_rhs = make_diffusion_rhs(graph["senders"], graph["receivers"], graph["edge_weights"], weights["W_flux"])
            rxn_rhs  = make_reaction_rhs(env_jax, weights)

            rho = float("nan")
            if seed == seeds[0]:
                try: rho = estimate_spectral_radius(rxn_rhs, h0_jax)
                except Exception: pass

            t0_imex = time.perf_counter()
            h_imex, nfe_diff, nfe_rxn, ok_imex = imex_strang_integrate(h0_jax, diff_rhs, rxn_rhs)
            wall_imex = time.perf_counter() - t0_imex

            t0_dop = time.perf_counter()
            h_dop, nfe_dop, ok_dop = dopri5_integrate(h0_jax, diff_rhs, rxn_rhs)
            wall_dop = time.perf_counter() - t0_dop

            l2_err = float(jnp.linalg.norm(h_imex - h_dop)) / float(jnp.linalg.norm(h_dop) + 1e-12)

            rows.append({
                "event_id": event_id, "lifecycle_class": lc_class, "N": N, "N_actual": N_act, "E": E, "seed": seed,
                "stiffness_scale": scale, "spectral_radius": rho,
                "nfe_diffusion": nfe_diff, "nfe_reaction": nfe_rxn, "nfe_imex_total": nfe_diff + nfe_rxn,
                "wall_imex_sec": round(wall_imex, 4), "imex_converged": ok_imex,
                "nfe_dopri5": nfe_dop, "wall_dopri5_sec": round(wall_dop, 4), "dopri5_converged": ok_dop,
                "nfe_ratio": round(nfe_dop / (nfe_diff + nfe_rxn + 1), 4), "solution_l2_error": round(l2_err, 6),
            })
    return rows

# ─────────────────────────────────────────────────────────────────────────────
# MAIN BENCHMARK LOOP
# ─────────────────────────────────────────────────────────────────────────────
def run_benchmark(catalogue_path: str = CATALOGUE_PATH, n_events_per_class: int = N_EVENTS_PER_CLASS, n_seeds: int = N_SEEDS, n_values: list = N_VALUES) -> pd.DataFrame:
    catalogue = pd.read_csv(catalogue_path, low_memory=False)
    sevir_catalog = pd.read_csv(os.path.join(DATA_ROOT, "CATALOG.csv"), low_memory=False)

    sampled = catalogue.groupby("lifecycle_class", group_keys=False).apply(lambda g: g.sample(min(n_events_per_class, len(g)), random_state=42)).reset_index(drop=True)

    tasks = [{"event_id": str(row["id"]), "lifecycle_class": str(row["lifecycle_class"]), "n_values": n_values, "seeds": list(range(n_seeds)), "sevir_catalog": sevir_catalog} for _, row in sampled.iterrows()]

    all_rows = []
    
    # ── GIL Bypass execution ── 
    max_workers = multiprocessing.cpu_count()
    log.info(f"Bypassing GIL: Spawning {max_workers} independent isolated processes...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_event_worker, task): task["event_id"] for task in tasks}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Events"):
            event_id = futures[future]
            try:
                event_rows = future.result()
                all_rows.extend(event_rows)
            except Exception as exc:
                log.error(f"Event {event_id} failed: {exc}")

    return pd.DataFrame(all_rows)

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["lifecycle_class", "N"]).agg(
        nfe_diffusion_mean=("nfe_diffusion", "mean"), nfe_reaction_mean=("nfe_reaction", "mean"),
        nfe_imex_total_mean=("nfe_imex_total", "mean"), nfe_dopri5_mean=("nfe_dopri5", "mean"),
        nfe_ratio_mean=("nfe_ratio", "mean"), dopri5_converged_frac=("dopri5_converged", "mean"),
        imex_converged_frac=("imex_converged", "mean")
    ).reset_index()

def main():
    parser = argparse.ArgumentParser(description="Block A: IMEX vs DOPRI5 parallelized benchmark")
    parser.add_argument("--n_events", type=int, default=N_EVENTS_PER_CLASS)
    parser.add_argument("--seeds",    type=int, default=N_SEEDS)
    parser.add_argument("--n_values", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--out",      type=str, default=OUTPUT_CSV)
    parser.add_argument("--summary",  type=str, default=SUMMARY_CSV)
    args = parser.parse_args()

    if not JAX_OK: sys.exit(1)

    results_df = run_benchmark(CATALOGUE_PATH, args.n_events, args.seeds, args.n_values)
    
    if results_df.empty: sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    results_df.to_csv(args.out, index=False)
    summary_df = build_summary(results_df)
    summary_df.to_csv(args.summary, index=False)
    log.info("Benchmark complete.")

if __name__ == "__main__":
    # Required for Windows multiprocessing to properly spawn isolated processes
    multiprocessing.freeze_support()
    main()
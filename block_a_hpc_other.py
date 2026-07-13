"""
block_a_hpc.py
==============
Block A solver benchmark — PARAM Rudra (A100 GPU) version.

Measures NFE breakdown (diffusion vs reaction) for IMEX vs DOPRI5
across all 7 lifecycle classes over N in {250,500,750,1000,1150,1500}.

GPU additions over smoke test
-----------------------------
  1. GPU Temporal SLIC  — JAX-based iterative k-means with SLIC distance
                          (O(H*W*K) per iteration, JIT-compiled once per K)
  2. GPU Optical Flow   — Dense Lucas-Kanade in JAX for centroid advection
  3. Temporal coherence — Advect centroids by flow, warm-start SLIC at t>0
  4. Full SEVIR loading — VIL + IR069 + IR107 from HDF5

Usage
-----
    python block_a_hpc.py \
        --catalogue /lustre/path/event_catalogue.csv \
        --raw_catalog /lustre/path/CATALOG.csv \
        --data_root /local_ssd/SEVIR \
        --n_per_class 5 \
        --n_seeds 5 \
        --out /lustre/path/results/block_a_full.csv
"""

import os, sys, time, warnings, argparse
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from tqdm import tqdm
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

warnings.filterwarnings("ignore")

# =============================================================================
# SECTION 1 — Configuration
# =============================================================================

DATA_ROOT    = "/lustre/scratch/your_project/SEVIR"
CATALOG_PATH = os.path.join(DATA_ROOT, "CATALOG.csv")
DEFAULT_CAT  = os.path.join(DATA_ROOT, "event_catalogue.csv")
DEFAULT_OUT  = "/lustre/scratch/your_project/results/block_a_full.csv"

N_VALUES    = [250, 500, 750, 1000, 1150, 1500]
N_PER_CLASS = 5
N_SEEDS     = 5

H_GRID, W_GRID = 384, 384
N_FRAMES       = 13

NODE_DIM   = 3
ENV_DIM    = 4
HIDDEN_DIM = 64
DT         = 300.0
N_MACRO    = 12

R_MAX     = 108
SIGMA_RBF = 15.0

SLIC_KAPPA      = 10.0
SLIC_ITERS_INIT = 10
SLIC_ITERS_TEMP = 5

LK_WINDOW   = 15
MAX_FLOW_PX = 15.0

STIFFNESS_SCALE = {
    "RAPID_GROWTH": 4.0,
    "GROWTH_DECAY": 3.0,
    "EPISODIC":     2.0,
    "PLATEAU":      1.5,
    "RAPID_DECAY":  1.5,
    "STEADY":       1.0,
    "QUIESCENT":    0.5,
}
LIFECYCLE_ORDER = [
    "RAPID_GROWTH", "GROWTH_DECAY", "EPISODIC",
    "PLATEAU", "RAPID_DECAY", "STEADY", "QUIESCENT",
]
NFE_MUL = {"tsit5": 5, "kvaerno5": 7, "dopri5": 6}


# =============================================================================
# SECTION 2 — GPU Temporal SLIC (JAX)
# =============================================================================

def _norm_np(arr):
    return (arr - arr.mean()) / (arr.std() + 1e-6)


def init_centers_np(features_norm, K):
    """
    Initialise SLIC centres on a regular pixel grid (CPU, called once per N).
    Returns (K, C+2) float32 where last 2 cols are [x_norm, y_norm].
    """
    H, W, C = features_norm.shape
    S = max(1, int(np.sqrt(H * W / K)))
    ys = np.arange(S // 2, H, S, dtype=np.float32)
    xs = np.arange(S // 2, W, S, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy, xx = yy.ravel(), xx.ravel()
    if len(yy) < K:
        rep = int(np.ceil(K / len(yy)))
        yy  = np.tile(yy, rep)[:K]
        xx  = np.tile(xx, rep)[:K]
    yy, xx = yy[:K], xx[:K]
    yi = np.clip(yy.astype(int), 0, H - 1)
    xi = np.clip(xx.astype(int), 0, W - 1)
    feat_vals = features_norm[yi, xi, :]
    x_norm    = (xx / W).reshape(-1, 1)
    y_norm    = (yy / H).reshape(-1, 1)
    return np.concatenate([feat_vals, x_norm, y_norm], axis=1).astype(np.float32)


@partial(jax.jit, static_argnums=(2, 3, 4))
def gpu_slic_iterations(pixels, centers_init, K, C, n_iters):
    """
    GPU SLIC core: iterative k-means with SLIC distance metric.

    pixels       : (H*W, C+2)   flattened features + [x_norm, y_norm]
    centers_init : (K, C+2)     initial cluster centres
    K            : int  static  (enables JIT specialisation per N value)
    C            : int  static  number of feature channels
    n_iters      : int  static

    Distance: d^2 = d_colour^2 + (m/S)^2 * d_spatial^2
    Spatial constraint: ignore clusters beyond 2*S_norm (SLIC locality).

    Memory: O(H*W*K) per iteration.
    At (384^2, K=1500): ~170 M float32 entries per distance matrix = 680 MB.
    Well within 80 GB A100 HBM.

    Returns (labels (H*W,), centers (K,C+2)).
    """
    S_norm = jnp.sqrt(1.0 / K)
    m2     = SLIC_KAPPA ** 2

    def one_iter(centers, _):
        d_col = jnp.sum(
            (pixels[:, None, :C] - centers[None, :, :C]) ** 2, axis=-1
        )  # (H*W, K)
        d_spa = jnp.sum(
            (pixels[:, None, C:] - centers[None, :, C:]) ** 2, axis=-1
        )  # (H*W, K)

        mask  = d_spa > (2.0 * S_norm) ** 2
        d_tot = d_col + (m2 / S_norm ** 2) * d_spa
        d_tot = jnp.where(mask, jnp.inf, d_tot)

        labels   = jnp.argmin(d_tot, axis=1).astype(jnp.int32)  # (H*W,)
        new_sum  = jnp.zeros((K, C + 2)).at[labels].add(pixels)
        counts   = jnp.zeros(K).at[labels].add(1.0)
        new_cen  = new_sum / (counts[:, None] + 1e-8)
        new_cen  = new_cen.at[:, C:].set(jnp.clip(new_cen[:, C:], 0.0, 1.0))
        return new_cen, labels

    final_centers, all_labels = jax.lax.scan(
        one_iter, centers_init, None, length=n_iters
    )
    return all_labels[-1], final_centers   # (H*W,), (K, C+2)


def gpu_slic_frame(frames_norm, K, t, centers_prev=None, n_iters=SLIC_ITERS_INIT):
    """
    Segment one frame on GPU.
    At t=0: uniform grid init.
    At t>0: warm start from advected centres with SLIC_ITERS_TEMP iterations.
    """
    H, W, C = frames_norm.shape[1:]
    feat_t   = frames_norm[t]                       # (H, W, C)

    xs  = jnp.linspace(0, 1, W)
    ys  = jnp.linspace(0, 1, H)
    yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
    spatial   = jnp.stack([xx, yy], axis=-1)         # (H, W, 2)
    feat_jax  = jnp.array(feat_t)
    feat_full = jnp.concatenate([feat_jax, spatial], axis=-1)   # (H, W, C+2)
    pixels    = feat_full.reshape(-1, C + 2)

    if centers_prev is None:
        centers_init = jnp.array(init_centers_np(feat_t, K))
    else:
        # Refresh feature values at advected centroid positions
        cen_xi = jnp.clip((centers_prev[:, C]   * W).astype(int), 0, W - 1)
        cen_yi = jnp.clip((centers_prev[:, C+1] * H).astype(int), 0, H - 1)
        new_f  = feat_jax[cen_yi, cen_xi, :]           # (K, C)
        centers_init = centers_prev.at[:, :C].set(new_f)

    labels, centers = gpu_slic_iterations(
        pixels, centers_init, K, C, n_iters
    )
    return labels, centers   # (H*W,) int32,  (K, C+2) float32


# =============================================================================
# SECTION 3 — GPU Optical Flow (Dense Lucas-Kanade, JAX)
# =============================================================================

@jax.jit
def gpu_lk_flow(frame_prev, frame_curr):
    """
    Dense Lucas-Kanade optical flow on GPU.
    All operations are JAX-native — runs entirely on A100.

    frame_prev, frame_curr : (H, W) float32
    Returns : (H, W, 2) flow field [u=dx, v=dy] in pixels/frame,
              clipped to [-MAX_FLOW_PX, MAX_FLOW_PX].
    """
    Ix = (jnp.roll(frame_curr, -1, axis=1) - jnp.roll(frame_curr, 1, axis=1)) * 0.5
    Iy = (jnp.roll(frame_curr, -1, axis=0) - jnp.roll(frame_curr, 1, axis=0)) * 0.5
    It = frame_curr - frame_prev

    def box(x):
        w   = LK_WINDOW
        k4  = jnp.ones((1, 1, w, w), dtype=jnp.float32) / (w * w)
        pad = w // 2
        return jax.lax.conv_general_dilated(
            x[None, None], k4,
            window_strides=(1, 1),
            padding=[(pad, pad), (pad, pad)],
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        )[0, 0]

    Sxx = box(Ix * Ix)
    Syy = box(Iy * Iy)
    Sxy = box(Ix * Iy)
    Sxt = box(Ix * It)
    Syt = box(Iy * It)

    det = Sxx * Syy - Sxy * Sxy + 1e-6
    u   = jnp.clip((-Sxt * Syy + Syt * Sxy) / det, -MAX_FLOW_PX, MAX_FLOW_PX)
    v   = jnp.clip((-Syt * Sxx + Sxt * Sxy) / det, -MAX_FLOW_PX, MAX_FLOW_PX)
    return jnp.stack([u, v], axis=-1)   # (H, W, 2)


@jax.jit
def bilinear_sample_flow(flow, centers, C=NODE_DIM):
    """
    Bilinear interpolation of dense flow at superpixel centroid positions.
    centers[:, C:] = [x_norm, y_norm].
    Returns (K, 2) flow in normalised [0,1] coordinates.
    """
    H, W = flow.shape[:2]
    xp   = centers[:, C]     * W
    yp   = centers[:, C + 1] * H
    x0   = jnp.floor(xp).astype(jnp.int32).clip(0, W - 2)
    y0   = jnp.floor(yp).astype(jnp.int32).clip(0, H - 2)
    wx   = (xp - x0)[:, None]
    wy   = (yp - y0)[:, None]
    f    = ((1-wx)*(1-wy)*flow[y0,x0]   + wx*(1-wy)*flow[y0,x0+1] +
            (1-wx)*wy   *flow[y0+1,x0] + wx*wy     *flow[y0+1,x0+1])
    return f / jnp.array([W, H], dtype=jnp.float32)   # normalised


# =============================================================================
# SECTION 4 — Temporal SLIC pipeline
# =============================================================================

def temporal_slic_pipeline(frames_norm, K):
    """
    Run GPU Temporal SLIC over T frames.

    t=0 : full GPU SLIC (SLIC_ITERS_INIT=10 iterations).
    t>0 : dense LK optical flow on GPU → advect centroids →
          warm-start GPU SLIC (SLIC_ITERS_TEMP=5 iterations).

    Returns list of T dicts:
        labels        (H, W)             int32
        centers       (K, C+2)           float32
        h_state       (actual_K, C)      float32  area-weighted mean
        centroids_px  (actual_K, 2)      float32  pixel coords [x, y]
        actual_K      int
    """
    T, H, W, C = frames_norm.shape
    results      = []
    centers_prev = None

    for t in range(T):
        if t > 0 and centers_prev is not None:
            prev = jnp.array(frames_norm[t - 1, :, :, 0])
            curr = jnp.array(frames_norm[t,     :, :, 0])
            flow = gpu_lk_flow(prev, curr)                          # GPU LK flow
            fa   = bilinear_sample_flow(flow, centers_prev, C)     # flow at centroids
            adv  = jnp.clip(centers_prev[:, C:] + fa, 0.0, 1.0)
            centers_prev = centers_prev.at[:, C:].set(adv)
            n_it = SLIC_ITERS_TEMP
        else:
            n_it = SLIC_ITERS_INIT

        labels, centers = gpu_slic_frame(
            frames_norm, K, t, centers_prev=centers_prev, n_iters=n_it
        )
        centers_prev = centers

        # Node states on CPU for graph construction
        labels_np    = np.array(labels).reshape(H, W)
        actual_K     = int(labels_np.max()) + 1
        h_state      = np.zeros((actual_K, C),  dtype=np.float32)
        centroids_px = np.zeros((actual_K, 2),  dtype=np.float32)
        for sid in range(actual_K):
            mask = labels_np == sid
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            centroids_px[sid] = [xs.mean(), ys.mean()]
            h_state[sid]      = frames_norm[t][mask].mean(axis=0)

        results.append({
            "labels":       labels_np,
            "centers":      np.array(centers),
            "h_state":      h_state,
            "centroids_px": centroids_px,
            "actual_K":     actual_K,
        })

    return results


# =============================================================================
# SECTION 5 — Lagrangian graph construction
# =============================================================================

def build_lagrangian_graph(centroids_px, h_state, H=H_GRID, W=W_GRID):
    """
    Build edge superset E_super at t=0 (Section 4.3.2 of DRIFT report).
    All edges within R_MAX pre-allocated; deactivated beyond R_ACT via RBF weight.
    For Block A, positions are static so weights remain constant.
    """
    actual_K = centroids_px.shape[0]
    env = np.stack([
        centroids_px[:, 0] / W,
        centroids_px[:, 1] / H,
        h_state[:, 0],
        h_state[:, 1],
    ], axis=-1).astype(np.float32)

    tree  = KDTree(centroids_px)
    pairs = np.array(sorted(tree.query_pairs(R_MAX)), dtype=np.int32)
    if len(pairs) == 0:
        k   = min(6, actual_K - 1)
        _, idx = tree.query(centroids_px, k=k + 1)
        pairs = np.array(
            [(i, int(idx[i, j])) for i in range(actual_K)
             for j in range(1, k + 1) if i < int(idx[i, j])],
            dtype=np.int32,
        )

    snd = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    rcv = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)
    dsq = np.sum((centroids_px[snd] - centroids_px[rcv]) ** 2, axis=1)
    ew  = np.exp(-dsq / (2.0 * SIGMA_RBF ** 2)).astype(np.float32)

    return {
        "actual_K":   actual_K,
        "E_super":    len(snd),
        "h0":         jnp.array(h_state),
        "env":        jnp.array(env),
        "graph_args": (jnp.array(snd), jnp.array(rcv), jnp.array(ew)),
    }


# =============================================================================
# SECTION 6 — Neural modules
# =============================================================================

class ExplicitDiffusion(eqx.Module):
    """
    Isotropic graph diffusion: dh_i/dt = D * sum_j ew_ij * (h_j - h_i)
    Scalar D>0 guarantees stability (negative-semidefinite graph Laplacian).
    CFL: Dt_max = 3.5/(D*lambda_max); D=0.05, lambda_max~3 -> Dt~23s -> 7 steps/150s.
    """
    log_D: jax.Array
    def __init__(self, key=None):
        self.log_D = jnp.log(jnp.array(0.05))
    def __call__(self, t, h, args):
        snd, rcv, ew = args
        D = jnp.exp(self.log_D)
        return jnp.zeros_like(h).at[rcv].add(D * ew[:, None] * (h[rcv] - h[snd]))


class ImplicitReaction(eqx.Module):
    """
    Stiff linear decay toward learned equilibrium: f_I = f_eq(env) - k_decay*h
    Jacobian: df_I/dh = -k_decay*I  ->  eigenvalue = -k_decay  (stiff).
    f_eq in [0,1]^d (bounded) -> output stays bounded, no blowup.

    SR per macro step = k_decay * DT:
        RAPID_GROWTH (scale=4.0): k_decay=1.2/s -> SR=360
        STEADY       (scale=1.0): k_decay=0.3/s -> SR=90
        QUIESCENT    (scale=0.5): k_decay=0.15/s -> SR=45
    """
    k_decay: float
    eq_w1:   jax.Array
    eq_w2:   jax.Array
    def __init__(self, stiffness_scale=1.0, key=None):
        if key is None: key = jax.random.PRNGKey(1)
        k1, k2        = jax.random.split(key, 2)
        self.k_decay  = float(stiffness_scale * 0.3)
        inp           = NODE_DIM + ENV_DIM
        self.eq_w1    = jax.random.normal(k1, (inp, HIDDEN_DIM)) * 0.1
        self.eq_w2    = jax.random.normal(k2, (HIDDEN_DIM, NODE_DIM)) * 0.1
    def __call__(self, t, h, env):
        h_env = jnp.concatenate([h, env], axis=-1)
        f_eq  = jax.nn.sigmoid(jnp.tanh(h_env @ self.eq_w1) @ self.eq_w2)
        return f_eq - self.k_decay * h


# =============================================================================
# SECTION 7 — IMEX integrator (Strang: Tsit5 + Kvaerno5)
# =============================================================================

def _pid():
    return diffrax.PIDController(rtol=1e-2, atol=1e-3)


def strang_macro_step(h, t, dt, diff_fn, rxn_fn, ga, env, max_steps=16384):
    """One macro step: L_D(dt/2) -> L_R(dt) -> L_D(dt/2). Returns (h, sd, sr, rej)."""
    term_E = diffrax.ODETerm(lambda t, h, a: diff_fn(t, h, a))
    term_I = diffrax.ODETerm(lambda t, h, e: rxn_fn(t, h, e))
    pid    = _pid()

    sol1  = diffrax.diffeqsolve(term_E, diffrax.Tsit5(),
                t0=t, t1=t+dt/2, dt0=dt/20, y0=h, args=ga,
                stepsize_controller=pid, saveat=diffrax.SaveAt(t1=True),
                max_steps=max_steps)
    h1    = sol1.ys[-1]
    sd1   = int(sol1.stats["num_accepted_steps"])
    rej1  = int(sol1.stats["num_rejected_steps"])

    sol2  = diffrax.diffeqsolve(term_I, diffrax.Kvaerno5(),
                t0=t, t1=t+dt, dt0=dt/2, y0=h1, args=env,
                stepsize_controller=pid, saveat=diffrax.SaveAt(t1=True),
                max_steps=max_steps)
    h2    = sol2.ys[-1]
    sr    = int(sol2.stats["num_accepted_steps"])
    rej2  = int(sol2.stats["num_rejected_steps"])

    sol3  = diffrax.diffeqsolve(term_E, diffrax.Tsit5(),
                t0=t+dt/2, t1=t+dt, dt0=dt/20, y0=h2, args=ga,
                stepsize_controller=pid, saveat=diffrax.SaveAt(t1=True),
                max_steps=max_steps)
    h_out = sol3.ys[-1]
    sd2   = int(sol3.stats["num_accepted_steps"])
    rej3  = int(sol3.stats["num_rejected_steps"])

    return h_out, sd1+sd2, sr, rej1+rej2+rej3


def imex_integrate(h0, diff_fn, rxn_fn, ga, env, dt=DT, n_steps=N_MACRO):
    h = h0; sd = sr = rej = 0
    t0 = time.perf_counter()
    for k in range(n_steps):
        h, d, r, rj = strang_macro_step(h, k*dt, dt, diff_fn, rxn_fn, ga, env)
        sd += d; sr += r; rej += rj
    wall = time.perf_counter() - t0
    nfe_d = sd * NFE_MUL["tsit5"]
    nfe_r = sr * NFE_MUL["kvaerno5"]
    return {"h_T": h, "steps_diff": sd, "steps_rxn": sr,
            "nfe_diff": nfe_d, "nfe_rxn": nfe_r,
            "nfe_imex_total": nfe_d+nfe_r, "rejected_imex": rej, "wall_imex": wall}


# =============================================================================
# SECTION 8 — DOPRI5 baseline
# =============================================================================

def dopri5_integrate(h0, diff_fn, rxn_fn, ga, env,
                     dt=DT, n_steps=N_MACRO, max_steps=100_000):
    def rhs(t, h, args): return diff_fn(t, h, args[0]) + rxn_fn(t, h, args[1])
    term = diffrax.ODETerm(rhs); pid = _pid()
    h = h0; steps = rej = 0
    t0 = time.perf_counter()
    for k in range(n_steps):
        sol = diffrax.diffeqsolve(term, diffrax.Dopri5(),
                  t0=k*dt, t1=(k+1)*dt, dt0=dt/10, y0=h,
                  args=(ga, env), stepsize_controller=pid,
                  saveat=diffrax.SaveAt(t1=True), max_steps=max_steps)
        h     = sol.ys[-1]
        steps += int(sol.stats["num_accepted_steps"])
        rej   += int(sol.stats["num_rejected_steps"])
    wall = time.perf_counter() - t0
    nfe  = steps * NFE_MUL["dopri5"]
    return {"h_T": h, "steps_dopri5": steps, "nfe_dopri5": nfe,
            "rejected_dopri5": rej, "wall_dopri5": wall}


# =============================================================================
# SECTION 9 — SEVIR data loading
# =============================================================================

def get_local_path(catalog_filename, data_root):
    p1 = os.path.join(data_root, catalog_filename)
    if os.path.exists(p1): return p1
    parts = catalog_filename.replace("\\", "/").split("/")
    if len(parts) == 3:
        p2 = os.path.join(data_root, parts[1], parts[0], parts[2])
        if os.path.exists(p2): return p2
    return None


def load_sevir_event(event_id, catalog, data_root, img_types=("vil","ir069","ir107")):
    channels = {}
    for imt in img_types:
        rows = catalog[(catalog["id"]==event_id) & (catalog["img_type"]==imt)]
        if rows.empty: continue
        fp = get_local_path(str(rows.iloc[0]["file_name"]), data_root)
        if fp is None: continue
        try:
            with h5py.File(fp, "r") as f:
                if "id" not in f or imt not in f: continue
                ids = [x.decode() if isinstance(x,bytes) else str(x) for x in f["id"][:]]
                if event_id not in ids: continue
                idx  = ids.index(event_id)
                data = f[imt][idx].astype(np.float32)
                if data.ndim == 3 and data.shape[2] < data.shape[0]:
                    data = data.transpose(2,0,1)
                channels[imt] = data
        except Exception as e:
            print(f"  [warn] {imt}/{event_id}: {e}")
    return channels


def prepare_frames(channels):
    """Stack channels into (T, H, W, C) float32, normalised per channel."""
    vil = channels.get("vil")
    if vil is None: return None
    T, H, W = vil.shape
    zero = np.zeros((T,H,W), dtype=np.float32)
    def n(x): return ((x-x.mean())/(x.std()+1e-6)).astype(np.float32)
    return np.stack([n(vil),
                     n(channels["ir069"]) if "ir069" in channels else zero,
                     n(channels["ir107"]) if "ir107" in channels else zero],
                    axis=-1)  # (T, H, W, 3)


# =============================================================================
# SECTION 10 — Per-event benchmark
# =============================================================================

def run_event_benchmark(event_id, lc_class, catalog, data_root,
                        N_values, n_seeds=N_SEEDS, jit_warmup=True):
    channels = load_sevir_event(event_id, catalog, data_root)
    frames   = prepare_frames(channels)
    if frames is None:
        print(f"  [skip] {event_id}: VIL missing"); return []

    scale   = STIFFNESS_SCALE.get(lc_class, 1.0)
    results = []

    if jit_warmup:
        _warmup(frames, scale)

    for N_req in N_values:
        try:
            slic_out = temporal_slic_pipeline(frames, N_req)
        except Exception as e:
            print(f"  [skip] SLIC N={N_req}: {e}"); continue

        g        = build_lagrangian_graph(slic_out[0]["centroids_px"],
                                          slic_out[0]["h_state"])
        actual_K = g["actual_K"]

        for seed in range(n_seeds):
            kd   = jax.random.PRNGKey(seed*100)
            kr   = jax.random.PRNGKey(seed*100+1)
            diff = ExplicitDiffusion(key=kd)
            rxn  = ImplicitReaction(stiffness_scale=scale, key=kr)

            imex_r  = imex_integrate(g["h0"], diff, rxn, g["graph_args"], g["env"])
            dopri_r = dopri5_integrate(g["h0"], diff, rxn, g["graph_args"], g["env"])

            l2g = float(jnp.linalg.norm(imex_r["h_T"]-dopri_r["h_T"])
                        / (jnp.linalg.norm(dopri_r["h_T"])+1e-8))

            results.append({
                "event_id": event_id, "lifecycle_class": lc_class,
                "N_requested": N_req, "actual_N": actual_K,
                "E_super": g["E_super"], "seed": seed,
                "stiffness_scale": scale, "k_decay": float(scale*0.3),
                # IMEX
                "steps_diff": imex_r["steps_diff"], "steps_rxn": imex_r["steps_rxn"],
                "nfe_diff": imex_r["nfe_diff"], "nfe_rxn": imex_r["nfe_rxn"],
                "nfe_imex_total": imex_r["nfe_imex_total"],
                "rej_imex": imex_r["rejected_imex"], "wall_imex": imex_r["wall_imex"],
                # DOPRI5
                "steps_dopri5": dopri_r["steps_dopri5"], "nfe_dopri5": dopri_r["nfe_dopri5"],
                "rej_dopri5": dopri_r["rejected_dopri5"], "wall_dopri5": dopri_r["wall_dopri5"],
                # comparison
                "l2_gap": l2g,
                "nfe_ratio":  dopri_r["nfe_dopri5"] / max(imex_r["nfe_imex_total"],1),
                "wall_ratio": dopri_r["wall_dopri5"] / max(imex_r["wall_imex"],1e-9),
            })
            print(
                f"    N={N_req:5d}({actual_K}) seed={seed} | "
                f"IMEX diff={imex_r['nfe_diff']:5d} rxn={imex_r['nfe_rxn']:5d} "
                f"tot={imex_r['nfe_imex_total']:5d} | "
                f"DOPRI5={dopri_r['nfe_dopri5']:5d} | "
                f"ratio={dopri_r['nfe_dopri5']/max(imex_r['nfe_imex_total'],1):.2f}x"
            )
    return results


def _warmup(frames, scale):
    try:
        s = temporal_slic_pipeline(frames, 100)
        g = build_lagrangian_graph(s[0]["centroids_px"], s[0]["h_state"])
        d = ExplicitDiffusion(); r = ImplicitReaction(stiffness_scale=scale)
        imex_integrate(g["h0"], d, r, g["graph_args"], g["env"], n_steps=2)
        dopri5_integrate(g["h0"], d, r, g["graph_args"], g["env"], n_steps=2)
        print("  [warmup] JIT compilation done.")
    except Exception as e:
        print(f"  [warmup] {e}")


# =============================================================================
# SECTION 11 — Main loop
# =============================================================================

def _save(records, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def run_benchmark(args):
    print(f"\nJAX devices : {jax.devices()}")
    print(f"Backend     : {jax.default_backend()}\n")

    ev_cat  = pd.read_csv(args.catalogue, low_memory=False)
    raw_cat = pd.read_csv(args.raw_catalog, low_memory=False)
    print(f"Events loaded: {len(ev_cat)}\n{ev_cat['lifecycle_class'].value_counts()}\n")

    sampled = (ev_cat
               .groupby("lifecycle_class", group_keys=False)
               .apply(lambda g: g.sample(min(args.n_per_class, len(g)), random_state=42))
               .reset_index(drop=True))
    print(f"Sampled {len(sampled)} events ({args.n_per_class}/class)\n")

    all_results = []; jit_done = False
    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Events"):
        eid = str(row["id"]); cls = str(row["lifecycle_class"])
        print(f"\n── {eid}  [{cls}]")
        res = run_event_benchmark(eid, cls, raw_cat, args.data_root,
                                  args.n_values, args.n_seeds,
                                  jit_warmup=not jit_done)
        jit_done = True
        all_results.extend(res)
        if all_results: _save(all_results, args.out)

    _save(all_results, args.out)
    print(f"\n✓ {args.out}  ({len(all_results)} rows)")
    return pd.DataFrame(all_results)


# =============================================================================
# SECTION 12 — Plots
# =============================================================================

CLASS_COLOR = {
    "RAPID_GROWTH":"#d62728","GROWTH_DECAY":"#ff7f0e","EPISODIC":"#9467bd",
    "PLATEAU":"#2ca02c","RAPID_DECAY":"#8c564b","STEADY":"#1f77b4","QUIESCENT":"#7f7f7f",
}


def plot_results(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    agg = (df.groupby(["lifecycle_class","N_requested"])
             .agg(nfe_diff_mean=("nfe_diff","mean"), nfe_diff_std=("nfe_diff","std"),
                  nfe_rxn_mean=("nfe_rxn","mean"),
                  nfe_imex_mean=("nfe_imex_total","mean"), nfe_imex_std=("nfe_imex_total","std"),
                  nfe_dopri_mean=("nfe_dopri5","mean"), nfe_dopri_std=("nfe_dopri5","std"),
                  nfe_ratio_mean=("nfe_ratio","mean"),
                  wall_imex_mean=("wall_imex","mean"), wall_dopri_mean=("wall_dopri5","mean"))
             .reset_index())
    agg["rxn_frac"] = agg["nfe_rxn_mean"] / (agg["nfe_imex_mean"]+1e-9)

    classes = [c for c in LIFECYCLE_ORDER if c in agg["lifecycle_class"].unique()]
    cols    = min(4, len(classes)); rows = int(np.ceil(len(classes)/cols))

    # Fig 1: NFE stacked + DOPRI5 per class
    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4*rows), squeeze=False)
    for i, cls in enumerate(classes):
        ax  = axes.flatten()[i]
        sub = agg[agg["lifecycle_class"]==cls].sort_values("N_requested")
        Ns  = sub["N_requested"].values
        ax.stackplot(Ns, sub["nfe_diff_mean"], sub["nfe_rxn_mean"],
                     labels=["IMEX diffusion","IMEX reaction"],
                     colors=["steelblue","tomato"], alpha=0.75)
        ax.plot(Ns, sub["nfe_imex_mean"], color=CLASS_COLOR.get(cls,"k"), lw=2, label="IMEX total")
        ax.plot(Ns, sub["nfe_dopri_mean"], "k--", lw=2, label="DOPRI5")
        ax.fill_between(Ns, sub["nfe_dopri_mean"]-sub["nfe_dopri_std"],
                        sub["nfe_dopri_mean"]+sub["nfe_dopri_std"], alpha=0.10, color="k")
        ax.axvline(500,  color="grey",   ls=":",  lw=1.3, label="ISV elbow")
        ax.axvline(1150, color="purple", ls="-.", lw=1.3, label="N*=1150")
        ax.set_title(cls, fontweight="bold", fontsize=9)
        ax.set_xlabel("N"); ax.set_ylabel("NFE")
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    for j in range(i+1, len(axes.flatten())): axes.flatten()[j].set_visible(False)
    fig.suptitle("Block A — IMEX diffusion + reaction vs DOPRI5 NFE", weight="bold")
    fig.tight_layout(); p=os.path.join(out_dir,"block_a_nfe_vs_N.pdf")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  {p}")

    # Fig 2: NFE ratio heatmap
    pivot = (agg.pivot(index="lifecycle_class", columns="N_requested", values="nfe_ratio_mean")
               .reindex(classes))
    fig2, ax2 = plt.subplots(figsize=(9,3.5))
    im = ax2.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=6.0)
    Nc = list(pivot.columns)
    ax2.set_xticks(range(len(Nc))); ax2.set_xticklabels(Nc, rotation=45)
    ax2.set_yticks(range(len(classes))); ax2.set_yticklabels(classes, fontsize=8)
    if 500  in Nc: ax2.axvline(Nc.index(500),  color="grey",   ls=":",  lw=1.5)
    if 1150 in Nc: ax2.axvline(Nc.index(1150), color="purple", ls="-.", lw=1.5)
    for r in range(len(classes)):
        for ci in range(len(Nc)):
            v = pivot.values[r,ci]
            if not np.isnan(v): ax2.text(ci,r,f"{v:.1f}",ha="center",va="center",fontsize=7)
    plt.colorbar(im, ax=ax2, label="NFE ratio DOPRI5/IMEX (>1 = IMEX cheaper)")
    ax2.set_title("NFE ratio heat map", fontsize=9)
    fig2.tight_layout(); p2=os.path.join(out_dir,"block_a_ratio_heatmap.pdf")
    fig2.savefig(p2,dpi=150,bbox_inches="tight"); plt.close(fig2); print(f"  {p2}")

    # Fig 3: reaction fraction
    fig3, ax3 = plt.subplots(figsize=(7,3.5))
    for cls in classes:
        sub = agg[agg["lifecycle_class"]==cls].sort_values("N_requested")
        ax3.plot(sub["N_requested"], sub["rxn_frac"],
                 color=CLASS_COLOR.get(cls,"k"), marker="o", ms=4, lw=1.5, label=cls)
    ax3.axvline(500,  color="grey",  ls=":",  lw=1.3)
    ax3.axvline(1150, color="purple",ls="-.", lw=1.3)
    ax3.set_xlabel("N"); ax3.set_ylabel("NFE_rxn / NFE_imex_total")
    ax3.set_title("Reaction fraction of IMEX NFE\n"
                  "(high -> reaction dominates; low -> diffusion/CFL dominates)", fontsize=9)
    ax3.legend(fontsize=7, ncol=2); ax3.grid(alpha=0.3)
    fig3.tight_layout(); p3=os.path.join(out_dir,"block_a_rxn_fraction.pdf")
    fig3.savefig(p3,dpi=150,bbox_inches="tight"); plt.close(fig3); print(f"  {p3}")

    tbl = agg[agg["N_requested"].isin([500,1150])][
        ["lifecycle_class","N_requested","nfe_diff_mean","nfe_rxn_mean",
         "nfe_imex_mean","nfe_dopri_mean","nfe_ratio_mean",
         "wall_imex_mean","wall_dopri_mean"]].round(1)
    tbl.to_csv(os.path.join(out_dir,"block_a_summary.csv"), index=False)
    print("\n── Summary at N=500 and N=1150 ──")
    print(tbl.to_string(index=False))


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--catalogue",   default=DEFAULT_CAT)
    p.add_argument("--raw_catalog", default=CATALOG_PATH)
    p.add_argument("--data_root",   default=DATA_ROOT)
    p.add_argument("--n_per_class", type=int,       default=N_PER_CLASS)
    p.add_argument("--n_seeds",     type=int,       default=N_SEEDS)
    p.add_argument("--n_values",    type=int, nargs="+", default=N_VALUES)
    p.add_argument("--out",         default=DEFAULT_OUT)
    p.add_argument("--plot_only",   action="store_true")
    args = p.parse_args()

    if args.plot_only:
        df = pd.read_csv(args.out)
        print(f"Loaded {len(df)} rows from {args.out}")
    else:
        df = run_benchmark(args)

    plot_results(df, str(Path(args.out).parent))
    print("\nDone.")


if __name__ == "__main__":
    main()
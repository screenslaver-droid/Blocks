"""
block_a_hpc.py  —  DRIFT Block A: Empirical IMEX vs DOPRI5 Solver Benchmark
=============================================================================
Production script for PARAM Rudra (A100 × 2, 80 GB HBM2e per GPU).

What is benchmarked
-------------------
IMEX (Strang split: Tsit5 diffusion + Kvaerno5 reaction) vs DOPRI5 (unsplit)
on the Lagrangian superpixel graph built from real SEVIR events.

NFE breakdown reported per integration:
  nfe_diff    — diffusion half-steps  (Tsit5, explicit, CFL-bound)
  nfe_rxn     — reaction full-step    (Kvaerno5, implicit, L-stable)
  nfe_dopri5  — combined unsplit      (DOPRI5, stability-bound for stiff Rx)

GPU acceleration
----------------
  • JAX backend: all ODE integration and SLIC distance/assignment/update ops
    run on CUDA via XLA. Falls back to CPU transparently.
  • TemporalSLIC: fully JAX-driven SLIC iterations for every frame.
    Centroid advection uses a JAX phase-correlation optical flow approximation.

Usage
-----
    python block_a_hpc.py                          # uses defaults below
    python block_a_hpc.py --n_per_class 5 \\
        --n_values 250 500 750 1000 1150 1500 \\
        --out $SCRATCH/results/block_a_full.csv
    python block_a_hpc.py --plot_only \\
        --out $SCRATCH/results/block_a_full.csv

Dependencies
------------
    jax[cuda12]  equinox  diffrax  h5py  pandas  numpy
    scipy  scikit-image  matplotlib  tqdm
"""

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — Environment (must be set BEFORE jax import)
# ─────────────────────────────────────────────────────────────────────────────
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE",   "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION",  "0.75")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",            "3")

import argparse
import logging
import time
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from skimage.segmentation import slic as cpu_slic   # fallback only
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ── Edit these for your PARAM Rudra paths ────────────────────────────────────
DATA_ROOT    = "/path/to/sevir"            # e.g. /scratch/username/SEVIR
CATALOG_PATH = os.path.join(DATA_ROOT, "CATALOG.csv")
DEFAULT_CAT  = os.path.join(DATA_ROOT, "event_catalogue.csv")
DEFAULT_OUT  = "results/block_a_full.csv"

# ── Sweep ─────────────────────────────────────────────────────────────────────
N_VALUES     = [250, 500, 750, 1000, 1150, 1500]   # N=500: ISV elbow; N=1150: global N*
N_PER_CLASS  = 5
N_SEEDS      = 5
DT_SECONDS   = 300.0       # one 5-minute SEVIR frame
N_MACRO      = 12          # 12 × 5 min = 60-min forecast window
TOL          = 1e-2        # rtol; atol = TOL/10

# ── Architecture ──────────────────────────────────────────────────────────────
NODE_DIM   = 3    # VIL, IR069, IR107
ENV_DIM    = 4    # x_norm, y_norm, mean_VIL, mean_IR069
HIDDEN_DIM = 64
R_MAX      = 108  # px: v_max(9 px/frame) × T_fc(12 frames)  Table B.1
SIGMA_RBF  = 15.0 # px: RBF bandwidth for edge weights

# ── Per-class stiffness scale (from SRbase, Table 5.6) ───────────────────────
# k_decay = scale × BASE_RATE [s⁻¹]; stiffness ratio SR = k_decay × DT_SECONDS
BASE_RATE = 0.3
STIFFNESS_SCALE = {
    "RAPID_GROWTH": 4.0,    # SR ≈ 360
    "GROWTH_DECAY": 3.0,    # SR ≈ 270
    "EPISODIC":     2.0,    # SR ≈ 180
    "PLATEAU":      1.5,    # SR ≈ 135
    "RAPID_DECAY":  1.5,    # SR ≈ 135
    "STEADY":       1.0,    # SR ≈  90
    "QUIESCENT":    0.5,    # SR ≈  45
}

# ── NFE stage multipliers ─────────────────────────────────────────────────────
# Accepted solver steps × stages per step = estimated RHS evaluations
NFE_MUL = {"tsit5": 5, "kvaerno5": 7, "dopri5": 6}

# ── SLIC ──────────────────────────────────────────────────────────────────────
SLIC_KAPPA  = 10.0
SLIC_ITER   = 10
# Chunk size for SLIC distance matrix to limit GPU memory use.
# None → full matrix (fine on A100 80 GB for N ≤ 2000).
SLIC_CHUNK  = int(os.environ.get("SLIC_CHUNK", "0")) or None

# ── Lifecycle display order ───────────────────────────────────────────────────
LIFECYCLE_ORDER = [
    "RAPID_GROWTH", "GROWTH_DECAY", "EPISODIC",
    "PLATEAU", "RAPID_DECAY", "STEADY", "QUIESCENT",
]
CLASS_COLOR = {
    "RAPID_GROWTH": "#d62728", "GROWTH_DECAY": "#ff7f0e",
    "EPISODIC":     "#9467bd", "PLATEAU":      "#2ca02c",
    "RAPID_DECAY":  "#8c564b", "STEADY":       "#1f77b4",
    "QUIESCENT":    "#7f7f7f",
}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("block_a")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SEVIR data loading  (mirrors Growth_Decay_Classify.py)
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_path(catalog_filename: str, data_root: str) -> str | None:
    p1 = os.path.join(data_root, catalog_filename)
    if os.path.exists(p1):
        return p1
    parts = catalog_filename.replace("\\", "/").split("/")
    if len(parts) == 3:
        p2 = os.path.join(data_root, parts[1], parts[0], parts[2])
        if os.path.exists(p2):
            return p2
    return None


def load_channels(
    event_id: str,
    catalog: pd.DataFrame,
    data_root: str,
    img_types: tuple[str, ...] = ("vil", "ir069", "ir107"),
) -> dict[str, np.ndarray]:
    """Return {img_type: (T, H, W) float32} for the requested event."""
    out = {}
    for img_type in img_types:
        rows = catalog[
            (catalog["id"] == event_id) & (catalog["img_type"] == img_type)
        ]
        if rows.empty:
            continue
        fpath = _resolve_path(str(rows.iloc[0]["file_name"]), data_root)
        if fpath is None:
            continue
        try:
            with h5py.File(fpath, "r") as f:
                if "id" not in f or img_type not in f:
                    continue
                ids = [
                    x.decode() if isinstance(x, bytes) else str(x)
                    for x in f["id"][:]
                ]
                if event_id not in ids:
                    continue
                idx  = ids.index(event_id)
                data = f[img_type][idx].astype(np.float32)
                if data.ndim == 3 and data.shape[2] < data.shape[0]:
                    data = data.transpose(2, 0, 1)  # → (T, H, W)
                out[img_type] = data
        except Exception as exc:
            log.warning(f"  Could not load {img_type}/{event_id}: {exc}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — GPU-driven Temporal SLIC
# ─────────────────────────────────────────────────────────────────────────────

class TemporalSLIC:
    """
    JAX-accelerated Temporal SLIC superpixel segmentation.

    All inner-loop work (distance computation, assignment, centroid update)
    runs on the active JAX device (CUDA on A100, CPU otherwise). Python-level
    for-loops handle frame iteration and SLIC outer iterations; each JAX
    operation inside is JIT-compiled on first call.

    Frame 0:  Standard GPU SLIC on the fused multi-channel feature cube.
    Frame t>0 (temporal propagation):
        1. Phase-correlation optical flow (JAX FFT, GPU) between consecutive
           VIL frames → per-centroid displacement Δc.
        2. Advect centroids:  c̃_t = c_{t-1} + Δc
        3. Warm-start GPU SLIC from advected centroids instead of grid init.
           Equivalent to a soft watershed from the advected seeds.

    Parameters
    ----------
    N       : target superpixel count
    kappa   : SLIC compactness (report Table B.1: 10)
    n_iter  : SLIC inner iterations per frame
    chunk   : pixel chunk size for distance computation (None = full matrix)
    """

    def __init__(
        self,
        N: int,
        kappa: float = SLIC_KAPPA,
        n_iter: int  = SLIC_ITER,
        chunk: int | None = SLIC_CHUNK,
    ):
        self.N      = N
        self.kappa  = kappa
        self.n_iter = n_iter
        self.chunk  = chunk

    # ── public API ────────────────────────────────────────────────────────────

    def fit_sequence(
        self, channels: dict[str, np.ndarray]
    ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
        """
        Segment all T frames.

        Returns
        -------
        labels_list : list of T × (H, W) int32  label maps
        h_all       : (T, N, 3)  node states [VIL, IR069, IR107]
        centroids_t0: (N, 2)     pixel centroids [x, y] at t=0
        """
        vil   = channels.get("vil")
        ir069 = channels.get("ir069")
        ir107 = channels.get("ir107")
        T, H, W = vil.shape

        def _norm(x: np.ndarray) -> np.ndarray:
            return (x - x.mean()) / (x.std() + 1e-6)

        # ── Frame 0: standard GPU SLIC ────────────────────────────────────────
        feat0 = np.stack([
            _norm(vil[0]),
            _norm(ir069[0]) if ir069 is not None else np.zeros((H, W)),
            _norm(ir107[0]) if ir107 is not None else np.zeros((H, W)),
        ], axis=-1).astype(np.float32)  # (H, W, 3)

        labels0, centroids = self._gpu_slic(feat0, init_centroids=None)
        actual_N = int(labels0.max()) + 1
        # Adjust self.N to actual (SLIC may produce slightly different count)
        self.N = actual_N

        labels_list = [labels0]
        h_all       = [self._node_states(labels0, vil[0], ir069[0] if ir069 is not None else None,
                                         ir107[0] if ir107 is not None else None)]

        # ── Frames t > 0: temporal propagation ───────────────────────────────
        for t in range(1, T):
            # 1. Phase-correlation flow between consecutive VIL frames
            delta_xy = self._phase_corr_flow(
                vil[t-1], vil[t], centroids, patch=32
            )  # (actual_N, 2)  displacement [Δx, Δy] in pixels

            # 2. Advect centroids
            adv_centroids = centroids + delta_xy   # (actual_N, 2)
            adv_centroids[:, 0] = np.clip(adv_centroids[:, 0], 0, W - 1)
            adv_centroids[:, 1] = np.clip(adv_centroids[:, 1], 0, H - 1)

            # 3. Warm-start GPU SLIC from advected centroids
            feat_t = np.stack([
                _norm(vil[t]),
                _norm(ir069[t]) if ir069 is not None else np.zeros((H, W)),
                _norm(ir107[t]) if ir107 is not None else np.zeros((H, W)),
            ], axis=-1).astype(np.float32)

            labels_t, centroids = self._gpu_slic(feat_t, init_centroids=adv_centroids)
            labels_list.append(labels_t)
            h_all.append(self._node_states(labels_t, vil[t],
                                           ir069[t] if ir069 is not None else None,
                                           ir107[t] if ir107 is not None else None))

        return labels_list, np.stack(h_all, axis=0), labels_list[0], centroids

    def fit_t0(
        self, channels: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment first frame only (Block A uses static graph from t=0).

        Returns
        -------
        labels    : (H, W) int32
        h0        : (N, 3)  node states at t=0
        centroids : (N, 2)  pixel centroids [x, y]
        """
        vil   = channels["vil"]
        ir069 = channels.get("ir069")
        ir107 = channels.get("ir107")
        H, W  = vil.shape[1], vil.shape[2]

        def _norm(x): return (x - x.mean()) / (x.std() + 1e-6)

        feat0 = np.stack([
            _norm(vil[0]),
            _norm(ir069[0]) if ir069 is not None else np.zeros((H, W)),
            _norm(ir107[0]) if ir107 is not None else np.zeros((H, W)),
        ], axis=-1).astype(np.float32)

        labels, centroids = self._gpu_slic(feat0, init_centroids=None)
        self.N = int(labels.max()) + 1
        h0 = self._node_states(labels, vil[0],
                               ir069[0] if ir069 is not None else None,
                               ir107[0] if ir107 is not None else None)
        return labels, h0, centroids

    # ── internals ─────────────────────────────────────────────────────────────

    def _gpu_slic(
        self,
        feat_HWC: np.ndarray,
        init_centroids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Core GPU SLIC loop (JAX).

        Parameters
        ----------
        feat_HWC       : (H, W, C) float32  normalized spectral features
        init_centroids : (N, 2) float32 pixel coords [x, y], or None for grid init

        Returns
        -------
        labels    : (H, W) int32  (numpy, CPU)
        centroids : (N, 2) float32 pixel coords [x, y]  (numpy, CPU)
        """
        H, W, C = feat_HWC.shape
        S       = float(np.sqrt(H * W / self.N))
        kappa   = self.kappa

        # ── Build augmented pixel features (GPU) ─────────────────────────────
        # Augmented space: [spectral / kappa, x / S, y / S]
        # Distance: d² = d_spectral² / kappa² + d_spatial² / S²
        # Normalise so both terms are on comparable scales when kappa ≈ S.
        ys = np.arange(H, dtype=np.float32) / S
        xs = np.arange(W, dtype=np.float32) / S
        yy, xx = np.meshgrid(ys, xs, indexing="ij")   # (H, W)

        pixels_np = np.concatenate([
            feat_HWC.reshape(H * W, C) / kappa,       # spectral scaled by 1/kappa
            xx.reshape(H * W, 1),                      # x / S
            yy.reshape(H * W, 1),                      # y / S
        ], axis=-1).astype(np.float32)                 # (H*W, C+2)

        pixels = jnp.array(pixels_np)   # move to GPU

        # ── Initialise centroids ──────────────────────────────────────────────
        if init_centroids is not None:
            # Warm start from advected centroids
            cx = np.clip(init_centroids[:, 0].astype(int), 0, W - 1)
            cy = np.clip(init_centroids[:, 1].astype(int), 0, H - 1)
            px_idx = cy * W + cx
            cents_np = pixels_np[px_idx]  # (N, C+2)
        else:
            cents_np = self._grid_init(pixels_np, H, W, C, S)

        N_actual = cents_np.shape[0]
        centroids = jnp.array(cents_np)   # GPU

        # ── SLIC iterations (GPU inner ops) ──────────────────────────────────
        for _ in range(self.n_iter):
            labels     = self._assign(pixels, centroids, N_actual)   # (H*W,) GPU
            centroids  = self._update(pixels, labels, N_actual)       # (N, C+2) GPU

        # ── Recover pixel centroids [x, y] from spatial components ───────────
        labels_np   = np.array(labels).reshape(H, W).astype(np.int32)
        centroids_np = np.array(centroids)                            # (N, C+2)
        # Spatial components are stored at indices C and C+1, scaled by 1/S
        cent_x = centroids_np[:, C]   * S   # back to pixel coords
        cent_y = centroids_np[:, C+1] * S
        centroids_px = np.stack([cent_x, cent_y], axis=-1)  # (N, 2) [x, y]

        return labels_np, centroids_px.astype(np.float32)

    def _assign(
        self, pixels: jax.Array, centroids: jax.Array, N: int
    ) -> jax.Array:
        """
        Assign each pixel to its nearest centroid.  GPU-parallelised.
        Uses chunked matrix-multiply to cap working memory.
        """
        HW = pixels.shape[0]
        chunk = self.chunk or HW          # None → full matrix

        if chunk >= HW:
            # Full distance matrix in one shot: (HW, N) × float32
            dists = jnp.sum(
                (pixels[:, None, :] - centroids[None, :, :]) ** 2,
                axis=-1,
            )   # (HW, N)
            return jnp.argmin(dists, axis=-1).astype(jnp.int32)

        # Chunked: (chunk, N) working memory per batch
        chunks = []
        for start in range(0, HW, chunk):
            blk   = pixels[start : start + chunk]
            dists = jnp.sum(
                (blk[:, None, :] - centroids[None, :, :]) ** 2, axis=-1
            )
            chunks.append(jnp.argmin(dists, axis=-1))
        return jnp.concatenate(chunks, axis=0).astype(jnp.int32)

    @staticmethod
    def _update(
        pixels: jax.Array, labels: jax.Array, N: int
    ) -> jax.Array:
        """
        Update centroids as mean of assigned pixels.  GPU scatter-add.
        """
        D      = pixels.shape[1]
        counts = jnp.zeros(N).at[labels].add(1.0)
        counts = jnp.maximum(counts, 1.0)            # avoid /0 for empty segments
        sums   = jnp.zeros((N, D)).at[labels].add(pixels)
        return sums / counts[:, None]

    @staticmethod
    def _grid_init(
        pixels_np: np.ndarray, H: int, W: int, C: int, S: float
    ) -> np.ndarray:
        """Initialise centroids on a regular spatial grid (NumPy, CPU)."""
        n_gy = max(1, int(round(H / S)))
        n_gx = max(1, int(round(W / S)))
        gy   = np.linspace(S / 2, H - S / 2, n_gy, dtype=np.float32)
        gx   = np.linspace(S / 2, W - S / 2, n_gx, dtype=np.float32)
        gyy, gxx = np.meshgrid(gy / S, gx / S, indexing="ij")  # normalised
        grid     = np.stack([gxx.ravel(), gyy.ravel()], axis=-1)   # (n_gy×n_gx, 2)

        n_init = len(grid)
        N_req  = int(pixels_np.shape[0] ** 0  )   # placeholder; use len later
        # Trim / pad to nominal N (caller's self.N)
        N_req  = pixels_np.shape[0]  # will be trimmed after
        # Return first N rows (N determined by caller)
        grid_pts = grid                            # (n_gy*n_gx, 2)

        # Map normalised [x/S, y/S] back to pixel indices for feature look-up
        cx = np.clip((grid_pts[:, 0] * S).astype(int), 0, W - 1)
        cy = np.clip((grid_pts[:, 1] * S).astype(int), 0, H - 1)
        px_idx   = cy * W + cx                    # (n_init,)
        cents_np = pixels_np[px_idx]              # (n_init, C+2)

        # Trim / pad to self-consistent count
        return cents_np

    def _grid_init_n(
        self, pixels_np: np.ndarray, H: int, W: int, C: int, S: float
    ) -> np.ndarray:
        """Grid init returning exactly self.N centroids."""
        n_gy = max(1, int(round(H / S)))
        n_gx = max(1, int(round(W / S)))
        gy   = np.linspace(S / 2, H - S / 2, n_gy, dtype=np.float32)
        gx   = np.linspace(S / 2, W - S / 2, n_gx, dtype=np.float32)
        gyy, gxx = np.meshgrid(gy / S, gx / S, indexing="ij")
        grid_pts = np.stack([gxx.ravel(), gyy.ravel()], axis=-1)   # (K, 2)

        K = len(grid_pts)
        if K >= self.N:
            idx      = np.round(np.linspace(0, K - 1, self.N)).astype(int)
            grid_pts = grid_pts[idx]
        else:
            reps     = (self.N + K - 1) // K
            grid_pts = np.tile(grid_pts, (reps, 1))[: self.N]

        cx = np.clip((grid_pts[:, 0] * S).astype(int), 0, W - 1)
        cy = np.clip((grid_pts[:, 1] * S).astype(int), 0, H - 1)
        return pixels_np[cy * W + cx].astype(np.float32)   # (N, C+2)

    @staticmethod
    def _phase_corr_flow(
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        centroids: np.ndarray,
        patch: int = 32,
    ) -> np.ndarray:
        """
        Estimate per-centroid optical flow via JAX phase-correlation (GPU FFT).

        For each centroid, extract a (patch × patch) window from both frames,
        compute the normalised cross-power spectrum, and read off the integer
        shift from the peak.  Sub-pixel accuracy is not needed for centroid
        advection.

        Parameters
        ----------
        frame_a, frame_b : (H, W) float32
        centroids        : (N, 2) pixel coords [x, y]
        patch            : window size (must be a power of 2 for efficiency)

        Returns
        -------
        delta_xy : (N, 2) float32  displacement [Δx, Δy] in pixels
        """
        H, W  = frame_a.shape
        half  = patch // 2
        N     = centroids.shape[0]

        fa_j = jnp.array(frame_a)
        fb_j = jnp.array(frame_b)

        def centroid_flow(cx_cy):
            cx, cy = cx_cy[0], cx_cy[1]
            # Clamp window to image bounds using integer pixel coords
            x0 = jnp.clip(jnp.int32(cx) - half, 0, W - patch)
            y0 = jnp.clip(jnp.int32(cy) - half, 0, H - patch)

            patch_a = jax.lax.dynamic_slice(fa_j, (y0, x0), (patch, patch))
            patch_b = jax.lax.dynamic_slice(fb_j, (y0, x0), (patch, patch))

            # Phase correlation: F⁻¹{ Fa · conj(Fb) / |Fa · conj(Fb)| }
            Fa  = jnp.fft.fft2(patch_a)
            Fb  = jnp.fft.fft2(patch_b)
            R   = Fa * jnp.conj(Fb)
            R   = R / (jnp.abs(R) + 1e-6)
            r   = jnp.real(jnp.fft.ifft2(R))           # (patch, patch)

            # Peak location → shift
            idx = jnp.argmax(r)
            dy  = (idx // patch).astype(jnp.float32)
            dx  = (idx  % patch).astype(jnp.float32)
            # Wrap-around: shifts > patch/2 are negative displacements
            dy  = jnp.where(dy > half, dy - patch, dy)
            dx  = jnp.where(dx > half, dx - patch, dx)
            return jnp.array([dx, dy])

        cents_jax = jnp.array(centroids, dtype=jnp.float32)  # (N, 2)
        flows     = jax.vmap(centroid_flow)(cents_jax)        # (N, 2) GPU

        return np.array(flows).astype(np.float32)

    @staticmethod
    def _node_states(
        labels: np.ndarray,
        vil:    np.ndarray,
        ir069:  np.ndarray | None,
        ir107:  np.ndarray | None,
    ) -> np.ndarray:
        """Compute area-weighted mean (VIL, IR069, IR107) per superpixel."""
        N   = int(labels.max()) + 1
        h   = np.zeros((N, 3), dtype=np.float32)
        for k in range(N):
            mask    = labels == k
            if not mask.any():
                continue
            h[k, 0] = vil[mask].mean() / 255.0
            h[k, 1] = (ir069[mask].mean() / 255.0) if ir069 is not None else 0.0
            h[k, 2] = (ir107[mask].mean() / 255.0) if ir107 is not None else 0.0
        return h


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph(
    labels: np.ndarray,
    h0:     np.ndarray,
    centroids: np.ndarray,
) -> dict:
    """
    Build the pre-allocated Lagrangian superpixel graph (Section 4.3.2).

    Returns dict with JAX arrays ready for the ODE integrators.
    """
    actual_N = int(labels.max()) + 1
    H, W     = labels.shape

    # ── Static environment: [x_norm, y_norm, mean_VIL, mean_IR069] ───────────
    env = np.stack([
        centroids[:, 0] / W,
        centroids[:, 1] / H,
        h0[:, 0],
        h0[:, 1],
    ], axis=-1).astype(np.float32)   # (N, ENV_DIM)

    # ── Edge superset: all pairs within R_MAX (Eq. 4.6) ──────────────────────
    tree  = KDTree(centroids)
    pairs = np.array(sorted(tree.query_pairs(R_MAX)), dtype=np.int32)

    if len(pairs) == 0:
        k    = min(8, actual_N - 1)
        _, idx = tree.query(centroids, k=k + 1)
        pairs = np.array(
            [(i, int(idx[i, j])) for i in range(actual_N)
             for j in range(1, k + 1) if i < int(idx[i, j])],
            dtype=np.int32,
        )

    senders   = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.int32)
    receivers = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.int32)

    dist_sq      = np.sum(
        (centroids[senders] - centroids[receivers]) ** 2, axis=1
    )
    edge_weights = np.exp(-dist_sq / (2.0 * SIGMA_RBF ** 2)).astype(np.float32)

    return {
        "actual_N":     actual_N,
        "h0":           jnp.array(h0),
        "env":          jnp.array(env),
        "graph_args":   (jnp.array(senders), jnp.array(receivers), jnp.array(edge_weights)),
        "E_super":      len(senders),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — Neural modules  (Equinox / JAX)
# ─────────────────────────────────────────────────────────────────────────────

class ExplicitDiffusion(eqx.Module):
    """
    Isotropic graph diffusion:  dh_i/dt = D × Σ_j ew_ij (h_j − h_i)

    Scalar D > 0 guarantees stability (all Laplacian eigenvalues ≤ 0).
    Treated explicitly (Tsit5); cost ∝ |E_super| ∝ N².
    CFL:  Δt ≤ C_tsit5 / (D × λ_max(L))  →  step count grows with N.
    """
    log_D: jax.Array   # D = exp(log_D) > 0

    def __init__(self, key=None):   # key kept for API uniformity
        self.log_D = jnp.log(jnp.array(0.05))   # D = 0.05

    def __call__(self, t, h, args):
        snd, rcv, ew = args
        D    = jnp.exp(self.log_D)
        diff = h[rcv] - h[snd]                   # (E, d)
        flux = D * ew[:, None] * diff
        return jnp.zeros_like(h).at[rcv].add(flux)


class ImplicitReaction(eqx.Module):
    """
    Stiff linear decay toward a learned equilibrium:

        f_I(h, env) = f_eq(env) − k_decay × h

    Jacobian ∂f_I/∂h = −k_decay × I  →  eigenvalue = −k_decay (stiff).
    Output is bounded (f_eq ∈ [0, 1]^d, h tracked near [0, 1]).
    Treated implicitly (Kvaerno5, L-stable): handles any k_decay in O(1) steps.

    Stiffness ratio per macro step:  SR = k_decay × DT_SECONDS
        RAPID_GROWTH : k_decay = 1.2 s⁻¹  → SR ≈ 360
        STEADY       : k_decay = 0.3 s⁻¹  → SR ≈  90
        QUIESCENT    : k_decay = 0.15 s⁻¹ → SR ≈  45

    DOPRI5 stability ceiling:  Δt ≤ 3.5 / k_decay
    → min steps per macro step ≈ DT_SECONDS × k_decay / 3.5
    """
    k_decay: float
    eq_w1:   jax.Array   # (NODE_DIM + ENV_DIM, HIDDEN_DIM)
    eq_w2:   jax.Array   # (HIDDEN_DIM, NODE_DIM)

    def __init__(self, stiffness_scale: float = 1.0, key: jax.Array = None):
        if key is None:
            key = jax.random.PRNGKey(1)
        k1, k2  = jax.random.split(key, 2)
        self.k_decay = float(stiffness_scale * BASE_RATE)
        inp = NODE_DIM + ENV_DIM
        self.eq_w1 = jax.random.normal(k1, (inp, HIDDEN_DIM)) * 0.1
        self.eq_w2 = jax.random.normal(k2, (HIDDEN_DIM, NODE_DIM)) * 0.1

    def __call__(self, t, h, env):
        h_env = jnp.concatenate([h, env], axis=-1)            # (N, d+e)
        f_eq  = jax.nn.sigmoid(
            jnp.tanh(h_env @ self.eq_w1) @ self.eq_w2
        )                                                       # (N, d) ∈ [0,1]
        return f_eq - self.k_decay * h                         # Jacobian: −k_decay I


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Integrators
# ─────────────────────────────────────────────────────────────────────────────

def _pid(tol: float = TOL) -> diffrax.PIDController:
    return diffrax.PIDController(rtol=tol, atol=tol * 0.1)


def run_imex(
    h0:         jax.Array,
    diff_fn:    ExplicitDiffusion,
    rxn_fn:     ImplicitReaction,
    graph_args: tuple,
    env:        jax.Array,
    dt:         float = DT_SECONDS,
    n_steps:    int   = N_MACRO,
    max_steps:  int   = 16384,
) -> dict:
    """
    Strang-split IMEX integration over n_steps × dt seconds.

    Each macro step:
        L_D(dt/2) → L_R(dt) → L_D(dt/2)

    Returns per-macro-step NFE traces and totals.
    """
    term_E  = diffrax.ODETerm(lambda t, h, a: diff_fn(t, h, a))
    term_I  = diffrax.ODETerm(lambda t, h, e: rxn_fn(t, h, e))
    pid     = _pid()

    h              = h0
    trace_diff     = []   # (steps_d1 + steps_d2) per macro step
    trace_rxn      = []   # steps_r per macro step
    trace_rej      = []

    t_wall = time.perf_counter()
    for k in range(n_steps):
        t = float(k * dt)

        # ── Half-step diffusion ───────────────────────────────────────────────
        sol = diffrax.diffeqsolve(
            term_E, diffrax.Tsit5(),
            t0=t, t1=t + dt / 2, dt0=dt / 20,
            y0=h, args=graph_args,
            stepsize_controller=pid,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
        )
        h      = sol.ys[-1]
        sd1    = int(sol.stats["num_accepted_steps"])
        rd1    = int(sol.stats["num_rejected_steps"])

        # ── Full-step reaction (implicit) ─────────────────────────────────────
        sol = diffrax.diffeqsolve(
            term_I, diffrax.Kvaerno5(),
            t0=t, t1=t + dt, dt0=dt / 2,
            y0=h, args=env,
            stepsize_controller=pid,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
        )
        h   = sol.ys[-1]
        sr  = int(sol.stats["num_accepted_steps"])
        rr  = int(sol.stats["num_rejected_steps"])

        # ── Second half-step diffusion ────────────────────────────────────────
        sol = diffrax.diffeqsolve(
            term_E, diffrax.Tsit5(),
            t0=t + dt / 2, t1=t + dt, dt0=dt / 20,
            y0=h, args=graph_args,
            stepsize_controller=pid,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
        )
        h      = sol.ys[-1]
        sd2    = int(sol.stats["num_accepted_steps"])
        rd2    = int(sol.stats["num_rejected_steps"])

        trace_diff.append(sd1 + sd2)
        trace_rxn.append(sr)
        trace_rej.append(rd1 + rr + rd2)

    wall = time.perf_counter() - t_wall

    steps_diff  = int(sum(trace_diff))
    steps_rxn   = int(sum(trace_rxn))
    nfe_diff    = steps_diff * NFE_MUL["tsit5"]
    nfe_rxn     = steps_rxn  * NFE_MUL["kvaerno5"]

    return {
        "h_T":            h,
        "steps_diff":     steps_diff,
        "steps_rxn":      steps_rxn,
        "nfe_diff":       nfe_diff,
        "nfe_rxn":        nfe_rxn,
        "nfe_imex_total": nfe_diff + nfe_rxn,
        "rej_imex":       int(sum(trace_rej)),
        "wall_imex":      wall,
        "trace_diff":     trace_diff,
        "trace_rxn":      trace_rxn,
    }


def run_dopri5(
    h0:         jax.Array,
    diff_fn:    ExplicitDiffusion,
    rxn_fn:     ImplicitReaction,
    graph_args: tuple,
    env:        jax.Array,
    dt:         float = DT_SECONDS,
    n_steps:    int   = N_MACRO,
    max_steps:  int   = 100_000,
) -> dict:
    """
    Unsplit DOPRI5 integration — the combined RHS is the same that
    torchdiffeq provides today.
    """
    def rhs(t, h, args):
        ga, env = args
        return diff_fn(t, h, ga) + rxn_fn(t, h, env)

    term = diffrax.ODETerm(rhs)
    pid  = _pid()

    h           = h0
    trace_steps = []
    trace_rej   = []

    t_wall = time.perf_counter()
    for k in range(n_steps):
        t   = float(k * dt)
        sol = diffrax.diffeqsolve(
            term, diffrax.Dopri5(),
            t0=t, t1=t + dt, dt0=dt / 10,
            y0=h, args=(graph_args, env),
            stepsize_controller=pid,
            saveat=diffrax.SaveAt(t1=True),
            max_steps=max_steps,
        )
        h = sol.ys[-1]
        trace_steps.append(int(sol.stats["num_accepted_steps"]))
        trace_rej.append(int(sol.stats["num_rejected_steps"]))
    wall = time.perf_counter() - t_wall

    steps_total = int(sum(trace_steps))
    return {
        "h_T":          h,
        "steps_dopri5": steps_total,
        "nfe_dopri5":   steps_total * NFE_MUL["dopri5"],
        "rej_dopri5":   int(sum(trace_rej)),
        "wall_dopri5":  wall,
        "trace_steps":  trace_steps,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — Per-event benchmark
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_event(
    event_id:   str,
    lc_class:   str,
    catalog:    pd.DataFrame,
    data_root:  str,
    N_values:   list[int],
    n_seeds:    int,
    done_keys:  set,
    warmup:     bool = True,
) -> list[dict]:
    """
    Sweep all N values × seeds for one SEVIR event.

    done_keys : set of (event_id, N, seed) already saved  → skip (resume).
    """
    channels = load_channels(event_id, catalog, data_root)
    if "vil" not in channels:
        log.warning(f"  VIL missing for {event_id} — skipped.")
        return []

    scale   = STIFFNESS_SCALE.get(lc_class, 1.0)
    results = []

    for i_N, N_req in enumerate(N_values):

        # ── GPU SLIC ──────────────────────────────────────────────────────────
        try:
            slic  = TemporalSLIC(N=N_req)
            labels, h0_np, centroids = slic.fit_t0(channels)
        except Exception as exc:
            log.warning(f"  SLIC failed N={N_req}: {exc}")
            continue

        g = build_graph(labels, h0_np, centroids)
        actual_N = g["actual_N"]
        h0       = g["h0"]
        env      = g["env"]
        ga       = g["graph_args"]
        E_super  = g["E_super"]

        # ── JIT warm-up (first N only, first call triggers XLA compilation) ──
        if warmup and i_N == 0:
            _d = ExplicitDiffusion()
            _r = ImplicitReaction(stiffness_scale=scale)
            run_imex  (h0, _d, _r, ga, env, n_steps=2)
            run_dopri5(h0, _d, _r, ga, env, n_steps=2)

        # ── Multi-seed trials ─────────────────────────────────────────────────
        for seed in range(n_seeds):
            key = (event_id, N_req, seed)
            if key in done_keys:
                continue

            k_diff = jax.random.PRNGKey(seed * 100)
            k_rxn  = jax.random.PRNGKey(seed * 100 + 1)
            d_fn   = ExplicitDiffusion(key=k_diff)
            r_fn   = ImplicitReaction(stiffness_scale=scale, key=k_rxn)

            try:
                ir = run_imex  (h0, d_fn, r_fn, ga, env)
                dr = run_dopri5(h0, d_fn, r_fn, ga, env)
            except Exception as exc:
                log.warning(f"  ODE failed event={event_id} N={N_req} seed={seed}: {exc}")
                continue

            l2_gap = float(jnp.linalg.norm(ir["h_T"] - dr["h_T"]) /
                           (jnp.linalg.norm(dr["h_T"]) + 1e-8))

            nfe_imex = ir["nfe_imex_total"]
            nfe_dop  = dr["nfe_dopri5"]

            results.append({
                "event_id":        event_id,
                "lifecycle_class": lc_class,
                "N_requested":     N_req,
                "actual_N":        actual_N,
                "E_super":         E_super,
                "seed":            seed,
                "stiffness_scale": scale,
                "k_decay":         float(scale * BASE_RATE),
                # IMEX breakdown
                "steps_diff":      ir["steps_diff"],
                "steps_rxn":       ir["steps_rxn"],
                "nfe_diff":        ir["nfe_diff"],
                "nfe_rxn":         ir["nfe_rxn"],
                "nfe_imex_total":  nfe_imex,
                "rxn_fraction":    ir["nfe_rxn"] / max(nfe_imex, 1),
                "rej_imex":        ir["rej_imex"],
                "wall_imex":       ir["wall_imex"],
                # DOPRI5
                "steps_dopri5":    dr["steps_dopri5"],
                "nfe_dopri5":      nfe_dop,
                "rej_dopri5":      dr["rej_dopri5"],
                "wall_dopri5":     dr["wall_dopri5"],
                # Comparison
                "nfe_ratio":       nfe_dop / max(nfe_imex, 1),
                "wall_ratio":      dr["wall_dopri5"] / max(ir["wall_imex"], 1e-9),
                "l2_gap":          l2_gap,
            })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def _load_done(out_path: str) -> set:
    """Return set of (event_id, N_requested, seed) already in out_path."""
    if not os.path.exists(out_path):
        return set()
    df = pd.read_csv(out_path, usecols=["event_id", "N_requested", "seed"])
    return {(r.event_id, r.N_requested, r.seed) for r in df.itertuples()}


def _append(records: list[dict], out_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df = pd.DataFrame(records)
    write_header = not os.path.exists(out_path)
    df.to_csv(out_path, mode="a", header=write_header, index=False)


def run_benchmark(
    catalogue_path:   str,
    catalog_raw_path: str,
    data_root:        str,
    N_values:         list[int],
    n_per_class:      int,
    n_seeds:          int,
    out_path:         str,
):
    event_cat = pd.read_csv(catalogue_path, low_memory=False)
    raw_cat   = pd.read_csv(catalog_raw_path, low_memory=False)

    log.info(f"Event catalogue: {len(event_cat)} events")
    log.info(f"Class distribution:\n{event_cat['lifecycle_class'].value_counts()}")

    sampled = (
        event_cat
        .groupby("lifecycle_class", group_keys=False)
        .apply(lambda g: g.sample(min(n_per_class, len(g)), random_state=42))
        .reset_index(drop=True)
    )
    log.info(f"Sampled {len(sampled)} events ({n_per_class}/class)")

    done_keys = _load_done(out_path)
    log.info(f"Resuming: {len(done_keys)} (event,N,seed) combos already done")

    for _, row in tqdm(sampled.iterrows(), total=len(sampled), desc="Events"):
        eid = str(row["id"])
        lc  = str(row["lifecycle_class"])
        log.info(f"\n── {eid}  [{lc}]")

        records = benchmark_event(
            eid, lc, raw_cat, data_root, N_values, n_seeds, done_keys
        )
        if records:
            _append(records, out_path)
            done_keys.update(
                (r["event_id"], r["N_requested"], r["seed"]) for r in records
            )
            log.info(f"   +{len(records)} rows saved → {out_path}")

    log.info(f"\n✓ Benchmark complete. Results in {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10 — Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    agg = (
        df.groupby(["lifecycle_class", "N_requested"])
          .agg(
              nfe_imex_mean   = ("nfe_imex_total", "mean"),
              nfe_imex_std    = ("nfe_imex_total", "std"),
              nfe_diff_mean   = ("nfe_diff",       "mean"),
              nfe_rxn_mean    = ("nfe_rxn",        "mean"),
              nfe_dopri_mean  = ("nfe_dopri5",     "mean"),
              nfe_dopri_std   = ("nfe_dopri5",     "std"),
              rxn_frac_mean   = ("rxn_fraction",   "mean"),
              nfe_ratio_mean  = ("nfe_ratio",      "mean"),
              wall_imex_mean  = ("wall_imex",      "mean"),
              wall_dopri_mean = ("wall_dopri5",    "mean"),
          )
          .reset_index()
    )

    classes = [c for c in LIFECYCLE_ORDER if c in agg["lifecycle_class"].unique()]
    ncols   = min(4, len(classes))
    nrows   = (len(classes) + ncols - 1) // ncols

    # ── Figure 1: NFE vs N — 3-curve breakdown per class ─────────────────────
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, cls in enumerate(classes):
        ax  = axes[i]
        sub = agg[agg["lifecycle_class"] == cls].sort_values("N_requested")
        Ns  = sub["N_requested"].values
        c   = CLASS_COLOR.get(cls, "black")

        ax.fill_between(Ns, 0, sub["nfe_diff_mean"],
                        alpha=0.30, color="steelblue",
                        label=f"IMEX diffusion (Tsit5 ×{NFE_MUL['tsit5']})")
        ax.fill_between(Ns,
                        sub["nfe_diff_mean"],
                        sub["nfe_diff_mean"] + sub["nfe_rxn_mean"],
                        alpha=0.30, color="tomato",
                        label=f"IMEX reaction (Kvaerno5 ×{NFE_MUL['kvaerno5']})")
        ax.plot(Ns, sub["nfe_diff_mean"] + sub["nfe_rxn_mean"],
                color=c, lw=2, label="IMEX total")
        ax.plot(Ns, sub["nfe_dopri_mean"],
                color="black", lw=2, ls="--", label="DOPRI5 (unsplit)")
        ax.fill_between(Ns,
                        sub["nfe_dopri_mean"] - sub["nfe_dopri_std"].fillna(0),
                        sub["nfe_dopri_mean"] + sub["nfe_dopri_std"].fillna(0),
                        alpha=0.10, color="black")

        ax.axvline(500,  color="grey",   ls=":",  lw=1.2, label="ISV elbow")
        ax.axvline(1150, color="purple", ls="-.", lw=1.2, label="Global N*")

        ax.set_title(f"{cls}\n(k_decay={sub['N_requested'].map(lambda _: STIFFNESS_SCALE.get(cls,1)*BASE_RATE).iloc[0]:.2f} s⁻¹)",
                     fontsize=9)
        ax.set_xlabel("N (superpixels)")
        ax.set_ylabel("Estimated NFE")
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Block A: NFE vs N  |  IMEX breakdown vs DOPRI5",
                 fontsize=12, weight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "block_a_nfe_vs_N.pdf")
    fig.savefig(p, dpi=150, bbox_inches="tight")
    log.info(f"Figure 1 → {p}")
    plt.close(fig)

    # ── Figure 2: NFE ratio heatmap ───────────────────────────────────────────
    pivot = (
        agg.pivot(index="lifecycle_class", columns="N_requested",
                  values="nfe_ratio_mean")
           .reindex(classes)
    )
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    im = ax2.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0.3, vmax=6.0)
    N_cols = list(pivot.columns)
    ax2.set_xticks(range(len(N_cols)))
    ax2.set_xticklabels(N_cols, rotation=45, ha="right")
    ax2.set_yticks(range(len(classes)))
    ax2.set_yticklabels(classes, fontsize=8)
    ax2.set_xlabel("N (superpixels)")
    ax2.set_title("NFE ratio: DOPRI5 / IMEX  (green >1 = IMEX cheaper)")
    if 500  in N_cols: ax2.axvline(N_cols.index(500),  color="grey",   ls=":", lw=1.5)
    if 1150 in N_cols: ax2.axvline(N_cols.index(1150), color="purple", ls="-.", lw=1.5)
    for r in range(len(classes)):
        for col_i, _ in enumerate(N_cols):
            v = pivot.values[r, col_i]
            if not np.isnan(v):
                ax2.text(col_i, r, f"{v:.1f}", ha="center", va="center",
                         fontsize=7)
    plt.colorbar(im, ax=ax2, label="DOPRI5 NFE / IMEX NFE")
    fig2.tight_layout()
    p2 = os.path.join(out_dir, "block_a_nfe_ratio_heatmap.pdf")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    log.info(f"Figure 2 → {p2}")
    plt.close(fig2)

    # ── Figure 3: reaction fraction of IMEX NFE ───────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 3.5))
    for cls in classes:
        sub = agg[agg["lifecycle_class"] == cls].sort_values("N_requested")
        ax3.plot(sub["N_requested"], sub["rxn_frac_mean"],
                 color=CLASS_COLOR.get(cls, "black"),
                 marker="o", ms=4, lw=1.5, label=cls)
    ax3.axvline(500,  color="grey",   ls=":",  lw=1.2)
    ax3.axvline(1150, color="purple", ls="-.", lw=1.2)
    ax3.set_xlabel("N (superpixels)")
    ax3.set_ylabel("NFE_rxn / NFE_imex_total")
    ax3.set_title(
        "Reaction fraction of IMEX NFE\n"
        "High → implicit handling critical; "
        "Low → explicit CFL (diffusion) dominates"
    )
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(alpha=0.3)
    fig3.tight_layout()
    p3 = os.path.join(out_dir, "block_a_rxn_fraction.pdf")
    fig3.savefig(p3, dpi=150, bbox_inches="tight")
    log.info(f"Figure 3 → {p3}")
    plt.close(fig3)

    # ── Summary table ─────────────────────────────────────────────────────────
    tbl = (
        agg[agg["N_requested"].isin([500, 1150])]
        [[
            "lifecycle_class", "N_requested",
            "nfe_diff_mean", "nfe_rxn_mean", "nfe_imex_mean",
            "nfe_dopri_mean", "nfe_ratio_mean",
            "wall_imex_mean", "wall_dopri_mean",
        ]]
        .round(1)
    )
    log.info("\n── Summary at ISV elbow (N=500) and global N* (N=1150) ──")
    log.info("\n" + tbl.to_string(index=False))
    tbl.to_csv(os.path.join(out_dir, "block_a_summary_table.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11 — Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DRIFT Block A — IMEX vs DOPRI5 solver benchmark (HPC version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--catalogue",   default=DEFAULT_CAT)
    parser.add_argument("--raw_catalog", default=CATALOG_PATH)
    parser.add_argument("--data_root",   default=DATA_ROOT)
    parser.add_argument("--n_per_class", type=int,       default=N_PER_CLASS)
    parser.add_argument("--n_seeds",     type=int,       default=N_SEEDS)
    parser.add_argument("--n_values",    type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--out",         default=DEFAULT_OUT)
    parser.add_argument("--plot_only",   action="store_true",
                        help="Skip benchmark; replot existing --out CSV.")
    args = parser.parse_args()

    log.info(f"JAX backend  : {jax.default_backend()}")
    log.info(f"JAX devices  : {jax.devices()}")
    log.info(f"N sweep      : {args.n_values}")
    log.info(f"n_per_class  : {args.n_per_class},  n_seeds: {args.n_seeds}")

    if args.plot_only:
        df = pd.read_csv(args.out)
        log.info(f"Loaded {len(df)} rows from {args.out}")
    else:
        run_benchmark(
            catalogue_path   = args.catalogue,
            catalog_raw_path = args.raw_catalog,
            data_root        = args.data_root,
            N_values         = args.n_values,
            n_per_class      = args.n_per_class,
            n_seeds          = args.n_seeds,
            out_path         = args.out,
        )
        df = pd.read_csv(args.out)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    plot_results(df, out_dir=out_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
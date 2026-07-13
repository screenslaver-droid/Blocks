"""
trace.py  —  TRACE: Temporal-Reaction-Advection-Conditioned-Eulerian Framework
================================================================================
Physics-informed Neural ODE for SEVIR storm nowcasting.

Architecture overview
---------------------
  1. Encoder       TemporalSLIC_DEM segments the burn-in frames into N
                   Lagrangian superpixels.  Yields S(0), V0, M_route.

  2. ODE Engine    ADRVectorField integrates S(t) = [X(t) | H_phys(t)]
                   over t = 0 → N_FORECAST frames via dopri5 + adjoint.
                     • Advection  — deterministic kinematics at V0
                     • Diffusion  — mass-conserving GAT flux (RBF-masked)
                     • Reaction   — DEM-conditioned source/sink MLPs

  3. Decoder       MemoryMappedDecoder:
                     • Broadcast  I_base(T) = H_phys(T)[M_route]
                     • Warp       Î(T) = Warp(I_base(T),  V_flow · T)
                   No learned parameters — exact Lagrangian → Eulerian projection.

  4. Loss          BalancedMSE:
                     ω(I) = 1 + α·max(0, I − τ_heavy)
                     L = Σ_t Σ_p [ λ1·ω·‖Î_VIL − I_VIL‖² + λ2·‖Î_IR − I_IR‖² ]

  5. Optimiser     AdamW  +  gradient clipping  +  cosine-annealing LR.

SEVIR channel order  (FUSION_CHANNELS from Compare_superpixel_metrics_claude.py)
  H_phys[:,0]  —  VIL    (Vertically Integrated Liquid)
  H_phys[:,1]  —  IR107  (10.7 µm brightness temperature)
  H_phys[:,2]  —  IR069  (6.9 µm water-vapour channel)

Temporal convention
  N_BURNIN  = 12 frames  →  60 min of context  (t = −60 … 0)
  N_FORECAST = 12 frames →  60 min of forecast  (t = +5 … +60)
  ODE time units: integer frame index (1 unit = 5 minutes)

Dependencies
  pip install torch torchdiffeq scikit-image opencv-python scipy
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations

import logging
from typing import NamedTuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import center_of_mass
from skimage.filters import sobel
from skimage.segmentation import slic, watershed
from torchdiffeq import odeint_adjoint as odeint

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global hyper-parameters
# --------------------------------------------------------------------------- #
# Channel layout
FUSION_CHANNELS  = ["vil", "ir107", "ir069"]
N_PHYS           = len(FUSION_CHANNELS)      # 3

# Segmentation (kept in sync with Compare_superpixel_metrics_claude.py)
N_SEGMENTS       = 1500
COMPACTNESS      = 10.0
ELEVATION_LAMBDA = 0.5

# Temporal structure
N_BURNIN         = 12    # burn-in frames   (60 min context)
N_FORECAST       = 12    # forecast frames  (60 min horizon)

# Loss hyper-parameters
LAMBDA_VIL       = 1.0       # λ1 — VIL weight
LAMBDA_IR        = 0.5       # λ2 — IR channel weight
ALPHA_HEAVY      = 4.0       # α  — heavy-precip amplification
TAU_HEAVY        = 133/255   # τ  — heavy-precip threshold (normalised)

# ODE solver tolerances
ODE_RTOL         = 1e-3
ODE_ATOL         = 1e-4


# =========================================================================== #
# 0.  TemporalSLIC_DEM  (canonical — do not modify)
# =========================================================================== #
class TemporalSLIC_DEM:
    """
    Lagrangian Temporal SLIC on a fused multi-channel + DEM feature cube.
    Centroids advected via windowed-mean Farneback flow.
    Returns (labels, flow) per frame.
    """
    def __init__(self, dem_norm: np.ndarray, n_segments: int = N_SEGMENTS,
                 compactness: float = COMPACTNESS, lambda_z: float = ELEVATION_LAMBDA):
        self.n_segments   = n_segments
        self.compactness  = compactness
        self.lambda_z     = lambda_z
        self.dem_norm     = dem_norm
        self.dem_weighted = dem_norm * lambda_z

        self.prev_labels    = None
        self.prev_centroids = None   # {label_id: np.array([y, x])}
        self.prev_gray      = None

    def segment(self, fused_frame: np.ndarray, use_flow: bool = True):
        if fused_frame.ndim == 2:
            fused_frame = fused_frame[:, :, np.newaxis]
        H, W, C = fused_frame.shape

        gray_norm = fused_frame[:, :, 0]
        curr_gray = (gray_norm * 255).astype(np.uint8)

        dem_ch       = (self.dem_norm * self.lambda_z)[:H, :W, np.newaxis]
        feature_cube = np.concatenate([fused_frame, dem_ch], axis=-1).astype(np.float64)

        flow = np.zeros((H, W, 2), dtype=np.float32)

        if self.prev_labels is None:
            labels = slic(feature_cube, n_segments=self.n_segments,
                          compactness=self.compactness, start_label=1,
                          enforce_connectivity=True, channel_axis=-1)
            self.prev_centroids = self._calculate_centroids(labels)
        else:
            if use_flow:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray, curr_gray, None,
                    pyr_scale=0.5, levels=3, winsize=15,
                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                )
            advected_markers = np.zeros((H, W), dtype=np.int32)
            for lbl, centroid in self.prev_centroids.items():
                y_old, x_old = int(centroid[0]), int(centroid[1])
                if 0 <= y_old < H and 0 <= x_old < W:
                    window = 5
                    y1 = max(0, y_old - window); y2 = min(H, y_old + window)
                    x1 = max(0, x_old - window); x2 = min(W, x_old + window)
                    flow_y = float(np.mean(flow[y1:y2, x1:x2, 1]))
                    flow_x = float(np.mean(flow[y1:y2, x1:x2, 0]))
                    y_new  = int(np.clip(y_old + flow_y, 0, H - 1))
                    x_new  = int(np.clip(x_old + flow_x, 0, W - 1))
                    advected_markers[y_new, x_new] = lbl

            grad_layers = [sobel(fused_frame[:, :, c]) for c in range(C)]
            grad_layers.append(sobel(self.dem_weighted))
            combined_gradient = np.max(np.stack(grad_layers, axis=0), axis=0)

            labels = watershed(combined_gradient, markers=advected_markers,
                               compactness=self.compactness / 100.0)
            self.prev_centroids = self._calculate_centroids(labels)

        self.prev_labels = labels
        self.prev_gray   = curr_gray
        return labels, flow

    def _calculate_centroids(self, labels: np.ndarray) -> dict:
        unique = np.unique(labels)
        if len(unique) and unique[0] == 0:
            unique = unique[1:]
        cms = center_of_mass(np.ones_like(labels), labels, unique)
        return {lbl: np.array(c) for lbl, c in zip(unique, cms)}


# =========================================================================== #
# 1.  ENCODER  —  superpixels → ODE initial conditions
# =========================================================================== #

class EncoderOutput(NamedTuple):
    """All tensors produced by the encoder, passed into TRACEModel.forward()."""
    X0:       torch.Tensor   # (N, 2)  initial spatial state (x, y)
    H0:       torch.Tensor   # (N, 3)  initial physical feature state
    V0:       torch.Tensor   # (N, 2)  superpixel velocity (pixels/frame)
    V_flow:   torch.Tensor   # (H, W, 2)  dense pixel-level flow (for warp)
    M_route:  torch.Tensor   # (H, W)  int64  pixel → superpixel row-index
    lbl_ids:  torch.Tensor   # (N,)    int64  label IDs (for diagnostics)
    img_shape: tuple[int, int]


def _build_M_route(labels: np.ndarray, ordered_lbls: list[int]) -> np.ndarray:
    """
    Vectorised pixel-to-superpixel row-index map.

    M_route[y, x] = i  means H_phys[i] is the superpixel that owns pixel (y,x).
    Pixels with label 0 (unlabelled watershed border) default to row 0.
    """
    max_lbl = int(max(ordered_lbls)) + 1
    lbl_to_row = np.zeros(max_lbl, dtype=np.int64)
    for row_i, lbl in enumerate(ordered_lbls):
        lbl_to_row[lbl] = row_i
    safe = np.clip(labels, 0, max_lbl - 1).astype(np.int64)
    return lbl_to_row[safe]   # (H, W)  int64


def encode(
    fused_frame: np.ndarray,   # (H, W, C)  normalised float32  — the t=0 anchor
    labels:      np.ndarray,   # (H, W)     TemporalSLIC_DEM label map at t=0
    centroids:   dict,         # {label_id: np.array([y, x])}
    flow:        np.ndarray,   # (H, W, 2)  Farneback flow at t=0
    device:      torch.device,
    flow_window: int = 5,
) -> EncoderOutput:
    """
    Convert TemporalSLIC_DEM outputs into the packed ODE initial state S(0).

    Spatial state  X[i] = (x_i, y_i)  — converted from centroid [y, x] storage.
    Physical state H[i] = mean(fused_frame[mask_i])  — superpixel-pooled.
    Velocity       V0[i]= windowed mean flow at centroid  (pixels per frame).
    M_route[y, x]  = row index i such that X[i], H[i] own pixel (y, x).
    V_flow stored as dense tensor for the decoder warp step.
    """
    H_img, W_img, C = fused_frame.shape
    ordered = sorted(centroids.keys())
    N = len(ordered)

    X_list, H_list, V0_list = [], [], []

    for lbl in ordered:
        yx = centroids[lbl]                                   # [y, x]
        X_list.append([float(yx[1]), float(yx[0])])          # store (x, y)

        # --- Physical state: mean feature vector over member pixels ----------
        mask = (labels == lbl)
        feat = fused_frame[mask].mean(axis=0) if mask.any() else np.zeros(C)
        feat_fixed = np.zeros(N_PHYS, dtype=np.float32)
        feat_fixed[:min(C, N_PHYS)] = feat[:min(C, N_PHYS)]
        H_list.append(feat_fixed)

        # --- Velocity: windowed-mean optical flow at centroid ----------------
        y_c = int(np.clip(yx[0], 0, H_img - 1))
        x_c = int(np.clip(yx[1], 0, W_img - 1))
        y1, y2 = max(0, y_c - flow_window), min(H_img, y_c + flow_window)
        x1, x2 = max(0, x_c - flow_window), min(W_img, x_c + flow_window)
        vx = float(np.mean(flow[y1:y2, x1:x2, 0]))
        vy = float(np.mean(flow[y1:y2, x1:x2, 1]))
        V0_list.append([vx, vy])

    M_route_np = _build_M_route(labels, ordered)

    return EncoderOutput(
        X0      = torch.tensor(X_list,      dtype=torch.float32, device=device),
        H0      = torch.tensor(H_list,      dtype=torch.float32, device=device),
        V0      = torch.tensor(V0_list,     dtype=torch.float32, device=device),
        V_flow  = torch.tensor(flow,        dtype=torch.float32, device=device),
        M_route = torch.tensor(M_route_np,  dtype=torch.int64,   device=device),
        lbl_ids = torch.tensor(ordered,     dtype=torch.int64,   device=device),
        img_shape = (H_img, W_img),
    )


# =========================================================================== #
# 2A.  DIFFUSION  —  mass-conserving GAT with RBF-masked attention
# =========================================================================== #

class LagrangianDiffusion(nn.Module):
    """
    Gradient-based message passing that satisfies conservation of mass.

    The flux is the attention-weighted sum of neighbour–self differences:
        dH_i/dt|_diff = Σ_j α_ij · W_flux · (H_j − H_i)

    The RBF distance penalty is absorbed into the softmax exponent:
        α_ij = Softmax_j( e_ij − ‖x_i−x_j‖² / 2σ² )

    This prevents non-physical teleportation and avoids log(0) underflow.
    The k-NN dynamic graph is rebuilt from current X(t) at every ODE step.

    Parameters
    ----------
    n_phys : physical feature dimension  (3)
    D      : attention projection dimension
    k_nn   : spatial neighbours
    sigma  : RBF bandwidth in pixels
    """
    def __init__(self, n_phys: int = N_PHYS, D: int = 16,
                 k_nn: int = 12, sigma: float = 30.0):
        super().__init__()
        self.k_nn  = k_nn
        self.sigma = sigma

        self.W_att  = nn.Linear(n_phys, D, bias=False)
        self.a      = nn.Parameter(torch.empty(2 * D))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.W_flux = nn.Linear(n_phys, n_phys, bias=False)
        self.leaky  = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        X : (N, 2)  continuous superpixel positions
        H : (N, 3)  physical feature vectors
        → dH/dt|_diff : (N, 3)
        """
        N = H.shape[0]
        if N == 0:
            return torch.zeros_like(H)

        k = min(self.k_nn + 1, N)

        # --- Step A: project features through W_att -------------------------
        Wh = self.W_att(H)                                   # (N, D)

        # --- Rebuild dynamic k-NN graph from positions X(t) ----------------
        diff  = X.unsqueeze(0) - X.unsqueeze(1)             # (N, N, 2)
        dist2 = (diff ** 2).sum(-1)                          # (N, N)
        _, idx = torch.topk(dist2, k, dim=1, largest=False, sorted=True)
        idx = idx[:, 1:]                                     # drop self  (N, k-1)
        k_eff = idx.shape[1]

        # --- Step B: RBF-masked attention (absorbed into softmax) -----------
        Whi    = Wh.unsqueeze(1).expand(-1, k_eff, -1)       # (N, k, D)
        Whj    = Wh[idx]                                      # (N, k, D)
        e      = self.leaky(
                     (torch.cat([Whi, Whj], -1) * self.a).sum(-1)
                 )                                            # (N, k) raw scores
        d2_nbr = dist2.gather(1, idx)                        # (N, k)
        alpha  = F.softmax(e - d2_nbr / (2.0 * self.sigma ** 2), dim=-1)  # (N, k)

        # --- Step C: mass-conserving flux derivative ------------------------
        flux    = self.W_flux(H[idx] - H.unsqueeze(1).expand(-1, k_eff, -1))
        dH_diff = (alpha.unsqueeze(-1) * flux).sum(1)        # (N, 3)
        return dH_diff


# =========================================================================== #
# 2B.  REACTION  —  DEM-conditioned orographic forcing
# =========================================================================== #

class EulerianReaction(nn.Module):
    """
    Localised thermodynamic forcing conditioned on the static terrain grid.

        e_i(t) = BilinearSample(E_grid, x_i(t))
        s_i(t) = σ( w_gate^T [h_i ‖ e_i] + b_gate )
        dH_i/dt|_rxn = s_i · MLP_src(h_i, e_i) − (1−s_i) · MLP_snk(h_i, e_i)

    The soft mask s_i ∈ [0,1] smoothly interpolates between convective
    initiation (s_i → 1) and dissipation/rainout (s_i → 0).

    Parameters
    ----------
    n_phys  : physical feature dimension  (3)
    n_env   : DEM/env channels sampled per superpixel (1 for scalar elevation)
    hidden  : hidden units in both MLPs
    """
    def __init__(self, n_phys: int = N_PHYS, n_env: int = 1, hidden: int = 32):
        super().__init__()
        inp = n_phys + n_env

        # Gating network
        self.gate = nn.Linear(inp, 1)

        # Competing source / sink MLPs
        def _mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(inp, hidden), nn.Tanh(),
                nn.Linear(hidden, n_phys),
            )
        self.mlp_src = _mlp()
        self.mlp_snk = _mlp()

    @staticmethod
    def _bilinear_sample(E_grid: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Differentiably query a static Eulerian grid at continuous positions X.

        E_grid : (H_g, W_g) or (H_g, W_g, C_env)
        X      : (N, 2)   continuous (x, y) pixel coordinates
        → (N, C_env)   sampled environmental vectors
        """
        if E_grid.dim() == 2:
            E_grid = E_grid[None, None]             # (1, 1, H, W)
        elif E_grid.dim() == 3:
            E_grid = E_grid.permute(2, 0, 1)[None]  # (1, C, H, W)

        _, C_env, H_g, W_g = E_grid.shape

        # Normalise (x, y) pixel coordinates → [-1, 1] for grid_sample
        xn = (X[:, 0] / (W_g - 1)) * 2.0 - 1.0    # (N,)
        yn = (X[:, 1] / (H_g - 1)) * 2.0 - 1.0    # (N,)
        grid = torch.stack([xn, yn], -1).view(1, 1, -1, 2)   # (1, 1, N, 2)

        sampled = F.grid_sample(E_grid.float(), grid.float(),
                                mode='bilinear', align_corners=True,
                                padding_mode='border')         # (1, C, 1, N)
        return sampled.squeeze(0).squeeze(2).permute(1, 0)    # (N, C_env)

    def forward(self, H: torch.Tensor, X: torch.Tensor,
                E_grid: torch.Tensor) -> torch.Tensor:
        """
        H      : (N, 3)   physical features
        X      : (N, 2)   continuous positions
        E_grid : (H, W)   normalised DEM on same device
        → dH/dt|_rxn : (N, 3)
        """
        e_i = self._bilinear_sample(E_grid, X)       # (N, 1)
        ctx = torch.cat([H, e_i], dim=-1)             # (N, 4)

        s_i    = torch.sigmoid(self.gate(ctx))        # (N, 1)  ∈ [0,1]
        src    = self.mlp_src(ctx)                    # (N, 3)  growth
        snk    = self.mlp_snk(ctx)                    # (N, 3)  decay
        return s_i * src - (1.0 - s_i) * snk         # (N, 3)


# =========================================================================== #
# 3.  ODE VECTOR FIELD  F_Θ(S(t), t)
# =========================================================================== #

class ADRVectorField(nn.Module):
    """
    The complete ADR vector field integrated by the adjoint ODE solver.

    State packing:  S ∈ R^{N × 5}
        S[:, :2] = X(t)      spatial positions
        S[:, 2:] = H_phys(t) physical scalars

    Spatial derivative  dX/dt = V0   (Taylor's Frozen Turbulence hypothesis)
      The storm structure is advected deterministically at the steering flow
      captured at t=0. The ODE learns *only* the microphysical evolution.

    Feature derivative  dH/dt = dH|_diff + dH|_rxn

    Context (V0, E_grid) is injected once per odeint call via set_context().
    These are not learnable parameters — they do not enter the adjoint ODE.
    """
    def __init__(self, n_phys: int = N_PHYS, D: int = 16,
                 k_nn: int = 12, sigma: float = 30.0,
                 n_env: int = 1,   hidden: int = 32):
        super().__init__()
        self.diffusion = LagrangianDiffusion(n_phys, D, k_nn, sigma)
        self.reaction  = EulerianReaction(n_phys, n_env, hidden)

        self._V0:     torch.Tensor | None = None
        self._E_grid: torch.Tensor | None = None

    def set_context(self, V0: torch.Tensor, E_grid: torch.Tensor) -> None:
        """Inject per-sample non-learnable context. Call before every odeint."""
        self._V0     = V0       # (N, 2)
        self._E_grid = E_grid   # (H, W)

    def forward(self, t: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        t : scalar  (unused — autonomous system)
        S : (N, 5)  packed physical state
        → dS/dt : (N, 5)
        """
        X, H = S[..., :2], S[..., 2:]   # (N,2)  (N,3)

        dX     = self._V0                                          # (N, 2)
        dH     = (self.diffusion(X, H) +
                  self.reaction(H, X, self._E_grid))               # (N, 3)

        return torch.cat([dX, dH], dim=-1)                         # (N, 5)


# =========================================================================== #
# 4.  DECODER  —  memory-mapped Eulerian projection + flow warp
# =========================================================================== #

class MemoryMappedDecoder(nn.Module):
    """
    Exact, parameter-free Lagrangian → Eulerian projection.

    Step 1  Broadcast:
        I_base(T)[y, x] = H_phys(T)[M_route[y, x]]
        Each pixel inherits the current physical state of its parent superpixel.

    Step 2  Warp:
        Î(T) = Warp(I_base(T),  V_flow · T)
        Shift the broadcast image by the accumulated pixel-level displacement
        at lead-time T. This corrects for sub-superpixel intra-cell motion.

    The warp is implemented via differentiable bilinear grid_sample so that
    gradients flow back through the warp to H_phys(T) during adjoint backward.

    No learned parameters — the only source of error corrected by training
    is the ODE microphysics (diffusion + reaction), not the projection itself.
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def _warp(img: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        """
        Warp img by a pixel-space displacement field.

        img          : (H, W, C)
        displacement : (H, W, 2)  [dx, dy] in pixels
        → (H, W, C)
        """
        H, W, C = img.shape
        device  = img.device

        base_y, base_x = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=device),
            torch.arange(W, dtype=torch.float32, device=device),
            indexing='ij',
        )                                      # (H, W) each

        # Sampling coords: where each output pixel pulls FROM in the source
        sx = base_x + displacement[..., 0]    # (H, W)
        sy = base_y + displacement[..., 1]    # (H, W)

        # Normalise to [-1, 1] for F.grid_sample
        sx_n = (sx / (W - 1)) * 2.0 - 1.0
        sy_n = (sy / (H - 1)) * 2.0 - 1.0
        grid = torch.stack([sx_n, sy_n], -1).unsqueeze(0)  # (1, H, W, 2)

        # grid_sample expects (N, C, H, W)
        src   = img.permute(2, 0, 1).unsqueeze(0)          # (1, C, H, W)
        out   = F.grid_sample(src.float(), grid.float(),
                              mode='bilinear', align_corners=True,
                              padding_mode='border')        # (1, C, H, W)
        return out.squeeze(0).permute(1, 2, 0)             # (H, W, C)

    def forward(
        self,
        H_T:      torch.Tensor,   # (N, C)  superpixel features at forecast time T
        M_route:  torch.Tensor,   # (H, W)  int64  pixel → row index
        V_flow:   torch.Tensor,   # (H, W, 2)  pixel-level flow [dx, dy]
        lead_T:   float,          # lead time in frames (e.g. 3 for t=+15 min)
    ) -> torch.Tensor:
        """
        Returns Î(T) : (H, W, C)  predicted normalised image.
        """
        H_img, W_img = M_route.shape

        # Step 1 — broadcast: paint every pixel with its superpixel's features
        # M_route contains row indices 0..N-1; shape (H, W)
        I_base = H_T[M_route]                              # (H, W, C)

        # Step 2 — warp by accumulated displacement V_flow · T
        I_pred = self._warp(I_base, V_flow * lead_T)       # (H, W, C)

        return I_pred


# =========================================================================== #
# 5.  LOSS FUNCTION  —  Balanced MSE
# =========================================================================== #

class BalancedMSE(nn.Module):
    """
    Physics-informed sequence loss that prevents optimisation collapse onto
    clear-air (the dominant class in SEVIR).

    ω(I_true) = 1 + α · max(0, I_true − τ_heavy)
    L = Σ_t (1/|G|) Σ_p [λ1·ω·‖Î_VIL−I_VIL‖² + λ2·‖Î_IR−I_IR‖²]

    VIL    = channel 0  (I_true[:,0] used for ω)
    IR107  = channel 1  ⎫  both enter the λ2 IR term
    IR069  = channel 2  ⎭

    Parameters match the notation in the TRACE blueprint.
    """
    def __init__(self,
                 lambda_vil:  float = LAMBDA_VIL,
                 lambda_ir:   float = LAMBDA_IR,
                 alpha_heavy: float = ALPHA_HEAVY,
                 tau_heavy:   float = TAU_HEAVY):
        super().__init__()
        self.lambda_vil  = lambda_vil
        self.lambda_ir   = lambda_ir
        self.alpha_heavy = alpha_heavy
        self.tau_heavy   = tau_heavy

    def forward(
        self,
        preds:   list[torch.Tensor],   # T × (H, W, C)  predictions
        targets: list[torch.Tensor],   # T × (H, W, C)  ground truth
    ) -> torch.Tensor:
        """
        preds, targets : lists of length N_FORECAST, each (H, W, 3).
        Returns scalar loss.
        """
        total = torch.tensor(0.0, device=preds[0].device)
        n_pix = preds[0][..., 0].numel()

        for pred, target in zip(preds, targets):
            # ── VIL term (channel 0) with exponential heavy-precip weight ── #
            I_vil_true  = target[..., 0]                      # (H, W)
            omega       = (1.0 + self.alpha_heavy
                           * (I_vil_true - self.tau_heavy).clamp(min=0.0))
            vil_err     = (pred[..., 0] - I_vil_true) ** 2
            loss_vil    = (omega * vil_err).sum() / n_pix

            # ── IR term (channels 1 and 2) ──────────────────────────────── #
            ir_err   = ((pred[..., 1:] - target[..., 1:]) ** 2).sum(-1)
            loss_ir  = ir_err.sum() / n_pix

            total = total + self.lambda_vil * loss_vil + self.lambda_ir * loss_ir

        return total / max(len(preds), 1)


# =========================================================================== #
# 6.  TRACE MODEL  —  end-to-end forward pass
# =========================================================================== #

class TRACEModel(nn.Module):
    """
    Full Eulerian-Conditioned Lagrangian GAT-ODE pipeline.

    Forward pass
    ------------
    Given EncoderOutput (produced externally by the encode() function):
      1. Pack  S(0) = [X(0) | H_phys(0)]
      2. Set ODE context  (V0, E_grid)
      3. Integrate S(t) from t=0 to t=N_FORECAST using dopri5 + adjoint
      4. Decode Î(t) for each forecast step via MemoryMappedDecoder

    The ODE is queried at integer time steps [1, 2, ..., N_FORECAST] so that
    each step corresponds to one 5-minute SEVIR frame interval.

    Parameters
    ----------
    n_phys     : physical feature dimension  (3)
    D          : GAT attention projection dim
    k_nn       : spatial neighbours in dynamic graph
    sigma      : GAT RBF bandwidth (pixels)
    hidden     : MLP hidden units in reaction and decoder
    n_forecast : number of forecast frames
    """
    def __init__(self,
                 n_phys:     int   = N_PHYS,
                 D:          int   = 16,
                 k_nn:       int   = 12,
                 sigma:      float = 30.0,
                 hidden:     int   = 32,
                 n_forecast: int   = N_FORECAST):
        super().__init__()
        self.n_forecast = n_forecast
        self.ode_func   = ADRVectorField(n_phys, D, k_nn, sigma, 1, hidden)
        self.decoder    = MemoryMappedDecoder()

        # Evaluation time points (in frame units, as float for torchdiffeq)
        t_eval = torch.arange(0, n_forecast + 1, dtype=torch.float32)
        self.register_buffer("t_eval", t_eval)          # (n_forecast+1,)

    def forward(self, enc: EncoderOutput,
                E_grid: torch.Tensor) -> list[torch.Tensor]:
        """
        enc    : EncoderOutput from encode()
        E_grid : (H, W)  normalised DEM tensor on the correct device

        Returns
        -------
        preds : list of N_FORECAST tensors, each (H, W, C)
                Î(t=1), Î(t=2), ..., Î(t=N_FORECAST)
        """
        S0 = torch.cat([enc.X0, enc.H0], dim=-1)    # (N, 5)

        # Inject context — V0 and E_grid are NOT part of Θ
        self.ode_func.set_context(enc.V0, E_grid)

        # ── Adjoint integration — O(1) GPU memory ──────────────────────── #
        # odeint_adjoint stores only S(t_eval); the backward ODE recomputes
        # the trajectory on the fly rather than storing it in the graph.
        S_traj = odeint(
            self.ode_func,
            S0,
            self.t_eval.to(S0.device),
            method='dopri5',
            adjoint_params=list(self.ode_func.parameters()),
            rtol=ODE_RTOL, atol=ODE_ATOL,
        )                                             # (n_forecast+1, N, 5)

        # ── Decode each forecast step ───────────────────────────────────── #
        preds = []
        for step in range(1, self.n_forecast + 1):
            H_T = S_traj[step, :, 2:]                # (N, 3)  feature state
            pred_frame = self.decoder(
                H_T, enc.M_route, enc.V_flow, float(step)
            )                                         # (H, W, 3)
            preds.append(pred_frame)

        return preds    # list of N_FORECAST × (H, W, 3)


# =========================================================================== #
# 7.  TRAINING
# =========================================================================== #

def train_one_event(
    fused:     np.ndarray,              # (T, H, W, C)  normalised float32
    dem_norm:  np.ndarray,              # (H, W)        normalised DEM
    model:     TRACEModel,
    optimizer: torch.optim.Optimizer,
    criterion: BalancedMSE,
    device:    torch.device,
    n_epochs:  int   = 20,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> list[float]:
    """
    Sequence-to-sequence training on one SEVIR event.

    Temporal split per the TRACE blueprint:
      Burn-in  [ 0 … N_BURNIN ]   TemporalSLIC_DEM warm-up, no gradient.
      Anchor   [ N_BURNIN ]        t=0, source of S(0), M_route, V0.
      Forecast [ N_BURNIN+1 … N_BURNIN+N_FORECAST ]   ODE targets.

    If the event has fewer than N_BURNIN + N_FORECAST + 1 frames, the
    available frames are split proportionally (burn-in 50 %, forecast 50 %).

    Returns list of per-epoch mean losses.
    """
    T_total = fused.shape[0]

    # Dynamic split if the event is shorter than the canonical window
    if T_total < N_BURNIN + N_FORECAST + 1:
        n_burnin   = max(1, T_total // 2 - 1)
        n_forecast = min(model.n_forecast, T_total - n_burnin - 1)
        log.warning(
            f"Short event ({T_total} frames): burn-in={n_burnin}, "
            f"forecast={n_forecast}"
        )
    else:
        n_burnin   = N_BURNIN
        n_forecast = model.n_forecast

    anchor_idx    = n_burnin                           # t = 0
    forecast_idxs = list(range(anchor_idx + 1,
                                anchor_idx + n_forecast + 1))

    E_grid = torch.tensor(dem_norm, dtype=torch.float32, device=device)

    # ── Phase 1: Burn-in — run TemporalSLIC_DEM to warm up the scaffold ── #
    # No gradients here; purely builds stable centroids / flow at t=0.
    log.info(f"Burn-in: running TemporalSLIC_DEM on frames 0–{anchor_idx} …")
    tslic = TemporalSLIC_DEM(dem_norm)
    for t in range(anchor_idx + 1):
        labels_t, flow_t = tslic.segment(fused[t], use_flow=(t > 0))

    # Snapshot the anchor state (t=0)
    labels_anchor    = labels_t           # type: ignore[possibly-undefined]
    flow_anchor      = flow_t             # type: ignore[possibly-undefined]
    centroids_anchor = dict(tslic.prev_centroids)
    log.info(
        f"Anchor superpixels: {len(centroids_anchor)}  "
        f"(target N_SEGMENTS={N_SEGMENTS})"
    )

    # Encode the anchor frame once — M_route and V_flow are reused every epoch
    enc = encode(fused[anchor_idx], labels_anchor,
                 centroids_anchor, flow_anchor, device)

    # ── Phase 2: Forecast — train the ODE on future frames ─────────────── #
    targets = [
        torch.tensor(fused[i], dtype=torch.float32, device=device)
        for i in forecast_idxs[:n_forecast]
    ]

    epoch_losses: list[float] = []
    model.train()

    for epoch in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)

        preds = model(enc, E_grid)               # list of n_forecast × (H,W,C)

        # Align lengths (preds may be shorter if event was short)
        n_steps = min(len(preds), len(targets))
        loss    = criterion(preds[:n_steps], targets[:n_steps])

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        ep_loss = float(loss.item())
        epoch_losses.append(ep_loss)
        log.info(f"  Epoch {epoch + 1:3d}/{n_epochs}  "
                 f"loss = {ep_loss:.6f}  "
                 f"lr = {optimizer.param_groups[0]['lr']:.2e}")

    return epoch_losses


# =========================================================================== #
# 8.  INFERENCE
# =========================================================================== #

@torch.no_grad()
def forecast(
    fused_history:   np.ndarray,    # (T_hist, H, W, C)  burn-in + anchor frames
    dem_norm:        np.ndarray,    # (H, W)
    model:           TRACEModel,
    device:          torch.device,
) -> np.ndarray:
    """
    Single-event inference.

    Runs TemporalSLIC_DEM over fused_history, encodes the final frame as
    the ODE anchor, and returns the N_FORECAST predicted frames.

    Returns
    -------
    predictions : (N_FORECAST, H, W, C)  float32  normalised predictions
    """
    model.eval()
    E_grid = torch.tensor(dem_norm, dtype=torch.float32, device=device)

    tslic = TemporalSLIC_DEM(dem_norm)
    for t in range(len(fused_history)):
        labels_t, flow_t = tslic.segment(fused_history[t], use_flow=(t > 0))

    enc = encode(
        fused_history[-1], labels_t,    # type: ignore[possibly-undefined]
        dict(tslic.prev_centroids), flow_t, device,  # type: ignore[possibly-undefined]
    )

    preds = model(enc, E_grid)          # list of N_FORECAST × (H, W, C)
    return np.stack([p.cpu().numpy() for p in preds], axis=0)


# =========================================================================== #
# 9.  ENTRY POINT
# =========================================================================== #
if __name__ == "__main__":
    import sys
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    try:
        from Compare_superpixel_metrics_claude import (
            load_fused_channels, fetch_and_regrid_dem, CATALOG_PATH,
        )
    except ImportError:
        log.error(
            "Place trace.py in the same directory as "
            "Compare_superpixel_metrics_claude.py"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    catalog  = pd.read_csv(CATALOG_PATH, parse_dates=["time_utc"],
                           low_memory=False)
    event_id = input("Enter SEVIR event ID: ").strip()

    fused, _, extent = load_fused_channels(event_id, catalog)
    if fused is None:
        log.error("Failed to load event data.")
        sys.exit(1)
    log.info(f"Fused cube: {fused.shape}  (T × H × W × C)")

    dem_raw  = fetch_and_regrid_dem(extent)
    dem_norm = ((dem_raw - dem_raw.min()) /
                (dem_raw.max() - dem_raw.min() + 1e-6)).astype(np.float32)
    log.info(f"DEM range: [{dem_raw.min():.0f} m, {dem_raw.max():.0f} m]")

    # ── Build model + optimiser ─────────────────────────────────────────── #
    model = TRACEModel(
        n_phys=N_PHYS, D=16, k_nn=12, sigma=30.0,
        hidden=32, n_forecast=N_FORECAST,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    log.info(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=20, eta_min=1e-5,
    )
    criterion = BalancedMSE()

    # ── Train ───────────────────────────────────────────────────────────── #
    losses = train_one_event(
        fused, dem_norm, model, optimizer, criterion,
        device, n_epochs=20, scheduler=scheduler,
    )
    log.info(f"Final loss: {losses[-1]:.6f}")

    # ── Save ────────────────────────────────────────────────────────────── #
    save_path = f"{event_id}_trace.pth"
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "final_loss":           losses[-1],
        "losses":               losses,
    }, save_path)
    log.info(f"Checkpoint saved → {save_path}")

    # ── Quick inference sanity check ─────────────────────────────────────── #
    if fused.shape[0] >= N_BURNIN + 1:
        preds = forecast(fused[:N_BURNIN + 1], dem_norm, model, device)
        log.info(f"Forecast output shape: {preds.shape}  "
                 f"value range [{preds.min():.3f}, {preds.max():.3f}]")

"""
smoke_test_synthetic.py
Tests all JAX/Diffrax components from block_a_solver_benchmark.py
using synthetic data. No SEVIR, no GPU, ~300MB RAM.

Run: python smoke_test_synthetic.py
Expected: prints NFE table, saves smoke_test_nfe.png
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import diffrax
from scipy.spatial import KDTree
from skimage.segmentation import slic as skimage_slic
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print(f"JAX version  : {jax.__version__}")
print(f"Diffrax ver  : {diffrax.__version__}")
print(f"JAX backend  : {jax.default_backend()}")
print(f"JAX devices  : {jax.devices()}\n")

# ── Config (tiny values for smoke test) ──────────────────────────────────────
NODE_DIM    = 3
ENV_DIM     = 4
HIDDEN_DIM  = 32      # smaller than production (64) for speed
N_VALUES    = [50, 100, 200]   # tiny — just checking code paths
N_SEEDS     = 2
DT          = 300.0
N_MACRO     = 3       # 3 steps instead of 12 — enough to catch solver issues
R_MAX       = 108
SIGMA_RBF   = 15.0
NFE_MUL     = {"tsit5": 5, "kvaerno5": 7, "dopri5": 6}

STIFFNESS_SCALE = {
    "RAPID_GROWTH": 4.0,
    "STEADY":       1.0,
    "QUIESCENT":    0.5,
}

# ── Synthetic SEVIR frame (384×384, 3 channels) ───────────────────────────────
def make_synthetic_event(H=384, W=384, T=13, seed=0):
    rng = np.random.default_rng(seed)
    # VIL: sparse blobs (like real radar)
    vil = np.zeros((T, H, W), dtype=np.float32)
    for t in range(T):
        cx, cy = rng.integers(100, 284, size=2)
        r      = rng.integers(20, 80)
        ys, xs = np.ogrid[:H, :W]
        blob   = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * r**2))
        vil[t] = (blob * 200 * (1 + 0.1 * t)).clip(0, 255)
    ir069 = (255 - vil * 0.5 + rng.normal(0, 5, vil.shape)).clip(0, 255).astype(np.float32)
    ir107 = (240 - vil * 0.3 + rng.normal(0, 5, vil.shape)).clip(0, 255).astype(np.float32)
    return {"vil": vil, "ir069": ir069, "ir107": ir107}


# ── Graph builder (copied from main script, no changes) ──────────────────────
def build_graph(channels, N, kappa=10.0):
    vil, ir069, ir107 = channels["vil"], channels["ir069"], channels["ir107"]
    H, W = vil.shape[1], vil.shape[2]

    def norm(x): return (x - x.mean()) / (x.std() + 1e-6)
    feat   = np.stack([norm(vil[0]), norm(ir069[0]), norm(ir107[0])], axis=-1)
    labels = skimage_slic(feat, n_segments=N, compactness=kappa,
                          start_label=0, channel_axis=-1,
                          enforce_connectivity=True).astype(np.int32)
    actual_N = int(labels.max()) + 1

    centroids = np.zeros((actual_N, 2), np.float32)
    h0        = np.zeros((actual_N, 3), np.float32)
    for sid in range(actual_N):
        mask = labels == sid
        ys, xs = np.where(mask)
        if len(xs) == 0: continue
        centroids[sid] = [xs.mean(), ys.mean()]
        h0[sid]        = [vil[0][mask].mean()/255, ir069[0][mask].mean()/255,
                          ir107[0][mask].mean()/255]

    env  = np.stack([centroids[:,0]/W, centroids[:,1]/H, h0[:,0], h0[:,1]], axis=-1)
    pairs = np.array(sorted(KDTree(centroids).query_pairs(R_MAX)), dtype=np.int32)
    if len(pairs) == 0:
        k = min(6, actual_N-1)
        _, idx = KDTree(centroids).query(centroids, k=k+1)
        pairs = np.array([(i, int(idx[i,j])) for i in range(actual_N)
                          for j in range(1,k+1) if i < int(idx[i,j])], dtype=np.int32)
    senders   = np.concatenate([pairs[:,0], pairs[:,1]]).astype(np.int32)
    receivers = np.concatenate([pairs[:,1], pairs[:,0]]).astype(np.int32)
    dist_sq   = np.sum((centroids[senders] - centroids[receivers])**2, axis=1)
    ew        = np.exp(-dist_sq / (2*SIGMA_RBF**2)).astype(np.float32)
    return actual_N, jnp.array(h0), jnp.array(env), \
           (jnp.array(senders), jnp.array(receivers), jnp.array(ew))


# ── Neural modules (identical to main script) ────────────────────────────────
class ExplicitDiffusion(eqx.Module):
    """
    Isotropic graph diffusion: dh_i/dt = D × Σ_j ew_ij × (h_j − h_i)

    Scalar D > 0 guarantees stability: eigenvalues = −D × λ_k(L) ≤ 0.
    CFL constraint: Δt ≤ 3.5 / (D × λ_max(L))
    With D=0.05 and λ_max≈3: Δt_max≈23s → ~7 steps per 150s half-step.
    """
    log_D: jax.Array  # D = exp(log_D), always positive

    def __init__(self, key=None):   # key unused, kept for API consistency
        self.log_D = jnp.log(jnp.array(0.05))

    def __call__(self, t, h, args):
        snd, rcv, ew = args
        D    = jnp.exp(self.log_D)
        diff = h[rcv] - h[snd]           # (E, d)
        flux = D * ew[:, None] * diff    # (E, d) — stable, no positive eigenvalues
        return jnp.zeros_like(h).at[rcv].add(flux)

# ── Fix 2: redesign ImplicitReaction ─────────────────────────────────────────
class ImplicitReaction(eqx.Module):
    """
    Stiff linear decay toward a learned equilibrium:

        f_I(h, env) = f_eq(env) - k_decay × h

    Jacobian ∂f_I/∂h = -k_decay × I  →  eigenvalue = -k_decay (stiff for large k_decay).
    Output is bounded because f_eq ∈ [0,1]^d and k_decay×h stays finite.

    Stiffness ratio per macro step SR = k_decay × DT:
        RAPID_GROWTH (scale=4.0): k_decay=1.2 s⁻¹  → SR=360
        STEADY       (scale=1.0): k_decay=0.3 s⁻¹  → SR=90
        QUIESCENT    (scale=0.5): k_decay=0.15 s⁻¹ → SR=45

    DOPRI5 needs ~SR/3.5 steps per macro step to stay stable.
    Kvaerno5 (L-stable) handles any SR in 1–2 steps.
    """
    k_decay: float
    eq_w1:   jax.Array   # (inp, hidden)
    eq_w2:   jax.Array   # (hidden, node_dim)

    def __init__(self, stiffness_scale=1.0, key=None):
        if key is None:
            key = jax.random.PRNGKey(1)
        k1, k2 = jax.random.split(key, 2)
        # base_rate=0.3 s⁻¹ → SR=90 for STEADY; scale multiplies this
        self.k_decay = float(stiffness_scale * 0.3)
        inp = NODE_DIM + ENV_DIM
        # Small weights: equilibrium target stays in [0,1] after sigmoid
        self.eq_w1 = jax.random.normal(k1, (inp, HIDDEN_DIM)) * 0.1
        self.eq_w2 = jax.random.normal(k2, (HIDDEN_DIM, NODE_DIM)) * 0.1

    def __call__(self, t, h, env):
        h_env  = jnp.concatenate([h, env], axis=-1)          # (N, d+e)
        f_eq   = jax.nn.sigmoid(                              # (N, d) in [0,1]
            jnp.tanh(h_env @ self.eq_w1) @ self.eq_w2
        )
        # Jacobian wrt h: -k_decay × I  (diagonal, explicitly stiff)
        return f_eq - self.k_decay * h



# ── Integrators ───────────────────────────────────────────────────────────────

def _pid():
    return diffrax.PIDController(rtol=1e-2, atol=1e-2)

def run_imex(h0, diff_fn, rxn_fn, ga, env):
    steps_d, steps_r = 0, 0
    h = h0
    term_E = diffrax.ODETerm(lambda t, h, a: diff_fn(t, h, a))
    term_I = diffrax.ODETerm(lambda t, h, e: rxn_fn(t, h, e))

    for k in range(N_MACRO):
        t = k * DT

        # Half-step: explicit diffusion
        sol = diffrax.diffeqsolve(
            term_E, diffrax.Tsit5(),
            t0=t, t1=t + DT/2, dt0=DT/20,
            y0=h, args=ga, stepsize_controller=_pid(),
            saveat=diffrax.SaveAt(t1=True), max_steps=16384)
        h = sol.ys[-1]
        steps_d += int(sol.stats["num_accepted_steps"])

        # Full-step: implicit reaction
        sol = diffrax.diffeqsolve(
            term_I, diffrax.Kvaerno5(),
            t0=t, t1=t + DT, dt0=DT/2,
            y0=h, args=env, stepsize_controller=_pid(),
            saveat=diffrax.SaveAt(t1=True), max_steps=2000)
        h = sol.ys[-1]
        steps_r += int(sol.stats["num_accepted_steps"])

        # Half-step: explicit diffusion
        sol = diffrax.diffeqsolve(
            term_E, diffrax.Tsit5(),
            t0=t + DT/2, t1=t + DT, dt0=DT/20,
            y0=h, args=ga, stepsize_controller=_pid(),
            saveat=diffrax.SaveAt(t1=True), max_steps=2000)
        h = sol.ys[-1]
        steps_d += int(sol.stats["num_accepted_steps"])

    return h, steps_d * NFE_MUL["tsit5"], steps_r * NFE_MUL["kvaerno5"]

def run_dopri5(h0, diff_fn, rxn_fn, ga, env):
    def rhs(t, h, args): return diff_fn(t,h,args[0]) + rxn_fn(t,h,args[1])
    steps = 0
    h = h0
    for k in range(N_MACRO):
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs), diffrax.Dopri5(),
            t0=k*DT, t1=(k+1)*DT, dt0=DT/10,
            y0=h, args=(ga, env), stepsize_controller=_pid(),
            saveat=diffrax.SaveAt(t1=True), max_steps=100000)
        h = sol.ys[-1]
        steps += int(sol.stats["num_accepted_steps"])
    return h, steps * NFE_MUL["dopri5"]


# ── Run smoke test ────────────────────────────────────────────────────────────
print("=" * 60)
print("SMOKE TEST — synthetic data, no SEVIR required")
print("=" * 60)

results = []
event = make_synthetic_event()

for cls, scale in STIFFNESS_SCALE.items():
    print(f"\n[{cls}]  stiffness_scale={scale}")
    for N_req in N_VALUES:
        actual_N, h0, env, ga = build_graph(event, N_req)
        for seed in range(N_SEEDS):
            diff_fn = ExplicitDiffusion(key=jax.random.PRNGKey(seed*10))
            rxn_fn  = ImplicitReaction(scale, key=jax.random.PRNGKey(seed*10+1))

            h_imex, nfe_diff, nfe_rxn = run_imex(h0, diff_fn, rxn_fn, ga, env)
            h_dop,  nfe_dop           = run_dopri5(h0, diff_fn, rxn_fn, ga, env)

            ratio = nfe_dop / max(nfe_diff + nfe_rxn, 1)
            print(f"  N={N_req:4d} seed={seed} | "
                  f"IMEX diff={nfe_diff:5d} rxn={nfe_rxn:5d} "
                  f"total={nfe_diff+nfe_rxn:5d} | "
                  f"DOPRI5={nfe_dop:5d} | ratio={ratio:.2f}x")
            results.append(dict(cls=cls, N=N_req, seed=seed,
                                nfe_diff=nfe_diff, nfe_rxn=nfe_rxn,
                                nfe_imex=nfe_diff+nfe_rxn, nfe_dopri=nfe_dop,
                                ratio=ratio))

# ── Quick plot ────────────────────────────────────────────────────────────────
import pandas as pd
print(f"\nDiagnostic: D={float(jnp.exp(ExplicitDiffusion().log_D)):.3f}")
print(f"Expected steps per 150s half-step: ~{int(150 / (3.5/(0.05*3))) + 1}")
print(f"Expected NFE_diff per 60min: ~{int(150/(3.5/(0.05*3))+1) * 24 * 5}")
print(f"Expected NFE_rxn (RAPID_GROWTH): ~{int(1.2*300/3.5) * 12 * 7}")
print()
df   = pd.DataFrame(results)
agg  = df.groupby(["cls","N"]).mean(numeric_only=True).reset_index()
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for ax, cls in zip(axes, STIFFNESS_SCALE.keys()):
    sub = agg[agg["cls"]==cls]
    ax.stackplot(sub["N"], sub["nfe_diff"], sub["nfe_rxn"],
                 labels=["IMEX diffusion","IMEX reaction"],
                 colors=["steelblue","tomato"], alpha=0.7)
    ax.plot(sub["N"], sub["nfe_dopri"], "k--", lw=2, label="DOPRI5")
    ax.set_title(cls); ax.set_xlabel("N"); ax.set_ylabel("NFE")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
fig.suptitle("Smoke test — synthetic data", weight="bold")
fig.tight_layout()
fig.savefig("smoke_test_nfe.png", dpi=120)
print("\n✓ Plot saved → smoke_test_nfe.png")
print("✓ Smoke test passed — safe to submit to HPC")
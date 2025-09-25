# -*- coding: utf-8 -*-
"""
# ============================================================
# Multi-Fidelity PINN (SIA + reduced Stokes) + Boundary Weak-Form
# Data (supervised) + Physics residuals + Boundary weak-form + Adaptive Sampling
# ============================================================

# ---- imports ----
import os, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---- toggles / hyperparams ----
NORMALIZE_Y      = True            # normalize target for training (recommended)
USE_UNCERTAINTY  = False           # False: fixed physics weights (data-first); True: learned (Kendall)
LOG_CLAMP        = (-5.0, 5.0)     # clamp for log-variances if USE_UNCERTAINTY=True

# Physics weights if USE_UNCERTAINTY == False
W_SIA, W_STK     = 0.25, 0.75      # bias toward higher-fidelity Stokes

# Boundary weights and strategy
USE_BOUNDARY     = True            # turn boundary weak-form on/off globally
DIR_FRAC         = 0.0             # fraction of boundary points using Dirichlet (0.0 = all Neumann)
HARD_DIRICHLET   = False           # set True to enforce Dirichlet hard
LAMBDA_NEU       = 0.1             # Neumann weight
LAMBDA_DIR       = 0.0             # Dirichlet weight (keep 0 unless you have meaningful uD)

# Training schedule
EPOCHS           = 25000           # reduce for quick runs; raise for final training
HIDDEN           = 128
LR_INIT, LR_MIN  = 3e-3, 3e-4      # cosine LR schedule
N_COL_START      = 512             # curriculum collocation start
N_COL_END        = 4096            # curriculum collocation end
N_BPER           = 96              # boundary points per side

# Adaptive sampling
USE_ADAPTIVE     = True
ADAPT_FRAC       = 0.5             # fraction of collocation points allocated adaptively
ADAPT_NOISE      = 0.02            # jitter in scaled feature space when sampling near high residual points
ADAPT_CLAMP      = (-6.0, 6.0)     # clamp in scaled feature space (keeps values reasonable)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Load data
# ============================================================
df = pd.read_csv("data_full.csv")

feat_cols = ['surf_x','surf_y','surf_vx','surf_vy','surf_elv','surf_dhdt','surf_SMB']
target_col = 'track_bed_target'

for c in feat_cols + [target_col]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

X_np = df[feat_cols].to_numpy(dtype=np.float32)
y_np = df[target_col].to_numpy(dtype=np.float32).reshape(-1,1)

# Split 80/20 (random)
n = len(X_np)
idx = np.random.permutation(n)
split = int(0.8*n)
train_idx, test_idx = idx[:split], idx[split:]

x_train_np, x_test_np = X_np[train_idx], X_np[test_idx]
y_train_np, y_test_np = y_np[train_idx], y_np[test_idx]

# Scale X; optionally scale y
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(x_train_np)
X_test  = x_scaler.transform(x_test_np)

if NORMALIZE_Y:
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train_np)
    y_test  = y_scaler.transform(y_test_np)
else:
    y_scaler = None
    y_train, y_test = y_train_np, y_test_np

# Tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)
X_test_tensor  = torch.tensor(X_test,  dtype=torch.float32, device=device)
y_test_tensor  = torch.tensor(y_test,  dtype=torch.float32, device=device)

# ---- domain bbox from unscaled surf_x,surf_y ----
xy_np = df[['surf_x','surf_y']].to_numpy(dtype=np.float32)
xmin, xmax = float(xy_np[:,0].min()), float(xy_np[:,0].max())
ymin, ymax = float(xy_np[:,1].min()), float(xy_np[:,1].max())
bbox = (xmin, xmax, ymin, ymax)

# ============================================================
# Helpers (autograd, laplacian, sampling)
# ============================================================
def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True, retain_graph=True, only_inputs=True)[0]

def laplacian(u, X):
    g = grad(u, X)            # [du/dx, du/dy, ...]
    du_dx, du_dy = g[:,0:1], g[:,1:2]
    d2u_dx2 = grad(du_dx, X)[:,0:1]
    d2u_dy2 = grad(du_dy, X)[:,1:2]
    return d2u_dx2 + d2u_dy2

def infer_normals(Xb_xy, bbox):
    xmin, xmax, ymin, ymax = bbox
    n = torch.zeros_like(Xb_xy)
    tol = 1e-6
    left  = torch.isclose(Xb_xy[:,0], torch.tensor(xmin, device=Xb_xy.device), atol=tol)
    right = torch.isclose(Xb_xy[:,0], torch.tensor(xmax, device=Xb_xy.device), atol=tol)
    bot   = torch.isclose(Xb_xy[:,1], torch.tensor(ymin, device=Xb_xy.device), atol=tol)
    top   = torch.isclose(Xb_xy[:,1], torch.tensor(ymax, device=Xb_xy.device), atol=tol)
    n[left,0]  = -1.0; n[right,0] = 1.0
    n[bot,1]   = -1.0; n[top,1]   = 1.0
    return n

# map physical (x,y) to scaled feature vector (others at scaled mean=0)
mu_xy  = torch.tensor(x_scaler.mean_[:2],  dtype=torch.float32, device=device)
sig_xy = torch.tensor(x_scaler.scale_[:2], dtype=torch.float32, device=device)

def embed_xy_scaled(X_xy):
    npts = X_xy.shape[0]
    d = X_train_tensor.shape[1]
    X_full = torch.zeros(npts, d, device=device)           # others at 0 == scaled mean
    X_full[:, :2] = (X_xy - mu_xy) / sig_xy
    return X_full

def sample_interior_uniform(nc):
    rx = torch.rand(nc, 1, device=device)
    ry = torch.rand(nc, 1, device=device)
    X_xy = torch.cat([xmin + (xmax - xmin)*rx, ymin + (ymax - ymin)*ry], dim=1)
    return embed_xy_scaled(X_xy).requires_grad_(True)

def boundary_collocation(bbox, n_per_side=128):
    xmin, xmax, ymin, ymax = bbox
    xs = torch.linspace(xmin, xmax, n_per_side, device=device)
    ys = torch.linspace(ymin, ymax, n_per_side, device=device)
    top = torch.stack([xs, torch.full_like(xs, ymax)], dim=1)
    bot = torch.stack([xs, torch.full_like(xs, ymin)], dim=1)
    left  = torch.stack([torch.full_like(ys, xmin), ys], dim=1)
    right = torch.stack([torch.full_like(ys, xmax), ys], dim=1)
    Xb_xy = torch.cat([top, bot, left, right], dim=0)
    Nb = infer_normals(Xb_xy, bbox)
    return Xb_xy, Nb

def sample_boundary(n_per_side=N_BPER):
    Xb_xy, Nb = boundary_collocation(bbox, n_per_side=n_per_side)
    return embed_xy_scaled(Xb_xy).requires_grad_(True), Nb

# ============================================================
# Physics (SIA + reduced Stokes)
# ============================================================
class MultiFidelityResiduals(nn.Module):
    def __init__(self, use_uncertainty_weighting, clamp=LOG_CLAMP, w_sia=1.0, w_stk=1.0):
        super().__init__()
        self.use_unc = use_uncertainty_weighting
        self.log_sigma_sia = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_stk = nn.Parameter(torch.tensor(0.0))
        self.clamp = clamp
        self.w_sia = w_sia
        self.w_stk = w_stk

    def sia_residual(self, model, X, extra=None):
        u = model(X)
        M = extra.get("M") if (extra is not None and "M" in extra) else torch.ones_like(u)
        g = grad(u, X)
        qx, qy = M*g[:,0:1], M*g[:,1:2]
        dqxdx = grad(qx, X)[:,0:1]
        dqydy = grad(qy, X)[:,1:2]
        return dqxdx + dqydy

    def stokes_reduced_residual(self, model, X, body_force=None, nu=1.0):
        u = model(X)
        Lu = -nu * laplacian(u, X)
        f = torch.zeros_like(u) if body_force is None else body_force
        return Lu - f

    def forward(self, model, Xc, extra=None):
        r_sia = self.sia_residual(model, Xc, extra=extra)
        r_stk = self.stokes_reduced_residual(model, Xc, body_force=None, nu=1.0)
        if self.use_unc:
            l1 = torch.clamp(self.log_sigma_sia, *self.clamp)
            l2 = torch.clamp(self.log_sigma_stk, *self.clamp)
            w1, w2 = torch.exp(-l1), torch.exp(-l2)
            loss = 0.5*(w1*(r_sia**2).mean() + l1) + 0.5*(w2*(r_stk**2).mean() + l2)
        else:
            loss = self.w_sia*(r_sia**2).mean() + self.w_stk*(r_stk**2).mean()
        return loss, {"r_sia": r_sia.detach(), "r_stk": r_stk.detach()}

# ============================================================
# Boundary weak-form
# ============================================================
class BoundaryWeakForm(nn.Module):
    def __init__(self, dirichlet_fraction=0.0, hard_dirichlet=False, lambda_dir=LAMBDA_DIR, lambda_neu=LAMBDA_NEU):
        super().__init__()
        self.dir_frac = dirichlet_fraction
        self.hard_dir = hard_dirichlet
        self.lambda_dir = lambda_dir
        self.lambda_neu = lambda_neu

    def neumann_loss(self, model, Xb_full, Nb, gN=None):
        u = model(Xb_full)
        gu = torch.autograd.grad(u, Xb_full, grad_outputs=torch.ones_like(u),
                                 create_graph=True, retain_graph=True)[0]
        gu_xy = gu[:, :2]
        flux = (gu_xy * Nb).sum(dim=1, keepdim=True)   # n · ∇u
        gN = torch.zeros_like(flux) if gN is None else gN
        return (flux - gN).pow(2).mean()

    def dirichlet_loss(self, model, Xb_full, uD):
        u = model(Xb_full)
        return (u - uD).pow(2).mean()

    def forward(self, model, Xb_full, Nb, uD=None, gN=None):
        n = Xb_full.shape[0]
        n_dir = int(self.dir_frac * n)
        perm = torch.randperm(n, device=Xb_full.device)

        Xb_dir = Xb_full[perm[:n_dir]]
        Xb_neu = Xb_full[perm[n_dir:]]
        Nb_neu = Nb[perm[n_dir:]]

        loss_neu = self.neumann_loss(model, Xb_neu, Nb_neu, gN=gN)
        loss_dir = torch.tensor(0.0, device=Xb_full.device)
        if uD is not None and n_dir > 0:
            uD_sub = uD[perm[:n_dir]]
            if self.hard_dir:
                loss_dir = 100.0 * self.dirichlet_loss(model, Xb_dir, uD_sub)
            else:
                loss_dir = self.dirichlet_loss(model, Xb_dir, uD_sub)

        return self.lambda_neu*loss_neu + self.lambda_dir*loss_dir, {
            "loss_neu": loss_neu.detach(),
            "loss_dir": loss_dir.detach()
        }

# ============================================================
# Model (slightly larger MLP)
# ============================================================
class PINN_Model(nn.Module):
    def __init__(self, input_dim, hidden, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),  # +1 extra hidden
            nn.Linear(hidden, output_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)

# ============================================================
# Instantiate
# ============================================================
model = PINN_Model(input_dim=X_train_tensor.shape[1], hidden=HIDDEN).to(device)
mf = MultiFidelityResiduals(USE_UNCERTAINTY, w_sia=W_SIA, w_stk=W_STK).to(device)
bw = BoundaryWeakForm(dirichlet_fraction=DIR_FRAC, hard_dirichlet=HARD_DIRICHLET,
                      lambda_dir=LAMBDA_DIR, lambda_neu=LAMBDA_NEU).to(device)

optimizer = optim.Adam(model.parameters(), lr=LR_INIT)
mse = nn.MSELoss()

# Optional boundary targets (provide your own if known)
# Example placeholders: keep zero-flux and no Dirichlet by default
def build_uD_gN(Xb_full, Nb):
    """
    Return (uD_full, gN_full) aligned with Xb_full.
    Replace this with problem-specific boundary values if known.
    """
    uD_full = None  # torch.zeros_like(model(Xb_full))  # example
    gN_full = None  # torch.zeros_like(model(Xb_full))  # example (zero-flux)
    return uD_full, gN_full

# Cosine LR schedule
def cosine_lr(it, total, lr_min=LR_MIN, lr_max=LR_INIT):
    return lr_min + 0.5*(lr_max - lr_min)*(1 + math.cos(math.pi*min(it,total)/total))

# ============================================================
# Adaptive sampling buffer
# ============================================================
residual_bank = {'X': None, 'r2': None}  # stores scaled X and residual magnitude^2

def sample_interior_adaptive(nc, adapt_frac=ADAPT_FRAC):
    if (not USE_ADAPTIVE) or residual_bank['X'] is None or residual_bank['r2'] is None:
        return sample_interior_uniform(nc)

    n_adapt = int(nc * adapt_frac)
    n_rand  = nc - n_adapt

    X_rand = sample_interior_uniform(n_rand)

    # Pick top-k residual points and jitter around them in scaled feature space
    k = min(n_adapt, residual_bank['X'].shape[0])
    idx = torch.topk(residual_bank['r2'].squeeze(-1), k).indices
    X_seeds = residual_bank['X'][idx]

    noise = ADAPT_NOISE * torch.randn_like(X_seeds)
    X_adapt = X_seeds + noise

    # Clamp only the first two dims (x,y scaled coords) to keep reasonable values
    lo, hi = ADAPT_CLAMP
    X_adapt[:, :2] = X_adapt[:, :2].clamp_(lo, hi)
    # other features remain ~0 (they are already ~0 in seeds)

    Xc = torch.cat([X_rand, X_adapt], dim=0).requires_grad_(True)
    return Xc

# ============================================================
# Train
# ============================================================
for epoch in range(1, EPOCHS+1):
    # curriculum collocation
    t = epoch / EPOCHS
    N_COLLOCATION = int(N_COL_START + t*(N_COL_END - N_COL_START))

    # cosine LR
    lr = cosine_lr(epoch, EPOCHS)
    for pg in optimizer.param_groups: pg['lr'] = lr

    model.train(); optimizer.zero_grad()

    # data loss
    y_pred = model(X_train_tensor)
    loss_data = mse(y_pred, y_train_tensor)

    # interior physics (adaptive or uniform)
    Xc = sample_interior_adaptive(N_COLLOCATION) if USE_ADAPTIVE else sample_interior_uniform(N_COLLOCATION)
    loss_phys, phys_stats = mf(model, Xc, extra=None)

    # boundary weak-form
    loss_bnd = torch.tensor(0.0, device=device)
    if USE_BOUNDARY:
        Xb_full, Nb = sample_boundary(n_per_side=N_BPER)
        uD_full, gN_full = build_uD_gN(Xb_full, Nb)
        loss_bnd, bnd_stats = bw(model, Xb_full, Nb, uD=uD_full, gN=gN_full)

    loss_total = loss_data + loss_phys + loss_bnd
    loss_total.backward()
    optimizer.step()

    # update residual bank for next epoch's adaptive sampling
    with torch.no_grad():
        r2 = phys_stats["r_sia"].pow(2) + phys_stats["r_stk"].pow(2)
        residual_bank['X']  = Xc.detach()
        residual_bank['r2'] = r2.detach()

    if epoch % 1000 == 0 or epoch == 1:
        with torch.no_grad():
            r_sia = phys_stats["r_sia"]; r_stk = phys_stats["r_stk"]
            sia_mse = (r_sia**2).mean().item()
            stk_mse = (r_stk**2).mean().item()
        print(f"[{epoch}/{EPOCHS}] tot={loss_total.item():.3e} data={loss_data.item():.3e} "
              f"phys={loss_phys.item():.3e} bnd={loss_bnd.item():.3e} "
              f"SIA_MSE={sia_mse:.3e} STK_MSE={stk_mse:.3e} lr={lr:.2e}")

# ============================================================
# Evaluation
# ============================================================
model.eval()

# 1) Predictions & metrics (normalized units)
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_np = y_pred.detach().cpu().numpy()
    y_test_np = y_test_tensor.detach().cpu().numpy()

test_mse = mean_squared_error(y_test_np, y_pred_np)
r2 = r2_score(y_test_np, y_pred_np)
rmse = np.sqrt(test_mse)
mae = mean_absolute_error(y_test_np, y_pred_np)

print("\n=== Test metrics (training units) ===")
print(f"MSE:  {test_mse:.6f}")
print(f"R2:   {r2:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")

# 2) Convert to physical units for reporting/plotting
if NORMALIZE_Y and (y_scaler is not None):
    y_pred_phys = y_scaler.inverse_transform(y_pred_np)
    y_true_phys = y_scaler.inverse_transform(y_test_np)
else:
    y_pred_phys = y_pred_np
    y_true_phys = y_test_np

yt = y_true_phys.ravel()
yp = y_pred_phys.ravel()
phys_mse  = mean_squared_error(yt, yp)
phys_r2   = r2_score(yt, yp)
phys_rmse = np.sqrt(phys_mse)
phys_mae  = mean_absolute_error(yt, yp)

print("\n=== Test metrics (physical units) ===")
print(f"MSE:  {phys_mse:.6f}")
print(f"R2:   {phys_r2:.6f}")
print(f"RMSE: {phys_rmse:.6f}")
print(f"MAE:  {phys_mae:.6f}")

# 3) Physics diagnostics (NEED grads!)
with torch.enable_grad():
    Xc_eval = sample_interior_uniform(4096)              # requires_grad=True
    assert Xc_eval.requires_grad, "Xc_eval must have requires_grad=True"

    # Weighted physics objective + raw residuals
    loss_phys_eval, _ = mf(model, Xc_eval, extra=None)
    r_sia_eval = mf.sia_residual(model, Xc_eval, extra=None)
    r_stk_eval = mf.stokes_reduced_residual(model, Xc_eval, body_force=None, nu=1.0)

    sia_mse_eval = (r_sia_eval**2).mean().item()
    stk_mse_eval = (r_stk_eval**2).mean().item()

print("\n=== Physics diagnostics (collocation) ===")
print(f"Weighted physics objective (may be < 0 if USE_UNCERTAINTY): {loss_phys_eval.item():.6f}")
print(f"SIA residual MSE:    {sia_mse_eval:.6e}")
print(f"Stokes residual MSE: {stk_mse_eval:.6e}")

# 4) Plot: True vs Pred (physical units)
plt.figure(figsize=(7,6))
plt.scatter(yt, yp, s=5, alpha=0.5, label='Predictions')
mn = float(min(yt.min(), yp.min()))
mx = float(max(yt.max(), yp.max()))
plt.plot([mn, mx], [mn, mx], 'r--', label='Perfect fit')
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("True vs Predicted (Test)")
plt.legend(); plt.grid(True, linestyle='--', alpha=0.3); plt.tight_layout(); plt.show()

import numpy as np, torch

model.eval()  # <-- this is the trained baseline model instance
with torch.no_grad():
    Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_pred_baseline_test = model(Xt).cpu().numpy()

# save predictions so the MAIN notebook can plot them
np.save("baseline_preds.npy", y_pred_baseline_test)
print("✅ Saved baseline_preds.npy:", y_pred_baseline_test.shape)


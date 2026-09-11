"""
First-arrival traveltime tomography on the synthetic data.

Closes the loop on the whole project: the DEM gives strain, the rock
physics gives Vp, the eikonal solver gives traveltimes -- and here those
traveltimes are treated as *observed data* and inverted back to a
velocity model, the way a real refraction survey would be processed.

The question this answers: of the structure each rock-physics model
puts into Vp, how much would a seismic survey actually be able to
recover?

Method (standard first-arrival tomography):
  - survey: N_SHOTS shots along the free surface, receivers at every
    REC_STEP-th surface grid node
  - forward: eikonal (fast marching) per shot on the fine grid
  - rays: back-traced from each receiver down the gradient of the
    traveltime field to the shot, accumulating path length per cell of
    a coarser INVERSION grid
  - update: damped, Laplacian-smoothed least squares for the slowness
    perturbation, solved with LSQR; repeated for N_ITER iterations
  - all three models are inverted from the SAME starting model (a plain
    linear v(z) gradient), so the comparison is fair
"""
import numpy as np
import skfmm
from scipy import sparse
from scipy.sparse.linalg import lsqr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_SHOTS = 15
REC_STEP = 3          # receiver every 3rd grid column
N_ITER = 12
CELL = 300.0          # m, inversion cell size
SMOOTH_REL = 1.0      # Laplacian weight, relative to the RMS entry of G
DAMP_REL = 0.2        # damping weight, relative to the RMS entry of G
MAX_DV = 0.25         # cap on the per-iteration fractional velocity change
V_START_TOP, V_START_BOT = 1500.0, 3500.0   # starting model, same for all


def surface_rows(inside):
    """Topmost in-domain row index for each column (the free surface)."""
    rows = np.full(inside.shape[1], -1)
    for k in range(inside.shape[1]):
        col = np.where(inside[:, k])[0]
        if len(col):
            rows[k] = col.max()
    return rows


def eikonal(vp, dx, si, sj):
    phi = np.ones_like(vp)
    phi[si, sj] = -1
    return skfmm.travel_time(phi, vp, dx=dx)


def trace_ray(grads, dx, ri, rj, si, sj, cell_of_node, n_cells, max_steps=6000):
    """Back-trace one ray from receiver (ri, rj) down grad(T) to the shot.
    `grads` is (dT/dy, dT/dx), precomputed once per shot.
    Returns (cell_indices, path_lengths) on the inversion grid."""
    gy_grad, gx_grad = grads
    ny, nx = gy_grad.shape
    y, x = float(ri), float(rj)          # in grid-index units
    step = 0.5                            # half a cell per step
    acc = np.zeros(n_cells)
    for _ in range(max_steps):
        if not (0 <= y <= ny - 1 and 0 <= x <= nx - 1):
            break
        if abs(y - si) <= 1 and abs(x - sj) <= 1:
            break
        # bilinear interpolation of the traveltime gradient
        y0, x0 = int(np.floor(y)), int(np.floor(x))
        y0 = min(max(y0, 0), ny - 2); x0 = min(max(x0, 0), nx - 2)
        fy, fx = y - y0, x - x0
        w = np.array([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx])
        gy_ = w @ np.array([gy_grad[y0, x0], gy_grad[y0, x0+1], gy_grad[y0+1, x0], gy_grad[y0+1, x0+1]])
        gx_ = w @ np.array([gx_grad[y0, x0], gx_grad[y0, x0+1], gx_grad[y0+1, x0], gx_grad[y0+1, x0+1]])
        n = np.hypot(gx_, gy_)
        if n < 1e-12:
            break
        y -= step * gy_ / n
        x -= step * gx_ / n
        iy2, ix2 = int(round(y)), int(round(x))
        if 0 <= iy2 < ny and 0 <= ix2 < nx:
            acc[cell_of_node[iy2, ix2]] += step * dx
    nz = np.nonzero(acc)[0]
    return nz, acc[nz]


def build_survey(inside, nx):
    srow = surface_rows(inside)
    valid_cols = np.where(srow >= 0)[0]
    shot_cols = valid_cols[np.linspace(0, len(valid_cols) - 1, N_SHOTS).astype(int)]
    rec_cols = valid_cols[::REC_STEP]
    return srow, shot_cols, rec_cols


def forward(vp, dx, inside, srow, shot_cols, rec_cols):
    """Traveltimes at all receivers for all shots; also return the T fields."""
    data, fields = [], []
    for sj in shot_cols:
        si = srow[sj]
        T = eikonal(vp, dx, si, sj)
        fields.append(T)
        data.append(np.array([T[srow[rj], rj] for rj in rec_cols]))
    return np.concatenate(data), fields


def laplacian(n_cy, n_cx):
    """2D Laplacian smoothing operator on the inversion grid."""
    rows, cols, vals = [], [], []
    r = 0
    for iy in range(n_cy):
        for ix in range(n_cx):
            c = iy * n_cx + ix
            nb = []
            if iy > 0: nb.append((iy - 1) * n_cx + ix)
            if iy < n_cy - 1: nb.append((iy + 1) * n_cx + ix)
            if ix > 0: nb.append(iy * n_cx + ix - 1)
            if ix < n_cx - 1: nb.append(iy * n_cx + ix + 1)
            rows.append(r); cols.append(c); vals.append(float(len(nb)))
            for j in nb:
                rows.append(r); cols.append(j); vals.append(-1.0)
            r += 1
    return sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_cy * n_cx))


def invert(vp_true, dx, gx, gy, inside, label):
    ny, nx = vp_true.shape
    srow, shot_cols, rec_cols = build_survey(inside, nx)

    # observed data from the true model
    d_obs, _ = forward(vp_true, dx, inside, srow, shot_cols, rec_cols)

    # inversion grid + node -> cell lookup
    n_cx = int(np.ceil((gx[-1] - gx[0]) / CELL)) + 1
    n_cy = int(np.ceil((gy[-1] - gy[0]) / CELL)) + 1
    n_cells = n_cx * n_cy
    IY, IX = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cy = np.clip(((gy[IY] - gy[0]) / CELL).astype(int), 0, n_cy - 1)
    cx = np.clip(((gx[IX] - gx[0]) / CELL).astype(int), 0, n_cx - 1)
    cell_of_node = cy * n_cx + cx

    # starting model: plain linear gradient, identical for every run
    t = (gy[:, None] - gy[0]) / (gy[-1] - gy[0])
    vp = np.repeat(V_START_BOT + (V_START_TOP - V_START_BOT) * t, nx, axis=1)

    L = laplacian(n_cy, n_cx)
    I = sparse.identity(n_cells, format="csr")

    def apply_update(vp_cur, ds, alpha):
        slow = 1.0 / vp_cur
        # cap the fractional velocity change so a single step cannot run away
        d_slow = np.clip(alpha * ds[cell_of_node], -MAX_DV * slow, MAX_DV * slow)
        return 1.0 / np.clip(slow + d_slow, 1.0 / 6000.0, 1.0 / 300.0)

    d_cal, fields = forward(vp, dx, inside, srow, shot_cols, rec_cols)
    res = d_obs - d_cal
    rms = np.sqrt(np.mean(res ** 2))
    rms_hist = [rms]
    print(f"  [{label}] iter  0: traveltime RMS residual {rms*1000:7.1f} ms")

    for it in range(1, N_ITER):
        rows, cols, vals = [], [], []
        r = 0
        for s, sj in enumerate(shot_cols):
            si, T = srow[sj], fields[s]
            grads = np.gradient(T, dx)
            for rj in rec_cols:
                idx, ln = trace_ray(grads, dx, srow[rj], rj, si, sj, cell_of_node, n_cells)
                rows.extend([r] * len(idx)); cols.extend(idx); vals.extend(ln)
                r += 1
        G = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_cells))

        # scale the regularisation to G itself -- G entries are ray path
        # lengths in metres, so an absolute weight would be meaningless
        g_scale = np.sqrt(G.multiply(G).sum() / max(G.nnz, 1))
        A = sparse.vstack([G, SMOOTH_REL * g_scale * L, DAMP_REL * g_scale * I]).tocsr()
        b = np.concatenate([res, np.zeros(L.shape[0]), np.zeros(n_cells)])
        ds = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=600)[0]

        # backtracking line search: only accept a step that actually helps
        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
            vp_try = apply_update(vp, ds, alpha)
            d_try, fields_try = forward(vp_try, dx, inside, srow, shot_cols, rec_cols)
            res_try = d_obs - d_try
            rms_try = np.sqrt(np.mean(res_try ** 2))
            if rms_try < rms:
                vp, res, rms, fields = vp_try, res_try, rms_try, fields_try
                accepted = True
                break
        rms_hist.append(rms)
        print(f"  [{label}] iter {it:2d}: traveltime RMS residual {rms*1000:7.1f} ms"
              f"{'' if accepted else '   (no improving step -- stopping)'}")
        if not accepted:
            break

    return vp, np.array(rms_hist)


def main():
    d = np.load("traveltime_compare.npz")
    gx, gy, inside = d["gx"], d["gy"], d["inside"]
    names = [str(n) for n in d["names"]]
    vp_grids = d["vp_grids"]
    dx = float(gx[1] - gx[0])

    results = {}
    for r, name in enumerate(names):
        print(f"inverting: {name}")
        results[name] = invert(vp_grids[r], dx, gx, gy, inside, name)

    # residual histories can differ in length (early stop) -- pad with NaN
    rms_pad = np.full((len(names), N_ITER), np.nan)
    for i, n in enumerate(names):
        h = results[n][1]
        rms_pad[i, :len(h)] = h

    np.savez("tomography.npz", gx=gx, gy=gy, inside=inside,
             names=np.array(names),
             vp_true=vp_grids,
             vp_inv=np.array([results[n][0] for n in names]),
             rms=rms_pad)
    print("saved tomography.npz")

    # ---------------- figures ----------------
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
        "axes.edgecolor": "#555", "axes.labelcolor": "#222",
        "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
    })
    extent = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]
    vmin = min(np.nanmin(np.ma.array(v, mask=~inside)) for v in vp_grids)
    vmax = max(np.nanmax(np.ma.array(v, mask=~inside)) for v in vp_grids)

    fig, axes = plt.subplots(3, 3, figsize=(16, 7.8))
    for r, name in enumerate(names):
        true = np.ma.array(vp_grids[r], mask=~inside)
        inv = np.ma.array(results[name][0], mask=~inside)
        diff = inv - true

        im0 = axes[r, 0].imshow(true, origin="lower", extent=extent, aspect="equal",
                                cmap="viridis", vmin=vmin, vmax=vmax)
        axes[r, 0].set_title(f"{name} — true $V_P$", fontsize=10.5)
        plt.colorbar(im0, ax=axes[r, 0], shrink=0.88, pad=0.01)

        im1 = axes[r, 1].imshow(inv, origin="lower", extent=extent, aspect="equal",
                                cmap="viridis", vmin=vmin, vmax=vmax)
        axes[r, 1].set_title(f"{name} — recovered", fontsize=10.5)
        plt.colorbar(im1, ax=axes[r, 1], shrink=0.88, pad=0.01)

        lim = np.nanpercentile(np.abs(diff.compressed()), 98)
        im2 = axes[r, 2].imshow(diff, origin="lower", extent=extent, aspect="equal",
                                cmap="RdBu_r", vmin=-lim, vmax=lim)
        axes[r, 2].set_title(f"{name} — recovered − true (m/s)", fontsize=10.5)
        plt.colorbar(im2, ax=axes[r, 2], shrink=0.88, pad=0.01)

        for c in range(3):
            axes[r, c].set_ylabel("Z (km)")
    for c in range(3):
        axes[-1, c].set_xlabel("X (km)")

    fig.suptitle(f"First-arrival traveltime tomography: what a survey would recover "
                 f"({N_SHOTS} shots, {N_ITER} iterations, identical linear starting model)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig_tomo_models.png", dpi=150)
    print("saved fig_tomo_models.png")

    # convergence + recovery metrics
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.3))
    COLORS = {"SSPX + Botter": "#2f6fb3", "IG-FEM + Botter": "#3f8f6b", "Hertz-Mindlin": "#c0562d"}
    for name in names:
        axes2[0].plot(results[name][1] * 1000, "o-", color=COLORS[name], lw=2, label=name)
    axes2[0].set_xlabel("iteration"); axes2[0].set_ylabel("traveltime RMS residual (ms)")
    axes2[0].set_title("Convergence"); axes2[0].legend(fontsize=9); axes2[0].set_yscale("log")

    depth_km = (gy - gy.min()) / 1000
    for name in names:
        r = names.index(name)
        true = np.ma.array(vp_grids[r], mask=~inside)
        inv = np.ma.array(results[name][0], mask=~inside)
        axes2[1].plot(true.mean(axis=1), depth_km, color=COLORS[name], lw=2, label=f"{name} (true)")
        axes2[1].plot(inv.mean(axis=1), depth_km, color=COLORS[name], lw=1.6, ls="--",
                      label=f"{name} (recovered)")
    axes2[1].set_xlabel("laterally averaged $V_P$ (m/s)"); axes2[1].set_ylabel("height above base (km)")
    axes2[1].set_title("Vertical profile: true (solid) vs recovered (dashed)")
    axes2[1].legend(fontsize=7.5)

    fig2.suptitle("Tomographic recovery of the three rock-physics models", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    fig2.savefig("fig_tomo_recovery.png", dpi=150)
    print("saved fig_tomo_recovery.png")

    print()
    for r, name in enumerate(names):
        true = vp_grids[r][inside]
        inv = results[name][0][inside]
        print(f"{name:>16s}: corr(true, recovered) {np.corrcoef(true, inv)[0,1]:+.3f}   "
              f"mean |err| {np.mean(np.abs(inv-true)):6.0f} m/s   "
              f"RMS residual {results[name][1][0]*1000:.0f} -> {results[name][1][-1]*1000:.0f} ms")

    # Split the recovery into the part a refraction survey is good at (the
    # 1D compaction trend) and the part it is not (lateral structure, i.e.
    # the shear bands), by removing the laterally averaged profile.
    print("\nrecovery split into 1D trend vs lateral anomaly:")
    for r, name in enumerate(names):
        T = np.ma.array(vp_grids[r], mask=~inside)
        R = np.ma.array(results[name][0], mask=~inside)
        pt, pr = T.mean(axis=1), R.mean(axis=1)
        Ta, Ra = T - pt[:, None], R - pr[:, None]
        print(f"  {name:>16s}: 1D trend corr {np.corrcoef(pt.compressed(), pr.compressed())[0,1]:+.4f} "
              f"(mean |err| {np.abs(pt - pr).mean():3.0f} m/s)   |   "
              f"lateral anomaly corr {np.corrcoef(Ta[inside], Ra[inside])[0,1]:+.3f}, "
              f"amplitude ratio {Ra[inside].std() / Ta[inside].std():.2f}")

    # can tomography still tell the models apart?
    print("\npairwise separability after inversion (corr between recovered models):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = results[names[i]][0][inside], results[names[j]][0][inside]
            at, bt = vp_grids[i][inside], vp_grids[j][inside]
            print(f"  {names[i]} vs {names[j]}:  true {np.corrcoef(at,bt)[0,1]:+.3f}  "
                  f"-> recovered {np.corrcoef(a,b)[0,1]:+.3f}")


if __name__ == "__main__":
    main()

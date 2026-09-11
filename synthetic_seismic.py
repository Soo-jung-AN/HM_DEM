"""
Step 3 of the Botter et al. (2014) workflow, in its lightweight form:
turn the synthetic elastic properties into a zero-offset seismic
section by 1D convolutional modelling.

Botter et al. used a ray-based PSDM simulator (SeisRoX) to include
illumination and resolution effects. They also note that with a
"perfect PSDM filter" -- all reflector dips illuminated -- the result
approximates plain 1D convolution, which is what is done here:

  1. acoustic impedance          Z = rho * Vp
  2. depth -> two-way time       t(z) = 2 * integral dz / Vp
  3. normal-incidence reflectivity in time,  R = dZ / (2 Z)
  4. convolve with a zero-phase Ricker wavelet of dominant frequency f
  5. map the trace back to depth, i.e. a perfectly-illuminated,
     perfectly-migrated depth section

Run for all three Vp/rho models built in traveltime_compare.py, so the
three rock-physics routes can be compared on a synthetic seismic image
the way they were compared on traveltime.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DT = 0.002          # s, time sampling for the convolution
FREQS = [10, 20, 30, 40]   # Hz, as in Botter et al. Figs. 11-12
F_MAIN = 30         # Hz, frequency used for the 3-model comparison


def ricker(f, dt, length=0.512):
    """Zero-phase Ricker wavelet, dominant frequency f."""
    t = np.arange(-length / 2, length / 2 + dt, dt)
    a = (np.pi * f * t) ** 2
    return t, (1 - 2 * a) * np.exp(-a)


def trace_synthetic(z, vp, rho, f, dt=DT):
    """One trace: depth-sampled Vp/rho (z increasing downward, uniform dz)
    -> depth-domain seismic amplitude, via a time-domain convolution."""
    if len(z) < 4:
        return np.zeros_like(z)
    dz = z[1] - z[0]
    twt = np.concatenate([[0.0], np.cumsum(2 * dz / vp[:-1])])   # s

    nt = int(np.ceil(twt[-1] / dt)) + 1
    t_uniform = np.arange(nt) * dt
    Z = rho * vp
    Z_t = np.interp(t_uniform, twt, Z)

    refl = np.zeros(nt)
    refl[:-1] = np.diff(Z_t) / (Z_t[:-1] + Z_t[1:])             # normal incidence

    _, w = ricker(f, dt)
    seis_t = np.convolve(refl, w, mode="same")
    return np.interp(twt, t_uniform, seis_t)                     # back to depth


def section(vp_grid, rho_grid, inside, gy, f):
    """Full 2D section: one trace per column, surface downward."""
    ny, nx = vp_grid.shape
    out = np.full((ny, nx), np.nan)
    dz = gy[1] - gy[0]
    for k in range(nx):
        col = np.where(inside[:, k])[0]
        if len(col) < 4:
            continue
        top, bot = col.max(), col.min()
        rows = np.arange(top, bot - 1, -1)          # downward from the surface
        z = np.arange(len(rows)) * dz
        tr = trace_synthetic(z, vp_grid[rows, k], rho_grid[rows, k], f)
        out[rows, k] = tr
    return out


def main():
    d = np.load("traveltime_compare.npz")
    gx, gy, inside = d["gx"], d["gy"], d["inside"]
    names = [str(n) for n in d["names"]]
    vp_grids, rho_grids = d["vp_grids"], d["rho_grids"]

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
        "axes.edgecolor": "#555", "axes.labelcolor": "#222",
        "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
    })
    extent = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]

    # ---- Figure A: impedance, reflectivity-bearing section, per model ----
    fig, axes = plt.subplots(3, 2, figsize=(15, 7.5))
    sections = {}
    for r, name in enumerate(names):
        Z = np.ma.array(vp_grids[r] * rho_grids[r], mask=~inside)
        im = axes[r, 0].imshow(Z / 1e6, origin="lower", extent=extent, aspect="equal",
                               cmap="cividis")
        axes[r, 0].set_title(f"{name}  —  acoustic impedance (10⁶ kg·m⁻²s⁻¹)", fontsize=11)
        plt.colorbar(im, ax=axes[r, 0], shrink=0.9, pad=0.01)

        s = section(vp_grids[r], rho_grids[r], inside, gy, F_MAIN)
        sections[name] = s
        lim = np.nanpercentile(np.abs(s), 99)
        im2 = axes[r, 1].imshow(np.ma.array(s, mask=~inside), origin="lower", extent=extent,
                                aspect="equal", cmap="gray", vmin=-lim, vmax=lim)
        axes[r, 1].set_title(f"{name}  —  synthetic section, {F_MAIN} Hz", fontsize=11)
        plt.colorbar(im2, ax=axes[r, 1], shrink=0.9, pad=0.01)

        for c in (0, 1):
            axes[r, c].set_ylabel("Z (km)")
    for c in (0, 1):
        axes[-1, c].set_xlabel("X (km)")

    fig.suptitle("Botter et al. step 3, lightweight form: impedance → 1D-convolution synthetic section "
                 f"({F_MAIN} Hz zero-phase Ricker, perfect illumination)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("fig_seismic_models.png", dpi=150)
    print("saved fig_seismic_models.png")

    # ---- Figure B: frequency sweep on one model (cf. Botter Figs. 11-12) ----
    ref_name = "IG-FEM + Botter"
    r = names.index(ref_name)
    fig2, axes2 = plt.subplots(len(FREQS), 1, figsize=(13, 9), sharex=True)
    for a, f in zip(axes2, FREQS):
        s = section(vp_grids[r], rho_grids[r], inside, gy, f)
        lim = np.nanpercentile(np.abs(s), 99)
        a.imshow(np.ma.array(s, mask=~inside), origin="lower", extent=extent,
                 aspect="equal", cmap="gray", vmin=-lim, vmax=lim)
        a.set_title(f"{f} Hz", fontsize=11)
        a.set_ylabel("Z (km)")
    axes2[-1].set_xlabel("X (km)")
    fig2.suptitle(f"Wave-frequency control on what the shear bands look like — {ref_name}", fontsize=12)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig("fig_seismic_frequency.png", dpi=150)
    print("saved fig_seismic_frequency.png")

    # ---- quantitative: how similar are the three sections? ----
    print()
    ref = sections[ref_name]
    m = np.isfinite(ref)
    for name in names:
        s = sections[name]
        mm = m & np.isfinite(s)
        cc = np.corrcoef(s[mm], ref[mm])[0, 1]
        rms = np.sqrt(np.nanmean((s[mm] - ref[mm]) ** 2)) / np.sqrt(np.nanmean(ref[mm] ** 2))
        print(f"{name:>16s}  vs {ref_name}:  corr {cc:+.3f}   rel. RMS diff {rms:.3f}")


if __name__ == "__main__":
    main()

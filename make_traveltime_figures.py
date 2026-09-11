import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("traveltime_compare.npz")
gx, gy = d["gx"], d["gy"]
inside = d["inside"]
src_x, src_y = float(d["src_x"]), float(d["src_y"])
names = [str(n) for n in d["names"]]
vp_grids, tts = d["vp_grids"], d["tts"]

COLORS = {"SSPX + Botter": "#2f6fb3", "IG-FEM + Botter": "#3f8f6b", "Hertz-Mindlin": "#c0562d"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})

extent = [gx.min() / 1000, gx.max() / 1000, gy.min() / 1000, gy.max() / 1000]
vmin_vp = min(np.nanmin(np.ma.array(v, mask=~inside)) for v in vp_grids)
vmax_vp = max(np.nanmax(np.ma.array(v, mask=~inside)) for v in vp_grids)
tt_max = max(np.ma.array(t, mask=~inside).max() for t in tts)

# ---------------------------------------------------------------
# Figure A: Vp fields and traveltime fields, true (equal) aspect
# ---------------------------------------------------------------
fig, axes = plt.subplots(3, 2, figsize=(15, 7.5))
levels = np.arange(0, tt_max + 0.25, 0.25)

for r, name in enumerate(names):
    vp = np.ma.array(vp_grids[r], mask=~inside)
    tt = np.ma.array(tts[r], mask=~inside)

    im = axes[r, 0].imshow(vp, origin="lower", extent=extent, aspect="equal",
                           cmap="viridis", vmin=vmin_vp, vmax=vmax_vp)
    axes[r, 0].set_title(f"{name}  —  $V_P$ (m/s)", fontsize=11)
    plt.colorbar(im, ax=axes[r, 0], shrink=0.9, pad=0.01)

    im2 = axes[r, 1].imshow(tt, origin="lower", extent=extent, aspect="equal",
                            cmap="magma", vmin=0, vmax=tt_max)
    axes[r, 1].contour(gx / 1000, gy / 1000, tt, levels=levels,
                       colors="white", linewidths=0.45)
    axes[r, 1].plot(src_x / 1000, src_y / 1000, "*", color="cyan",
                    markersize=13, markeredgecolor="k", markeredgewidth=0.6)
    axes[r, 1].set_title(f"{name}  —  traveltime (s), isochrone 0.25 s", fontsize=11)
    plt.colorbar(im2, ax=axes[r, 1], shrink=0.9, pad=0.01)

    for c in (0, 1):
        axes[r, c].set_ylabel("Z (km)")
for c in (0, 1):
    axes[-1, c].set_xlabel("X (km)")

fig.suptitle("Three Vp models on the identical DEM deformation, and their first-arrival traveltime fields "
             "(true 1:1 aspect, shot at mid-line surface)", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_tt_maps.png", dpi=150)
print("saved fig_tt_maps.png")

# ---------------------------------------------------------------
# Figure B: two-sided surface t-x curve + model spread
# ---------------------------------------------------------------
surface_y = np.full(len(gx), np.nan)
tt_surf = {n: np.full(len(gx), np.nan) for n in names}
for k in range(len(gx)):
    col = np.where(inside[:, k])[0]
    if len(col) == 0:
        continue
    top = col.max()
    surface_y[k] = gy[top]
    for r, n in enumerate(names):
        tt_surf[n][k] = tts[r][top, k]

offset = (gx - src_x) / 1000.0

fig2, axes2 = plt.subplots(3, 1, figsize=(9.5, 8), sharex=True,
                           height_ratios=[0.7, 1.6, 1.0])

axes2[0].plot(offset, surface_y / 1000, color="#555", lw=1.4)
axes2[0].plot(0, src_y / 1000, "*", color="#c0562d", markersize=14, markeredgecolor="k")
axes2[0].set_ylabel("surface Z (km)")
axes2[0].set_title("Free surface and shot position")

for n in names:
    axes2[1].plot(offset, tt_surf[n], color=COLORS[n], lw=2, label=n)
axes2[1].set_ylabel("first-arrival time (s)")
axes2[1].set_title("Two-sided surface t–x curve")
axes2[1].legend(fontsize=9)
axes2[1].invert_yaxis()

ref = tt_surf["IG-FEM + Botter"]
for n in names:
    if n == "IG-FEM + Botter":
        continue
    axes2[2].plot(offset, tt_surf[n] - ref, color=COLORS[n], lw=2,
                  label=f"{n} − IG-FEM + Botter")
axes2[2].axhline(0, color="#3f8f6b", lw=1.5, ls="--")
axes2[2].set_xlabel("offset from shot (km)")
axes2[2].set_ylabel("Δt (s)")
axes2[2].set_title("Traveltime residual relative to IG-FEM + Botter")
axes2[2].legend(fontsize=9)

fig2.suptitle("Same DEM, three rock-physics / strain routes → observable traveltime spread", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig("fig_tt_surface_curve.png", dpi=150)
print("saved fig_tt_surface_curve.png")

for n in names:
    if n == "IG-FEM + Botter":
        continue
    r = tt_surf[n] - ref
    print(f"{n:>16s} vs IG-FEM+Botter:  mean Δt {np.nanmean(r):+.3f} s, "
          f"max |Δt| {np.nanmax(np.abs(r)):.3f} s")

print(f"SSPX vs IG-FEM strain corr: {np.corrcoef(d['vol_sspx'], d['vol_igfem'])[0,1]:+.3f}")

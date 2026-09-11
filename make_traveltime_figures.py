import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load("traveltime_compare.npz")
gx, gy = d["gx"], d["gy"]
Vp_b, Vp_hm = d["Vp_grid_botter"], d["Vp_grid_hm"]
tt_b, tt_hm = d["tt_botter"], d["tt_hm"]
inside = d["inside"]
src_x, src_y = d["src_x"], d["src_y"]

tt_b_m = np.ma.array(tt_b, mask=~inside)
tt_hm_m = np.ma.array(tt_hm, mask=~inside)
diff = np.ma.array(tt_hm - tt_b, mask=~inside)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})

extent = [gx.min(), gx.max(), gy.min(), gy.max()]

fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))

im0 = axes[0,0].imshow(Vp_b, origin="lower", extent=extent, aspect="auto", cmap="viridis",
                        vmin=3000, vmax=5000)
axes[0,0].set_title("Botter et al. (2014) $V_P$ field (m/s)")
plt.colorbar(im0, ax=axes[0,0])

im1 = axes[0,1].imshow(np.ma.array(Vp_hm, mask=~inside), origin="lower", extent=extent,
                        aspect="auto", cmap="viridis", vmin=1000, vmax=4500)
axes[0,1].set_title("Hertz-Mindlin $V_P$ field (m/s)")
plt.colorbar(im1, ax=axes[0,1])

cont_levels = np.arange(0, 10, 0.5)
im2 = axes[1,0].imshow(tt_b_m, origin="lower", extent=extent, aspect="auto", cmap="magma")
axes[1,0].contour(gx, gy, tt_b_m, levels=cont_levels, colors="white", linewidths=0.5)
axes[1,0].plot(src_x, src_y, "*", color="cyan", markersize=14, markeredgecolor="k")
axes[1,0].set_title("Botter-based traveltime (s)")
plt.colorbar(im2, ax=axes[1,0])

im3 = axes[1,1].imshow(tt_hm_m, origin="lower", extent=extent, aspect="auto", cmap="magma")
axes[1,1].contour(gx, gy, tt_hm_m, levels=cont_levels, colors="white", linewidths=0.5)
axes[1,1].plot(src_x, src_y, "*", color="cyan", markersize=14, markeredgecolor="k")
axes[1,1].set_title("Hertz-Mindlin-based traveltime (s)")
plt.colorbar(im3, ax=axes[1,1])

for ax in axes.flat:
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
fig.suptitle("Eikonal first-arrival P traveltime from two Vp models (same DEM particles)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig("fig_tt_maps.png", dpi=150)
print("saved fig_tt_maps.png")

# ---- surface (topography-following) t-x curve ----
surface_row = np.full(len(gx), np.nan)
tt_surf_b = np.full(len(gx), np.nan)
tt_surf_hm = np.full(len(gx), np.nan)
for k in range(len(gx)):
    col_inside = np.where(inside[:, k])[0]
    if len(col_inside) == 0:
        continue
    top = col_inside.max()
    surface_row[k] = gy[top]
    tt_surf_b[k] = tt_b_m[top, k]
    tt_surf_hm[k] = tt_hm_m[top, k]

offset = gx - src_x
fig2, axes2 = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[1, 1.4])
axes2[0].plot(gx, surface_row, color="#555", lw=1.5)
axes2[0].set_ylabel("surface Y (m)")
axes2[0].set_title("Sample surface (top of particle pile)")

axes2[1].plot(offset, tt_surf_b, color="#2f6fb3", lw=2, label="Botter et al. (2014) empirical $V_P$")
axes2[1].plot(offset, tt_surf_hm, color="#c0562d", lw=2, label="Hertz-Mindlin $V_P$")
axes2[1].set_xlabel("offset from source (m)")
axes2[1].set_ylabel("first-arrival traveltime (s)")
axes2[1].set_title("Surface t-x curve (refraction-survey style)")
axes2[1].legend()
fig2.suptitle("Same DEM strain field, two rock-physics models -> different observable traveltimes", fontsize=12)
fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig("fig_tt_surface_curve.png", dpi=150)
print("saved fig_tt_surface_curve.png")

print(f"max |diff| = {np.nanmax(np.abs(tt_surf_hm - tt_surf_b)):.4f} s "
      f"at max offset; mean surface diff = {np.nanmean(tt_surf_hm - tt_surf_b):.4f} s")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import sys
sys.path.insert(0, ".")
from main import run

pos = np.loadtxt("/home/user/soo-jung-an/ig-fem/data/cood0_66494.txt")
rad = np.loadtxt("/home/user/soo-jung-an/ig-fem/data/radius.txt")
out = run(pos, rad)
v = out["valid"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "font.size": 11,
    "axes.edgecolor": "#555", "axes.labelcolor": "#222",
    "xtick.color": "#444", "ytick.color": "#444", "axes.titleweight": "bold",
})


def circle_plot(ax, x, y, r, c, mask, cmap, vmin, vmax, title):
    patches = [Circle((xi, yi), ri) for xi, yi, ri in zip(x[mask], y[mask], r[mask])]
    pc = PatchCollection(patches, cmap=cmap)
    pc.set_array(c[mask])
    pc.set_clim(vmin, vmax)
    pc.set_edgecolor("none")
    ax.add_collection(pc)
    # boundary (excluded) particles shown as thin outlines for context
    patches_b = [Circle((xi, yi), ri) for xi, yi, ri in zip(x[~mask], y[~mask], r[~mask])]
    pcb = PatchCollection(patches_b, facecolor="none", edgecolor="#ccc", linewidth=0.3)
    ax.add_collection(pcb)
    ax.set_xlim(x.min() - 200, x.max() + 200)
    ax.set_ylim(y.min() - 200, y.max() + 200)
    ax.set_aspect(1)
    ax.set_title(title)
    return pc

x, y = pos[:, 0], pos[:, 1]
fig, axes = plt.subplots(2, 2, figsize=(13, 6.6))
circle_plot(axes[0,0], x, y, rad, out["coordination_number"].astype(float), v, "viridis",
            2, 8, "Coordination number (contacts/particle)")
circle_plot(axes[0,1], x, y, rad, out["porosity"], v, "viridis",
            0.02, 0.45, "Local porosity (measurement circle)")
circle_plot(axes[1,0], x, y, rad, out["pressure"]/1e9, v, "inferno",
            np.nanpercentile(out["pressure"][v]/1e9, 2), np.nanpercentile(out["pressure"][v]/1e9, 98),
            "Local confining pressure (GPa, virial stress)")
circle_plot(axes[1,1], x, y, rad, out["Vp"], v, "viridis",
            np.nanpercentile(out["Vp"][v], 2), np.nanpercentile(out["Vp"][v], 98),
            "Hertz-Mindlin $V_P$ (m/s)")
for ax in axes.flat:
    fig.colorbar(ax.collections[0], ax=ax, shrink=0.85)
fig.suptitle("Hertz-Mindlin contact-based fields, 2D DEM (17,694 particles, large-strain stage)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("fig_hm_maps.png", dpi=150)
print("saved fig_hm_maps.png")

# Vp/Vs constancy check plot
fig2, ax = plt.subplots(figsize=(6, 4.2))
ax.scatter(out["pressure"][v]/1e9, out["VpVs"][v], s=4, alpha=0.3, color="#2f6fb3")
ax.set_xlabel("local pressure (GPa)"); ax.set_ylabel("$V_P/V_S$")
ax.set_title(f"$V_P/V_S$ is pressure-independent under pure HM\n(std = {out['VpVs'][v].std():.2e}, mean = {out['VpVs'][v].mean():.4f})")
fig2.tight_layout()
fig2.savefig("fig_vpvs_constant.png", dpi=150)
print("saved fig_vpvs_constant.png")

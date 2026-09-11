"""
Eikonal first-arrival P-wave traveltime from THREE independently-derived
Vp fields on the SAME 2D DEM particle data (data/cood0_66494.txt, the
large-strain stage, referenced to data/cood0_5932.txt):

  1. SSPX-style + Botter: nearest-neighbor local deformation gradient
     (the strain method Botter et al. themselves used, via Cardozo &
     Allmendinger's SSPX) -> rock_physics.py Eqs. 1-4.
  2. IG-FEM + Botter: the global mass-matrix FEM solve
     (igfem_assembly.py / igfem_preprocessing.py, from Soo-jung-AN/IG-FEM)
     -> the same rock_physics.py Eqs. 1-4.
  3. Hertz-Mindlin: contact-mechanics first principles (main.py here),
     no strain and no empirical curve involved at all.

Models 1 and 2 share an identical depth-dependent compaction trend for
the *initial* (pre-strain) properties -- Vp_ini from VP_TOP at the free
surface to VP_BOT at the base, phi_ini likewise -- assigned in the
UNDEFORMED configuration, the way Botter et al. assign properties to the
material before faulting. Without that trend a single homogeneous
Vp_ini produces an almost gradient-free velocity field, which in turn
produces no diving waves and a featureless straight-line t-x curve.

All Vp fields are gridded onto a common regular grid and the eikonal
equation is solved with the fast marching method (scikit-fmm) from a
surface source at mid-line, giving (a) 2D traveltime maps and (b) a
two-sided surface t-x curve as in a refraction survey.
"""
import numpy as np
import skfmm
from scipy.interpolate import griddata
from scipy.spatial import Delaunay, cKDTree
from scipy import sparse
from scipy.sparse.linalg import spsolve

from igfem_preprocessing import reshape, Get_shf_coef, Get_gp_cood
from igfem_assembly import M_assembly, A_assembly, R_assembly
from rock_physics import synthesize_vpvs
from main import run as run_hm

D = "data"
UNDEFORMED_STAGE = "5932"
STAGE = "66494"

# depth-dependent compaction trend for the pre-strain properties
VP_TOP, VP_BOT = 1.8, 4.0        # km/s at free surface / base
PHI_TOP, PHI_BOT = 0.35, 0.15    # porosity at free surface / base
RHO_GRAIN = 2650.0               # kg/m3


def initial_properties(undeformed_cood):
    """Pre-strain (initial) properties with a linear compaction trend in
    the UNDEFORMED configuration: loose/slow at the free surface,
    compacted/fast at the base."""
    y = undeformed_cood[:, 1]
    t = (y - y.min()) / (y.max() - y.min())   # 0 at base, 1 at surface
    Vp_ini = VP_BOT + (VP_TOP - VP_BOT) * t
    phi_ini = PHI_BOT + (PHI_TOP - PHI_BOT) * t
    rho_g = np.full(len(y), RHO_GRAIN)
    return phi_ini, rho_g, Vp_ini


def sspx_vol_strain(X0, X1, k=16):
    """Nearest-neighbor local deformation gradient (the SSPX approach of
    Cardozo & Allmendinger, 2009, used by Botter et al.): least-squares
    fit of F to dx = F dX over each particle's k nearest neighbors."""
    tree = cKDTree(X0)
    _, idx = tree.query(X0, k=k + 1)
    idx = idx[:, 1:]                       # drop self
    dX = X0[idx] - X0[:, None, :]
    dx = X1[idx] - X1[:, None, :]
    A = np.einsum("nkj,nkl->njl", dX, dX)
    B = np.einsum("nkj,nkl->njl", dx, dX)
    F = np.matmul(B, np.linalg.inv(A))
    return np.linalg.det(F) - 1.0


def igfem_vol_strain(undeformed_cood, deformed_cood):
    """The repo author's IG-FEM: global mass-matrix L2 projection of the
    element-wise deformation gradient onto the particles."""
    p_num = len(undeformed_cood)
    disp = deformed_cood - undeformed_cood

    tri = Delaunay(undeformed_cood)
    ele_id = reshape(undeformed_cood, tri.simplices.astype(np.int32), 115)
    TT_E = len(ele_id)

    SC_mat_e = np.zeros((TT_E, 3, 3)); Get_shf_coef(SC_mat_e, ele_id, undeformed_cood)
    PQ_detJ_e = np.zeros((TT_E, 3, 3)); Get_gp_cood(PQ_detJ_e, ele_id, undeformed_cood)

    M_RC = np.zeros((2, 36 * TT_E), dtype=np.int64); M_data = np.zeros(36 * TT_E)
    M_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, M_RC, M_data, p_num)
    M_CSR = sparse.csr_matrix((M_data, (M_RC[0], M_RC[1])), shape=(p_num * 4, p_num * 4)).tocsc()

    A_RC = np.zeros((2, 36 * TT_E), dtype=np.int64); A_data = np.zeros(36 * TT_E)
    A_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, A_RC, A_data, p_num)
    A_CSR = sparse.csr_matrix((A_data, (A_RC[0], A_RC[1])), shape=(p_num * 4, p_num * 4))

    R_vec = np.zeros(p_num * 4)
    R_assembly(SC_mat_e, ele_id, undeformed_cood, PQ_detJ_e, R_vec, p_num)

    U = np.hstack((disp[:, 0], disp[:, 1])); U = np.hstack((U, U))
    F = spsolve(M_CSR, A_CSR * U + R_vec)
    F11, F22, F12, F21 = F[:p_num], F[p_num:2*p_num], F[2*p_num:3*p_num], F[3*p_num:]
    return F11 * F22 - F21 * F12 - 1.0


def botter_vp(vol, undeformed_cood):
    """Botter et al. (2014) Eqs. 1-4 on a volumetric strain field, with the
    depth-trended initial properties. Returns Vp in m/s."""
    phi_ini, rho_g, Vp_ini = initial_properties(undeformed_cood)
    _, _, Vp_kms, _, _ = synthesize_vpvs(vol, phi_ini, rho_g, Vp_ini)
    return Vp_kms * 1000.0


def grid_field(pos, values, GX, GY, mask=None):
    if mask is not None:
        pos, values = pos[mask], values[mask]
    g = griddata(pos, values, (GX, GY), method="linear")
    g_nn = griddata(pos, values, (GX, GY), method="nearest")
    g[np.isnan(g)] = g_nn[np.isnan(g)]
    return g


def main():
    undeformed_cood = np.loadtxt(f"{D}/cood0_{UNDEFORMED_STAGE}.txt")
    deformed_cood = np.loadtxt(f"{D}/cood0_{STAGE}.txt")
    rad = np.loadtxt(f"{D}/radius.txt")

    vol_sspx = sspx_vol_strain(undeformed_cood, deformed_cood)
    vol_igfem = igfem_vol_strain(undeformed_cood, deformed_cood)
    print(f"vol strain  SSPX: mean {vol_sspx.mean():+.4f}  "
          f"IG-FEM: mean {vol_igfem.mean():+.4f}  "
          f"corr {np.corrcoef(vol_sspx, vol_igfem)[0,1]:+.3f}")

    models = {
        "SSPX + Botter": botter_vp(vol_sspx, undeformed_cood),
        "IG-FEM + Botter": botter_vp(vol_igfem, undeformed_cood),
    }
    out_hm = run_hm(deformed_cood, rad)
    models["Hertz-Mindlin"] = out_hm["Vp"]

    for name, vp in models.items():
        print(f"{name:>16s} Vp (m/s): {np.nanmin(vp):7.0f} / {np.nanmean(vp):7.0f} / {np.nanmax(vp):7.0f}")

    # common grid, on the deformed (observed) geometry
    xmin, xmax = deformed_cood[:, 0].min(), deformed_cood[:, 0].max()
    ymin, ymax = deformed_cood[:, 1].min(), deformed_cood[:, 1].max()
    dx = 60.0
    nx, ny = int((xmax - xmin) / dx) + 1, int((ymax - ymin) / dx) + 1
    gx, gy = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    print(f"grid: {nx} x {ny}, dx = {dx} m")

    tree = cKDTree(deformed_cood)
    dist, _ = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    inside = (dist.reshape(GX.shape) < 3 * dx)

    # shot at mid-line on the free surface, so the spread is two-sided
    src_x = 0.5 * (xmin + xmax)
    src_j = np.argmin((gx - src_x) ** 2)
    src_i = np.where(inside[:, src_j])[0].max()
    src_x, src_y = gx[src_j], gy[src_i]
    print(f"source at x={src_x:.0f} m, y={src_y:.0f} m (free surface, mid-line)")

    phi = np.ones_like(GX)
    phi[src_i, src_j] = -1

    vp_grids, tts = {}, {}
    for name, vp in models.items():
        mask = out_hm["valid"] if name == "Hertz-Mindlin" else None
        pos = deformed_cood
        vp_grids[name] = grid_field(pos, vp, GX, GY, mask=mask)
        tts[name] = skfmm.travel_time(phi, vp_grids[name], dx=dx)
        print(f"{name:>16s} max traveltime {np.ma.array(tts[name], mask=~inside).max():.3f} s")

    np.savez("traveltime_compare.npz",
             gx=gx, gy=gy, inside=inside, src_x=src_x, src_y=src_y,
             names=np.array(list(models.keys())),
             vp_grids=np.array([vp_grids[n] for n in models]),
             tts=np.array([tts[n] for n in models]),
             vol_sspx=vol_sspx, vol_igfem=vol_igfem)
    print("saved traveltime_compare.npz")


if __name__ == "__main__":
    main()

"""
Eikonal first-arrival P-wave traveltime from two independently-derived
Vp fields on the SAME 2D DEM particle data (data/cood0_66494.txt, the
large-strain stage):

  - Botter et al. (2014) empirical Vp: run the 2D IG-FEM solve
    (igfem_assembly.py / igfem_preprocessing.py, from Soo-jung-AN/IG-FEM)
    to get the finite volumetric (area) strain, then rock_physics.py
    (Eqs. 1-4) to get Vp.
  - Hertz-Mindlin contact-based Vp: main.py in this repo.

Both Vp fields are gridded onto a common regular grid and the eikonal
equation is solved with the fast marching method (scikit-fmm) from a
single surface source, to compare (a) full 2D traveltime maps, (b) a
surface t-x (offset) curve as in a refraction survey.
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
STAGE = "66494"


def botter_vp_field(undeformed_stage="5932", deformed_stage=STAGE):
    """Run the real 2D IG-FEM solve and Botter et al. (2014) rock physics
    on a single homogeneous material (quartz sandstone, Table 3)."""
    undeformed_cood = np.loadtxt(f"{D}/cood0_{undeformed_stage}.txt")
    deformed_cood = np.loadtxt(f"{D}/cood0_{deformed_stage}.txt")
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
    vol = F11 * F22 - F21 * F12 - 1.0

    phi_ini = np.full(p_num, 0.15); rho_g = np.full(p_num, 2650.0); Vp_ini = np.full(p_num, 4.0)
    _, _, Vp_kms, _, _ = synthesize_vpvs(vol, phi_ini, rho_g, Vp_ini)
    return deformed_cood, Vp_kms * 1000.0  # km/s -> m/s


def grid_field(pos, values, GX, GY, mask=None):
    if mask is not None:
        pos, values = pos[mask], values[mask]
    g = griddata(pos, values, (GX, GY), method="linear")
    g_nn = griddata(pos, values, (GX, GY), method="nearest")
    g[np.isnan(g)] = g_nn[np.isnan(g)]
    return g


def main():
    pos_botter, Vp_botter = botter_vp_field()
    print("Botter Vp (m/s): min/mean/max", Vp_botter.min(), Vp_botter.mean(), Vp_botter.max())

    pos_hm = np.loadtxt(f"{D}/cood0_{STAGE}.txt")
    rad_hm = np.loadtxt(f"{D}/radius.txt")
    out_hm = run_hm(pos_hm, rad_hm)
    print("HM     Vp (m/s): min/mean/max", np.nanmin(out_hm["Vp"]), np.nanmean(out_hm["Vp"]), np.nanmax(out_hm["Vp"]))

    xmin, xmax = pos_botter[:, 0].min(), pos_botter[:, 0].max()
    ymin, ymax = pos_botter[:, 1].min(), pos_botter[:, 1].max()
    dx = 60.0
    nx, ny = int((xmax - xmin) / dx) + 1, int((ymax - ymin) / dx) + 1
    gx, gy = np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny)
    GX, GY = np.meshgrid(gx, gy)
    print(f"grid: {nx} x {ny}")

    Vp_grid_botter = grid_field(pos_botter, Vp_botter, GX, GY)
    Vp_grid_hm = grid_field(pos_hm, out_hm["Vp"], GX, GY, mask=out_hm["valid"])

    tree = cKDTree(pos_botter)
    dist, _ = tree.query(np.column_stack([GX.ravel(), GY.ravel()]))
    inside = (dist.reshape(GX.shape) < 3 * dx)

    src_x, src_y = xmin + 200, ymax - 200
    phi = np.ones_like(GX)
    phi[np.argmin((gy - src_y) ** 2), np.argmin((gx - src_x) ** 2)] = -1

    tt_botter = skfmm.travel_time(phi, Vp_grid_botter, dx=dx)
    tt_hm = skfmm.travel_time(phi, Vp_grid_hm, dx=dx)

    np.savez("traveltime_compare.npz",
             gx=gx, gy=gy, Vp_grid_botter=Vp_grid_botter, Vp_grid_hm=Vp_grid_hm,
             tt_botter=tt_botter, tt_hm=tt_hm, inside=inside, src_x=src_x, src_y=src_y)
    print("saved traveltime_compare.npz")
    print(f"max traveltime: Botter {np.ma.array(tt_botter, mask=~inside).max():.4f}s   "
          f"HM {np.ma.array(tt_hm, mask=~inside).max():.4f}s")


if __name__ == "__main__":
    main()

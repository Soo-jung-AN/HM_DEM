"""
Hertz-Mindlin contact-based effective-medium Vp/Vs from real DEM
particle positions (2D IG-FEM sample data: Soo-jung-AN/IG-FEM).

Pipeline:
  1. Detect contacts geometrically from positions + radii.
  2. Per-contact Hertzian normal force from the overlap.
  3. Per-particle coordination number, local porosity (Voronoi),
     local confining pressure (micromechanical/virial stress).
  4. Hertz-Mindlin K, G from those local fields + assumed grain
     properties (mineral shear modulus, Poisson's ratio).
  5. Vp, Vs from K, G, density -- no empirical curve-fitting, unlike
     the Botter et al. (2014) approach used elsewhere in this project.
"""
import numpy as np
from local_fields import compute_local_fields
from effective_medium import hertz_mindlin_KG, vp_vs_from_KG

# Quartz (typical sand/sandstone grain mineral)
E_GRAIN = 94.5e9       # Pa
NU_GRAIN = 0.17
RHO_GRAIN = 2650.0     # kg/m3


def run(pos, rad, E_grain=E_GRAIN, nu_grain=NU_GRAIN, rho_grain=RHO_GRAIN):
    fields = compute_local_fields(pos, rad, E_grain, nu_grain)
    C, phi, P = fields["coordination_number"], fields["porosity"], fields["pressure"]
    # The measurement-circle porosity estimator (see local_fields.py) has a
    # known boundary bias for polydisperse packings: a particle whose center
    # is just inside the circle but whose own area extends outside it is
    # still counted in full, which can push the apparent local solid
    # fraction above 100% (phi <= 0) in dense/heterogeneous regions. Rather
    # than discard those particles, clip to a physically plausible 2D
    # random-packing porosity range.
    phi = np.clip(phi, 0.02, 0.60)
    fields["porosity"] = phi

    valid = ~fields["near_boundary"] & (C > 0) & np.isfinite(phi) & (P > 0)

    K = np.full(len(pos), np.nan); G = np.full(len(pos), np.nan)
    K[valid], G[valid] = hertz_mindlin_KG(C[valid], phi[valid], E_grain / (2 * (1 + nu_grain)), nu_grain, P[valid])

    rho = rho_grain * (1 - phi)  # dry pack
    Vp = np.full(len(pos), np.nan); Vs = np.full(len(pos), np.nan)
    Vp[valid], Vs[valid] = vp_vs_from_KG(K[valid], G[valid], rho[valid])

    fields.update(K=K, G=G, rho=rho, Vp=Vp, Vs=Vs, VpVs=Vp / Vs, valid=valid)
    return fields


if __name__ == "__main__":
    import sys
    D = sys.argv[1] if len(sys.argv) > 1 else "data"
    stage = sys.argv[2] if len(sys.argv) > 2 else "66494"
    pos = np.loadtxt(f"{D}/cood0_{stage}.txt")
    rad = np.loadtxt(f"{D}/radius.txt")
    out = run(pos, rad)
    v = out["valid"]
    print(f"n_particles={len(pos)}  valid(interior)={v.sum()}")
    print(f"coordination number: mean {out['coordination_number'][v].mean():.2f}")
    print(f"porosity: mean {out['porosity'][v].mean():.4f}")
    print(f"pressure: mean {out['pressure'][v].mean():.4g} Pa")
    print(f"K: mean {out['K'][v].mean():.4g} Pa   G: mean {out['G'][v].mean():.4g} Pa")
    print(f"Vp: mean {out['Vp'][v].mean():.2f} m/s   Vs: mean {out['Vs'][v].mean():.2f} m/s")
    print(f"Vp/Vs: mean {out['VpVs'][v].mean():.3f}")

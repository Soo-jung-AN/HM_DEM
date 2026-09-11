"""
Rock-physics relations linking DEM finite volumetric strain to synthetic
seismic velocities, following the empirical workflow of:

Botter, C., Cardozo, N., Hardy, S., Lecomte, I., Escalona, A. (2014).
From mechanical modeling to seismic imaging of faults: A synthetic
workflow to study the impact of faults on seismic reflection data.
Marine and Petroleum Geology, 57, 187-207. Equations (1)-(4), Section 2.2.
https://doi.org/10.1016/j.marpetgeo.2014.05.013

All velocities are in km/s (Han's relation, Eq. 4, is calibrated in km/s).
"""
import numpy as np

RHO_WATER = 1000.0  # kg/m3, saturated pore fluid density (Botter et al., 2014, Table 3)


def porosity_from_strain(vol_strain, phi_ini):
    """Eq. (1): linear porosity change with volumetric strain (dilatation)."""
    ev = np.clip(vol_strain, -1.0, 1.0)
    return phi_ini * (0.25 * ev + 1.0)


def density_from_porosity(phi, rho_grain, rho_fluid=RHO_WATER):
    """Eq. (2): saturated bulk density from porosity (linear mixing)."""
    return rho_grain * (1.0 - phi) + rho_fluid * phi


def vp_from_strain(vol_strain, Vp_ini):
    """Eq. (3): piecewise-quadratic (sigmoidal) Vp vs. volumetric strain,
    bounded to +/-25% change from Vp_ini at |vol_strain| = 1."""
    ev = np.clip(vol_strain, -1.0, 1.0)
    compaction = ev < 0.0
    factor = np.where(
        compaction,
        -0.25 * ev**2 - 0.5 * ev + 1.0,
        0.25 * ev**2 - 0.5 * ev + 1.0,
    )
    return Vp_ini * factor


def vs_from_vp(Vp):
    """Eq. (4): Han's (1986) empirical Vp-Vs relation (km/s), as used in
    Botter et al. (2014)."""
    return 0.794 * Vp - 0.787


def synthesize_vpvs(vol_strain, phi_ini, rho_grain, Vp_ini, rho_fluid=RHO_WATER):
    """Apply Eqs. (1)-(4) of Botter et al. (2014) to DEM finite volumetric
    strain (vol_strain = det(F) - 1), returning (phi, rho, Vp, Vs, Vp/Vs)."""
    phi = porosity_from_strain(vol_strain, phi_ini)
    rho = density_from_porosity(phi, rho_grain, rho_fluid)
    Vp = vp_from_strain(vol_strain, Vp_ini)
    Vs = vs_from_vp(Vp)
    return phi, rho, Vp, Vs, Vp / Vs

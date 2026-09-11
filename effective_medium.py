"""
Hertz-Mindlin effective-medium moduli for a random dense pack of
identical spheres (Digby, 1981; Walton, 1987; Mindlin, 1949), as
tabulated in Mavko, Mukerji & Dvorkin, "The Rock Physics Handbook":

    K_HM = [ C^2 (1-phi)^2 G_grain^2 / (18 pi^2 (1-nu_grain)^2) * P ] ^ (1/3)

    G_HM = (5 - 4*nu_grain) / (5*(2 - nu_grain))
           * [ 3 C^2 (1-phi)^2 G_grain^2 / (2 pi^2 (1-nu_grain)^2) * P ] ^ (1/3)

where C is coordination number, phi is porosity, G_grain/nu_grain are
the mineral (grain) shear modulus and Poisson's ratio, and P is the
effective (confining) pressure. These are closed-form results for an
isotropic random *3D sphere* pack -- applying them to a 2D disk
packing is the standard simplification used across granular rock-physics
work (2D DEM force laws are themselves usually calibrated as Hertzian),
not an exact solution of 2D (cylinder) contact mechanics. See README.
"""
import numpy as np


def hertz_mindlin_KG(coordination_number, porosity, G_grain, nu_grain, pressure):
    C, phi, P = coordination_number, porosity, np.clip(pressure, 0, None)
    common = (C**2 * (1 - phi)**2 * G_grain**2) / (np.pi**2 * (1 - nu_grain)**2)
    K_HM = (common * P / 18.0) ** (1.0 / 3.0)
    G_HM = ((5 - 4 * nu_grain) / (5 * (2 - nu_grain))) * (3 * common * P / 2.0) ** (1.0 / 3.0)
    return K_HM, G_HM


def vp_vs_from_KG(K, G, rho):
    Vp = np.sqrt((K + 4.0 * G / 3.0) / rho)
    Vs = np.sqrt(G / rho)
    return Vp, Vs

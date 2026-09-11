"""
Per-particle local fields needed by the Hertz-Mindlin effective-medium
formulas, all derived purely from particle positions + radii (+ the
Hertzian contact forces reconstructed in contact_mechanics.py):

- coordination number: contact count per particle
- local porosity: coarse-graining "measurement circle" method (Goldhirsch,
  2010-style local averaging, standard in granular micromechanics):
  1 - (area of all particles whose center falls within a measurement
  circle of radius R_m around i) / (measurement circle area). This is
  used instead of a plain (unweighted) Voronoi tessellation because, for
  a POLYDISPERSE packing (particles of different radii), an ordinary
  Euclidean Voronoi cell can be smaller than the particle's own circle
  (a large particle boxed in by tightly-packed small neighbors) --
  giving a negative "porosity". The measurement-circle estimator stays
  bounded in [0, 1] by construction whenever R_m is a few times the mean
  particle radius, which is what's used here (R_m = 4 * mean radius).
- local confining pressure: micromechanical (virial) stress tensor over
  the same measurement circle, Christoffersen, Mehrabadi & Nemat-Nasser
  (1981): for each particle i,
      sigma_i = (1 / (2 A_i)) * sum_{contacts c within the circle} F_n^c (x) l^c
  where l^c is the full center-to-center branch vector and A_i is the
  measurement circle's area; pressure P_i = 0.5*trace(sigma_i).
"""
import numpy as np
from scipy.spatial import cKDTree
from contact_mechanics import find_contacts, effective_radius, contact_normal_force


def measurement_circle_porosity(pos, rad, R_m_factor=8.0):
    """Local porosity via a fixed measurement circle per particle (area-
    weighted coarse-graining), robust to size polydispersity unlike a
    plain Voronoi tessellation. Also flags particles whose measurement
    circle extends outside the sample's bounding box (edge effects)."""
    R_m = R_m_factor * rad.mean()
    tree = cKDTree(pos)
    neighbor_lists = tree.query_ball_point(pos, r=R_m)
    circle_area = np.pi * R_m**2
    particle_area = np.pi * rad**2

    n = len(pos)
    porosity = np.empty(n)
    for i, neighbors in enumerate(neighbor_lists):
        porosity[i] = 1.0 - particle_area[neighbors].sum() / circle_area

    bmin, bmax = pos.min(axis=0), pos.max(axis=0)
    near_boundary = (
        (pos[:, 0] - R_m < bmin[0]) | (pos[:, 0] + R_m > bmax[0]) |
        (pos[:, 1] - R_m < bmin[1]) | (pos[:, 1] + R_m > bmax[1])
    )
    return porosity, near_boundary, circle_area


def compute_local_fields(pos, rad, E_grain, nu_grain):
    """Returns dict with coordination_number, porosity, pressure, contacts
    metadata, all shape (n_particles,) except the contact arrays."""
    n = len(pos)
    pairs, delta, normal = find_contacts(pos, rad)
    R_eff = effective_radius(rad[pairs[:, 0]], rad[pairs[:, 1]])
    Fn = contact_normal_force(delta, R_eff, E_grain, nu_grain)

    coordination_number = np.zeros(n, dtype=np.int64)
    np.add.at(coordination_number, pairs[:, 0], 1)
    np.add.at(coordination_number, pairs[:, 1], 1)

    porosity, near_boundary, circle_area = measurement_circle_porosity(pos, rad)
    areas = np.full(n, circle_area)  # same measurement-circle area for every particle

    # micromechanical (virial) stress tensor per particle
    sigma = np.zeros((n, 2, 2))
    branch = pos[pairs[:, 1]] - pos[pairs[:, 0]]  # full center-to-center vector
    F_vec = Fn[:, None] * normal  # normal-only contact force vector
    contrib = F_vec[:, :, None] * branch[:, None, :]  # outer product F (x) l, shape (n_contacts,2,2)
    np.add.at(sigma, pairs[:, 0], contrib)
    np.add.at(sigma, pairs[:, 1], contrib)  # branch vector direction is symmetric under this convention
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma = sigma / (2.0 * areas[:, None, None])
    pressure = 0.5 * (sigma[:, 0, 0] + sigma[:, 1, 1])

    return dict(coordination_number=coordination_number, porosity=porosity,
                pressure=pressure, near_boundary=near_boundary,
                contacts=pairs, overlap=delta, normal=normal, Fn=Fn, voronoi_area=areas)

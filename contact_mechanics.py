"""
Per-contact Hertz-Mindlin contact stiffness reconstructed from DEM
particle positions and radii alone (no raw contact-pair file needed).

Contacts are detected geometrically: two particles are in contact when
the distance between centers is <= the sum of their radii. The overlap
delta then gives the Hertzian normal force via the standard elastic
contact solution for two spheres (used here as the conventional
approximation for DEM disks too -- see README for the caveat).

References: Hertz (1882) normal contact; Mindlin (1949) tangential
contact (no-slip limit); as tabulated in Mavko, Mukerji & Dvorkin,
"The Rock Physics Handbook", section on Hertz-Mindlin contact models.
"""
import numpy as np
from scipy.spatial import cKDTree


def find_contacts(pos, rad, tol=1e-6):
    """Geometrically detect contacts: pairs (i, j) with i<j whose center
    distance <= r_i + r_j (+ tol). Returns (pairs, overlap, unit_normal)."""
    n = len(pos)
    max_r = rad.max()
    tree = cKDTree(pos)
    pairs_seen = set()
    idx_pairs, deltas, normals = [], [], []
    # query each particle for neighbors within its own contact range,
    # then filter by the pair-specific radius sum (KDTree only supports
    # a single query radius, so we over-query with 2*max_r and filter).
    neighbor_lists = tree.query_ball_point(pos, r=2 * max_r)
    for i, neighbors in enumerate(neighbor_lists):
        for j in neighbors:
            if j <= i:
                continue
            d_vec = pos[j] - pos[i]
            dist = np.linalg.norm(d_vec)
            contact_gap = rad[i] + rad[j] - dist
            if contact_gap > tol:
                idx_pairs.append((i, j))
                deltas.append(contact_gap)
                normals.append(d_vec / dist)
    return (np.array(idx_pairs, dtype=np.int64),
            np.array(deltas, dtype=np.float64),
            np.array(normals, dtype=np.float64))


def hertz_mindlin_contact_stiffness(delta, R_eff, E_grain, nu_grain):
    """Per-contact normal (kn) and tangential (ks, no-slip limit) stiffness.

    E_star = E_grain / (2*(1-nu_grain**2))         (identical-grain effective modulus)
    G_star = G_grain / (2*(2-nu_grain))             (identical-grain effective shear modulus)
    kn = 2 * E_star * sqrt(R_eff * delta)           (d F_n/d delta, Hertz)
    ks = 8 * G_star * sqrt(R_eff * delta)           (Mindlin, no-slip)
    """
    G_grain = E_grain / (2 * (1 + nu_grain))
    E_star = E_grain / (2 * (1 - nu_grain**2))
    G_star = G_grain / (2 - nu_grain) / 2  # = G_grain / (2*(2-nu_grain))
    kn = 2 * E_star * np.sqrt(R_eff * delta)
    ks = 8 * G_star * np.sqrt(R_eff * delta)
    return kn, ks


def contact_normal_force(delta, R_eff, E_grain, nu_grain):
    """Hertzian normal contact force F_n = (4/3) E* sqrt(R_eff) delta^1.5."""
    E_star = E_grain / (2 * (1 - nu_grain**2))
    return (4.0 / 3.0) * E_star * np.sqrt(R_eff) * delta**1.5


def effective_radius(r_i, r_j):
    return (r_i * r_j) / (r_i + r_j)

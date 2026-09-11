# hertz-mindlin-dem

Contact-mechanics-based (Hertz-Mindlin) synthetic Vp/Vs from real DEM
particle positions, as an alternative to the empirical Botter et al.
(2014) rock-physics workflow used in
[3D_IG-FEM](https://github.com/Soo-jung-AN/3D_IG-FEM). Rather than
fitting curves to lab-measured Vp/porosity/strain data, this derives
Vp and Vs from first principles: per-contact Hertzian stiffness ->
local coordination number / porosity / confining pressure -> effective
bulk and shear moduli (K, G) -> Vp = sqrt((K+4G/3)/rho), Vs = sqrt(G/rho).

Sample data (`data/`) is copied from
[Soo-jung-AN/IG-FEM](https://github.com/Soo-jung-AN/IG-FEM): particle
positions and radii from a horizontal-shortening (biaxial-style
compaction) DEM test, 17,694 particles. `igfem_assembly.py` /
`igfem_preprocessing.py` (also copied from that repo) and
`rock_physics.py` (copied from
[3D_IG-FEM](https://github.com/Soo-jung-AN/3D_IG-FEM)) are included so
`traveltime_compare.py` can reproduce the Botter et al. Vp field here
too, without depending on either sibling repo at runtime.

## Traveltime comparison (three routes)

`traveltime_compare.py` builds **three** Vp fields from the *same* DEM
deformation and pushes each through the same eikonal solve, so the
models can be compared on an actually observable quantity rather than
on the Vp fields themselves:

| route | strain | Vp from |
|---|---|---|
| SSPX + Botter | nearest-neighbor local deformation gradient (what Botter et al. themselves used, via Cardozo & Allmendinger's SSPX) | rock_physics.py Eqs. 1-4 |
| IG-FEM + Botter | global mass-matrix FEM projection (this project's method) | rock_physics.py Eqs. 1-4 |
| Hertz-Mindlin | none — contact mechanics only | K, G from contact stiffness |

```
pip install scikit-fmm
python traveltime_compare.py
python make_traveltime_figures.py
```

Both Botter routes get an identical **depth-dependent compaction trend**
for the pre-strain properties (Vp_ini 1.8 km/s at the free surface to
4.0 km/s at the base, phi_ini 0.35 to 0.15), assigned in the undeformed
configuration the way Botter et al. assign properties before faulting.
That trend matters: with a single homogeneous `Vp_ini` the velocity
field has essentially no vertical gradient, so there are no diving
waves and the t-x curve collapses to a featureless straight line.

Results (shot at mid-line on the free surface, two-sided spread):

- **SSPX vs IG-FEM: the strain fields barely correlate point-by-point
  (r = +0.055), yet the traveltimes agree to within 0.14 s** (mean
  +0.053 s). Traveltime is a path integral, so it averages out the
  pointwise disagreement between a local least-squares estimator and a
  globally smoothed FEM projection — the two strain methods are
  effectively interchangeable *for this observable*, even though their
  per-particle strain maps look quite different.
- **Hertz-Mindlin runs ~0.8 s slow throughout** (max 1.19 s), because
  it generates Vp from scratch via contact stiffness and so predicts a
  genuinely soft, slow near-surface layer (down to ~380 m/s) where
  confining pressure and coordination number are lowest. Botter's curve
  is anchored to `Vp_ini` and can only move it by +/-25%, so it never
  gets that soft.

See `fig_tt_maps.png` (Vp and traveltime fields at true 1:1 aspect) and
`fig_tt_surface_curve.png` (two-sided t-x curve and residuals).

## Why per-contact, not per-particle

Hertz-Mindlin contact stiffness is a property of a *contact* (a pair
of touching particles), not of a particle. To get a continuum-scale
elastic modulus usable in a wave-velocity formula, you need two steps:

1. **Per contact**: normal/tangential stiffness from the contact
   overlap and the grain material's elastic constants.
2. **Homogenization**: average many contacts (via each particle's local
   coordination number, packing density, and confining stress) into an
   effective bulk/shear modulus for that neighborhood.

## Pipeline

1. **`contact_mechanics.py`** -- detect contacts geometrically (center
   distance <= sum of radii) from positions + radii alone (no raw
   contact-pair file exists in the source data), and compute the
   Hertzian normal force from the overlap:
   - `F_n = (4/3) E* sqrt(R_eff) * delta^1.5` (Hertz, 1882)
   - `E* = E_grain / (2*(1-nu_grain^2))`, `R_eff = r_i*r_j/(r_i+r_j)`
2. **`local_fields.py`** -- per particle:
   - coordination number = contact count
   - local porosity via a **measurement-circle** coarse-graining
     estimator (radius = 8x mean particle radius): `1 - (area of
     particles centered inside the circle) / (circle area)`. This is
     used instead of a plain Voronoi tessellation because ordinary
     (unweighted) Voronoi cells of a *polydisperse* packing can be
     smaller than a large particle's own circle, giving nonsensical
     negative porosity for a sizeable fraction of particles -- see
     "Known limitations" below.
   - local confining pressure via the micromechanical/virial stress
     tensor (Christoffersen, Mehrabadi & Nemat-Nasser, 1981):
     `sigma_i = (1/(2*A)) * sum_contacts F_n (x) l`, `P_i = 0.5*trace(sigma_i)`
3. **`effective_medium.py`** -- Hertz-Mindlin effective moduli for a
   random dense sphere pack (Mindlin, 1949; Digby, 1981; Walton, 1987;
   as tabulated in Mavko, Mukerji & Dvorkin, *The Rock Physics
   Handbook*):

   ```
   K_HM = [ C^2 (1-phi)^2 G_grain^2 / (18 pi^2 (1-nu_grain)^2) * P ] ^ (1/3)
   G_HM = (5-4*nu_grain)/(5*(2-nu_grain))
          * [ 3 C^2 (1-phi)^2 G_grain^2 / (2 pi^2 (1-nu_grain)^2) * P ] ^ (1/3)
   ```

4. `Vp = sqrt((K + 4G/3)/rho)`, `Vs = sqrt(G/rho)`, `rho = rho_grain*(1-phi)` (dry).

Default grain properties are quartz: `E_grain=94.5 GPa`, `nu_grain=0.17`,
`rho_grain=2650 kg/m3` -- edit the constants at the top of `main.py` for
a different mineralogy.

## Run it

```
pip install numpy scipy matplotlib
python main.py data 66494       # or: 30283 (moderate strain) / 5932 (undeformed)
python make_figures.py
```

## Result: Vp/Vs is a fixed constant under pure Hertz-Mindlin

Running this on the real DEM data gives coordination number, porosity,
and confining pressure that vary substantially across the sample (see
`fig_hm_maps.png`) -- and Vp and Vs individually vary with them. But
**Vp/Vs itself comes out numerically constant across all 15,426 valid
particles** (`fig_vpvs_constant.png`, std ~1e-16, i.e. floating-point
noise). This isn't a bug: algebraically,

```
K_HM / G_HM = 5*(2-nu_grain) / (3*(5-4*nu_grain))
```

is independent of C, phi, and P -- they cancel because K_HM and G_HM
have the *same* dependence on coordination number, porosity and
pressure, differing only by a nu_grain-dependent prefactor. So plain
Hertz-Mindlin (no friction/slip, no cementation) predicts Vp/Vs is
fixed by the grain's Poisson's ratio alone, however heterogeneous the
packing is. This is a known property/critique of the basic HM contact
model in the rock-physics literature -- it's exactly what motivates
the *empirical* Botter et al. approach used in 3D_IG-FEM, where Vp and
Vs are fit to real strain-dependent lab data separately (via Han's
regression, which has a nonzero intercept), so their ratio *does* vary
with strain.

Natural extensions to break this degeneracy: partial slip at contacts
(a tangential/normal stiffness ratio below the no-slip Mindlin value),
a cementation model (grains bonded rather than in pure Hertzian
contact -- e.g. the constant-cement model), or anisotropic contact
fabric.

## Known limitations

- **Length units are assumed to already be SI (meters, Pa).** The
  source DEM's actual length unit is not documented; if positions/radii
  are really in mm (or another unit), rescale before running --
  absolute Vp/Vs/pressure values are sensitive to this (Vp/Vs itself is
  not, per above, since it only depends on nu_grain).
- **2D disks via 3D-sphere formulas.** Hertz-Mindlin's closed forms
  above are derived for an isotropic random pack of *spheres*; applying
  them to a 2D disk (cylinder) packing is the standard simplification
  used across 2D DEM/rock-physics work (2D DEM contact laws are
  themselves usually calibrated as Hertzian-like), not an exact
  solution of 2D contact mechanics.
- **Measurement-circle porosity has a boundary bias.** A particle whose
  center falls just inside the measurement circle but whose own area
  extends outside it is still counted in full, which can push apparent
  local solid fraction above 100% in dense regions -- `main.py` clips
  porosity to `[0.02, 0.60]` rather than discarding those particles.
  Increasing `R_m_factor` in `local_fields.py` trades resolution for
  robustness.
- Only the outer contact-detection tolerance from `contact_mechanics.py`
  determines what counts as "touching" -- there's no access to the
  original DEM engine's actual contact list or force-displacement law,
  so `delta` (and everything downstream) is a geometric reconstruction,
  not the simulator's ground truth.

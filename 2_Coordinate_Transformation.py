# [Distances and Coordinates]

# Distance adoption: d = 1/parallax (parallax-inversion)
#
# We adopt the simple inverse-parallax estimator  d = 1/varpi  (with varpi in arcsec,
# or equivalently d [pc] = 1000/varpi [mas]) rather than a Bayesian distance estimator
# (e.g. Bailer-Jones et al. 2021).
#
# Justification: our RUWE < 1.4 and parallax_over_error > 5 cuts guarantee that every
# star in the sample has a fractional parallax uncertainty sigma_varpi/varpi < 0.2.
# In this high-SNR regime the inverse-parallax estimator is nearly unbiased and its
# error is well-approximated by  sigma_d / d = sigma_varpi / varpi  (< 20%).
# The Bayesian estimator is essential only when sigma_varpi/varpi > ~0.2, where the
# asymmetric parallax-to-distance transformation introduces significant bias; that
# regime was already removed by our quality cuts, so the added complexity of a
# Bayesian prior is unnecessary here.

from astropy.table import Table
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

# Load K-Dwarf sample from 1_Data_Acquisition
kdwarfs = Table.read("gaia_KDwarfs.fits")

ra    = np.array(kdwarfs["ra"],              dtype=float)  # deg
dec   = np.array(kdwarfs["dec"],             dtype=float)  # deg
plx   = np.array(kdwarfs["parallax"],        dtype=float)  # mas
pmra  = np.array(kdwarfs["pmra"],            dtype=float)  # mas/yr (includes cos dec)
pmdec = np.array(kdwarfs["pmdec"],           dtype=float)  # mas/yr
rv    = np.array(kdwarfs["radial_velocity"], dtype=float)  # km/s

# varpi [mas] -> d [pc]
d_pc = 1000.0 / plx  # pc

# Solar parameters
R_sun = 8.122 * u.kpc        # Galactocentric distance (Reid et al. 2019)
z_sun = 20.8  * u.pc         # Height above midplane (Bennett & Bovy 2019)
v_sun = coord.CartesianDifferential([11.1, 232.24, 7.25] * u.km / u.s)
# (U, V_circ + V_pec, W) = (11.1, 220.0 + 12.24, 7.25) km/s
# Schoenrich, Binney & Dehnen 2010; V_LSR = 220 km/s folded into Y-component

galactocentric_frame = Galactocentric(
    galcen_distance=R_sun,
    z_sun=z_sun,
    galcen_v_sun=v_sun,
)

# SkyCoord with full 6-D phase space
coords = SkyCoord(
    ra=ra                * u.deg,
    dec=dec              * u.deg,
    distance=d_pc        * u.pc,
    pm_ra_cosdec=pmra    * u.mas/u.yr,
    pm_dec=pmdec         * u.mas/u.yr,
    radial_velocity=rv   * u.km/u.s,
    frame="icrs",
)

# Transform to Galactocentric frame
gc = coords.transform_to(galactocentric_frame)

# Galactocentric Cartesian positions [pc]
X = gc.x.to(u.pc).value
Y = gc.y.to(u.pc).value
Z = gc.z.to(u.pc).value  # height above/below midplane

# Galactocentric Cartesian velocities [km/s]
vX = gc.v_x.to(u.km/u.s).value
vY = gc.v_y.to(u.km/u.s).value
vZ = gc.v_z.to(u.km/u.s).value  # vertical velocity

# v_z = z-hat · v_Gal (equation 7)
# In the Galactocentric Cartesian frame z-hat = (0, 0, 1), so the dot product
# simply selects the Z-component of the velocity vector.
v_z = vZ  # km/s

print(f"Sample size : {len(v_z):,} K dwarfs")
print(f"Z range     : {Z.min():.1f} to {Z.max():.1f} pc")
print(f"v_z range   : {v_z.min():.1f} to {v_z.max():.1f} km/s")
print(f"v_z mean    : {v_z.mean():.3f} km/s  (should be near 0 for a relaxed disk)")
print(f"v_z std     : {v_z.std():.2f} km/s")

# Attach new columns and save
kdwarfs["X_gc"]  = X
kdwarfs["Y_gc"]  = Y
kdwarfs["Z_gc"]  = Z
kdwarfs["vX_gc"] = vX
kdwarfs["vY_gc"] = vY
kdwarfs["vZ_gc"] = vZ
kdwarfs["v_z"]   = v_z
kdwarfs["d_pc"]  = d_pc

kdwarfs.write("gaia_Kdwarfs_6D.fits", format="fits", overwrite=True)
print("\nSaved 6-D Galactocentric sample -> 'gaia_KDwarfs_6D.fits'")

# [Uncertainty Propogation]
# For each star, draw N_MC realisations from the multivariate Gaussian defined
# by Gaia's uncertainties, propagate through the full pipeline, and record the
# resulting distributions in z and v_z.

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

e_plx   = np.array(kdwarfs["parallax_error"],        dtype=float)
e_pmra  = np.array(kdwarfs["pmra_error"],             dtype=float)
e_pmdec = np.array(kdwarfs["pmdec_error"],            dtype=float)
e_rv    = np.array(kdwarfs["radial_velocity_error"],  dtype=float)
N_STARS = len(kdwarfs)

try:
    corr_pm = np.array(kdwarfs["pmra_pmdec_corr"], dtype=float)
except KeyError:
    corr_pm = np.zeros(N_STARS)
    print("Warning: pmra_pmdec_corr not found, assuming zero correlation.")

def propagate_mc(n_mc: int, seed: int = 0):
    rng = np.random.default_rng(seed)

    # Correlated pmra/pmdec draws, independent parallax and rv
    z1  = rng.standard_normal((N_STARS, n_mc))
    z2  = rng.standard_normal((N_STARS, n_mc))
    rho = corr_pm[:, None]

    plx_s   = plx[:,None]  + e_plx[:,None]  * rng.standard_normal((N_STARS, n_mc))
    pmra_s  = pmra[:,None] + e_pmra[:,None] * z1
    pmdec_s = pmdec[:,None] + e_pmdec[:,None] * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    rv_s    = rv[:,None]   + e_rv[:,None]   * rng.standard_normal((N_STARS, n_mc))

    plx_s = np.where(plx_s > 0.01, plx_s, np.nan)
    d_s   = 1000.0 / plx_s  # pc

    sc = SkyCoord(
        ra=np.repeat(ra, n_mc)               * u.deg,
        dec=np.repeat(dec, n_mc)             * u.deg,
        distance=d_s.ravel()                 * u.pc,
        pm_ra_cosdec=pmra_s.ravel()          * u.mas/u.yr,
        pm_dec=pmdec_s.ravel()               * u.mas/u.yr,
        radial_velocity=rv_s.ravel()         * u.km/u.s,
        frame="icrs",
    )
    gc_mc = sc.transform_to(galactocentric_frame)

    z_all  = gc_mc.z.to(u.pc).value.reshape(N_STARS, n_mc)
    vz_all = gc_mc.v_z.to(u.km/u.s).value.reshape(N_STARS, n_mc)

    return (
        np.nanmedian(z_all,  axis=1), np.nanstd(z_all,  axis=1),
        np.nanmedian(vz_all, axis=1), np.nanstd(vz_all, axis=1),
    )

# Main run
N_MC = 100
print(f"\nRunning MC with N_MC = {N_MC} draws per star ({N_STARS * N_MC:,} total) ...")
t0 = time.time()
z_med, z_std, vz_med, vz_std = propagate_mc(N_MC)
print(f"Done in {time.time() - t0:.1f}s")
print(f"Median sigma_z   : {np.nanmedian(z_std):.2f} pc")
print(f"Median sigma_v_z : {np.nanmedian(vz_std):.2f} km/s")

# Convergence test
mc_trials = [10, 20, 50, 100, 200]
conv_z, conv_vz = [], []
print("\nConvergence test:")
for nmc in mc_trials:
    t0 = time.time()
    _, z_s, _, vz_s = propagate_mc(nmc, seed=7)
    mz, mvz = np.nanmedian(z_s), np.nanmedian(vz_s)
    conv_z.append(mz)
    conv_vz.append(mvz)
    print(f"  N_MC={nmc:>4d}  median sigma_z={mz:.3f} pc   median sigma_vz={mvz:.3f} km/s  ({time.time()-t0:.1f}s)")

# Save with MC columns attached
kdwarfs["z_mc_med"]  = z_med
kdwarfs["z_mc_std"]  = z_std
kdwarfs["vz_mc_med"] = vz_med
kdwarfs["vz_mc_std"] = vz_std

kdwarfs.write("gaia_KDwarfs_6D.fits", format="fits", overwrite=True)
print("\nSaved -> 'gaia_KDwarfs_6D.fits'  (includes MC uncertainty columns)")

# Plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor("#0d1117")
for ax in axes:
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

axes[0].hist(z_std, bins=80, color="#4C8BF5", edgecolor="none", alpha=0.85, log=True)
axes[0].axvline(np.nanmedian(z_std), color="#FFC107", lw=1.5, linestyle="--",
                label=f"median = {np.nanmedian(z_std):.1f} pc")
axes[0].set_xlabel(r"$\sigma_z$  [pc]", fontsize=12, color="white")
axes[0].set_ylabel("Number of stars (log)", fontsize=11, color="white")
axes[0].set_title(r"MC uncertainty in $z$", fontsize=12, color="white", fontweight="bold")
axes[0].legend(fontsize=9, facecolor="#1a1f2b", edgecolor="#444", labelcolor="white")

axes[1].hist(vz_std, bins=80, color="#FF6B6B", edgecolor="none", alpha=0.85, log=True)
axes[1].axvline(np.nanmedian(vz_std), color="#FFC107", lw=1.5, linestyle="--",
                label=f"median = {np.nanmedian(vz_std):.2f} km/s")
axes[1].set_xlabel(r"$\sigma_{v_z}$  [km/s]", fontsize=12, color="white")
axes[1].set_ylabel("Number of stars (log)", fontsize=11, color="white")
axes[1].set_title(r"MC uncertainty in $v_z$", fontsize=12, color="white", fontweight="bold")
axes[1].legend(fontsize=9, facecolor="#1a1f2b", edgecolor="#444", labelcolor="white")

ax3 = axes[2]
ax3t = ax3.twinx()
ax3.plot(mc_trials, conv_z,  "o-", color="#4C8BF5", lw=2, label=r"median $\sigma_z$ [pc]")
ax3t.plot(mc_trials, conv_vz, "s-", color="#FF6B6B", lw=2, label=r"median $\sigma_{v_z}$ [km/s]")
ax3.set_xlabel("N Monte Carlo draws", fontsize=12, color="white")
ax3.set_ylabel(r"median $\sigma_z$  [pc]", fontsize=11, color="#4C8BF5")
ax3t.set_ylabel(r"median $\sigma_{v_z}$  [km/s]", fontsize=11, color="#FF6B6B")
ax3.set_title("Convergence w.r.t. N_MC", fontsize=12, color="white", fontweight="bold")
ax3.tick_params(colors="white")
ax3t.tick_params(colors="white")
for sp in ax3t.spines.values():
    sp.set_edgecolor("#444")
ax3.set_xscale("log")
ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3t.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
           facecolor="#1a1f2b", edgecolor="#444", labelcolor="white")

plt.suptitle("Monte Carlo Uncertainty Propagation - Gaia DR3 K Dwarfs",
             fontsize=13, fontweight="bold", color="white", y=1.01)
plt.tight_layout()
plt.savefig("uncertainty_Propagation.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Plot saved -> 'uncertainty_Propagation.png'")
# [Distances and Coordinates]

# Distance adoption: d = 1/parallax (parallax-inversion)
#
# We adopt the simple inverse-parallax estimator  d = 1/varpi  (with varpi in arcsec,
# or equivalently d [pc] = 1000/varpi [mas]) rather than a Bayesian distance estimator
# (e.g. Bailer-Jones et al. 2021).
#
# Justification: our RUWE < 1.4 and parallax_over_error > 10 cuts guarantee that every
# star in the sample has a fractional parallax uncertainty sigma_varpi/varpi < 0.1.
# In this high-SNR regime the inverse-parallax estimator is nearly unbiased and its
# error is well-approximated by  sigma_d / d = sigma_varpi / varpi  (< 10%).
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
kdwarfs = Table.read("[1]_Gaia_KDwarfs.fits")

ra    = np.array(kdwarfs["ra"],dtype=float)  # deg
dec   = np.array(kdwarfs["dec"], dtype=float)  # deg
plx   = np.array(kdwarfs["parallax"],dtype=float)  # mas
pmra  = np.array(kdwarfs["pmra"],dtype=float)  # mas/yr (includes cos dec)
pmdec = np.array(kdwarfs["pmdec"],dtype=float)  # mas/yr
rv    = np.array(kdwarfs["radial_velocity"],dtype=float)  # km/s

# varpi [mas] -> d [pc]
d_pc = 1000.0 / plx  # pc

# Solar parameters
R_sun = 8.122 * u.kpc # Galactocentric distance (Reid et al. 2019)
z_sun = 20.8  * u.pc # Height above midplane (Bennett & Bovy 2019)
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

kdwarfs.write("[2]_Gaia_Kdwarfs_6D.fits",format="fits",overwrite=True)
print("\nSaved 6-D Galactocentric sample -> '[2]_Gaia_KDwarfs_6D.fits'")

# [Uncertainty Propogation]
# For each star, draw N_MC realisations from the multivariate Gaussian defined
# by Gaia's uncertainties, propagate through the full pipeline, and record the
# resulting distributions in z and v_z.

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time

e_plx = np.array(kdwarfs["parallax_error"],dtype=float)
e_pmra = np.array(kdwarfs["pmra_error"],dtype=float)
e_pmdec = np.array(kdwarfs["pmdec_error"],dtype=float)
e_rv = np.array(kdwarfs["radial_velocity_error"],dtype=float)
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
N_MC = 500
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

kdwarfs.write("[2]_Gaia_KDwarfs_6D.fits", format="fits", overwrite=True)
print("\nSaved -> '[2]_Gaia_KDwarfs_6D.fits'  (includes MC uncertainty columns)")

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
plt.savefig("[2]_Uncertainty_Propagation.png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Plot saved -> '[2]_Uncertainty_Propagation.png'")

# [Measuring Density and Dispersion]

# We bin stars in vertical height z and compute:
#
#   ν(z) ≈ N(z) / V(z)
#
# where N(z) is the number of stars in a bin and V(z) is the effective survey volume.
# In this analysis, we approximate V(z) using the geometric volume available inside the d < 1.5 kpc sphere
# and treat star counts as a proxy for
# the vertical density profile up to a constant normalization.
#
# Justification:
#   - The sample is local (d < 1.5 kpc) and approximately all-sky.
#   - We are primarily interested in the SHAPE of ν(z), not its absolute normalization.
#   - The normalization cancels in the Jeans equation when computing derivatives.
#
# We assume symmetry about the Galactic midplane and use |z|.
#
# The vertical velocity dispersion is computed as:
#
#   σ_z^2(z) = <v_z^2> - <v_z>^2
#
# Bootstrap resampling is used to estimate uncertainties.

import numpy as np
import matplotlib.pyplot as plt

# Use MC medians (robust against measurement noise)
z  = kdwarfs["z_mc_med"]
vz = kdwarfs["vz_mc_med"]

# Symmetry about the midplane
z_abs = np.abs(z)

# Define z bins (tunable; will test sensitivity later)
z_bins = np.linspace(0, 1500, 25)  # pc
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

nu = []             # density proxy
sigma_z = []        # velocity dispersion
nu_err = []         # bootstrap uncertainty
sigma_z_err = []

N_boot = 200  # bootstrap resamples

d_max = 1500.0   # pc

for i in range(len(z_bins)-1):

    z_lo = z_bins[i]
    z_hi = z_bins[i+1]
    z_mid = 0.5*(z_lo+z_hi)
    dz = z_hi-z_lo

    in_bin = (z_abs >= z_lo) & (z_abs < z_hi)
    vz_bin = vz[in_bin]
    N = len(vz_bin)

    if N < 10:
        nu.append(np.nan)
        sigma_z.append(np.nan)
        nu_err.append(np.nan)
        sigma_z_err.append(np.nan)
        continue

    # effective survey volume
    Rmax = np.sqrt(max(d_max**2 - z_mid**2,0))
    Veff = np.pi * Rmax**2 * dz

    nu.append(N / Veff)

    mean_vz = np.mean(vz_bin)
    var_vz = np.mean(vz_bin**2) - mean_vz**2
    sigma = np.sqrt(var_vz)
    sigma_z.append(sigma)

    # bootstrap
    nu_boot=[]
    sig_boot=[]

    for _ in range(N_boot):
        sample=np.random.choice(vz_bin,size=N,replace=True)

        nu_boot.append(len(sample)/Veff)

        m=np.mean(sample)
        v=np.mean(sample**2)-m**2
        sig_boot.append(np.sqrt(v))

    nu_err.append(np.std(nu_boot))
    sigma_z_err.append(np.std(sig_boot))

# Convert to arrays
nu = np.array(nu)
sigma_z = np.array(sigma_z)
nu_err = np.array(nu_err)
sigma_z_err = np.array(sigma_z_err)

print("\nDensity/dispersion measurement complete.")
print(f"Number of z bins: {len(z_centers)}")
print(f"Median σ_z: {np.nanmedian(sigma_z):.2f} km/s")

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Density profile
ax[0].errorbar(z_centers, nu, yerr=nu_err, fmt='o', markersize=4)
ax[0].set_xlabel("z [pc]")
ax[0].set_ylabel(r"Volume-corrected tracer density $\nu(z)$ [stars pc$^{-3}$]")
ax[0].set_title("Vertical Density Profile")
ax[0].grid(alpha=0.3)

# Velocity dispersion profile
ax[1].errorbar(z_centers, sigma_z, yerr=sigma_z_err, fmt='o', markersize=4)
ax[1].set_xlabel("z [pc]")
ax[1].set_ylabel("σ_z [km/s]")
ax[1].set_title("Vertical Velocity Dispersion")
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("[2]_Density_dispersion.png", dpi=150)
plt.show()

print("Plot saved -> '[2]_Density_Dispersion.png'")

# [Mass Modeling and Statistical Inference]
#
# The central goal is to infer rho_DM by forward-modeling the vertical
# dynamics and comparing predicted sigma_z(z) to the observed profile.
#
# The inference chain is:
#   rho_DM -> rho(z) -> Kz(z) -> sigma_z(z)
#
# This file sets up the mass model and the forward model only.
# Statistical inference (likelihood, MCMC, posterior) is handled in [3].

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from astropy.table import Table

# Gravitational constant in convenient units: pc Msun^-1 (km/s)^2
G_grav = 4.3009e-3

# Load binned density and dispersion profiles from previous section
# (z_centers, nu, sigma_z, nu_err, sigma_z_err already in memory)

# [Parameterization of the Mass Model]
#
# We model the total density as (equation 10):
#   rho(z | theta) = rho_star(z | theta) + rho_gas(z) + rho_DM
#
# Stellar disk: two-component sech^2 (thin + thick disk)
#   rho_star(z) = rho_thin  * sech^2(z / (2 * h_thin))
#               + rho_thick * sech^2(z / (2 * h_thick))
#
#   Physical justification: K dwarfs trace both the thin and thick disk.
#   The sech^2 profile is the self-consistent solution for an isothermal
#   sheet in equilibrium, and is standard in Galactic disk modeling.
#
# Gas disk: fixed sech^2, not a free parameter
#   rho_gas(z) = rho_gas0 * sech^2(z / (2 * h_gas))
#   rho_gas0 = 0.04 Msun/pc^3,  h_gas = 150 pc  (Flynn et al. 2006)
#
#   Justification: the gas distribution is well-constrained by HI/CO
#   surveys and its uncertainty has a small effect on rho_DM compared
#   to the stellar disk. We fix it to reduce parameter degeneracy.
#
# Dark matter: constant over |z| < 1 kpc
#   Justification: the DM halo scale length is ~several kpc, so its
#   density gradient over the ~1 kpc range we probe is negligible.
#
# Free parameter vector theta = (rho_thin, h_thin, rho_thick, h_thick, rho_DM)
#   rho_thin   [Msun/pc^3]  midplane thin-disk density
#   h_thin     [pc]         thin-disk scale height
#   rho_thick  [Msun/pc^3]  midplane thick-disk density
#   h_thick    [pc]         thick-disk scale height
#   rho_DM     [Msun/pc^3]  local dark matter density  <-- target

RHO_GAS0 = 0.04    # Msun/pc^3  (Flynn et al. 2006)
H_GAS    = 150.0   # pc

def rho_sech2(z, rho0, h):
    """Sech^2 density profile: rho0 * sech^2(z / (2h))."""
    return rho0 / np.cosh(z / (2.0 * h))**2

def rho_total(z_arr, rho_thin, h_thin, rho_thick, h_thick, rho_DM):
    """
    Total vertical mass density [Msun/pc^3] at each z [pc].
    Equation (10): rho = rho_star + rho_gas + rho_DM
    """
    rho_star = (rho_sech2(z_arr, rho_thin,  h_thin) +
                rho_sech2(z_arr, rho_thick, h_thick))
    rho_gas  =  rho_sech2(z_arr, RHO_GAS0, H_GAS)
    return rho_star + rho_gas + rho_DM

# [From Density to Observable Quantities]
#
# For a given theta, we follow the inference chain (equation 14):
#
# Step 1 - Poisson's equation (equation 11):
#   dKz/dz = -4*pi*G * rho(z | theta)
#
# Step 2 - Integrate to get Kz(z) (equation 11), imposing Kz(0) = 0
#   (midplane symmetry: the disk is reflection-symmetric about z = 0,
#   so the net vertical force at the midplane must vanish)
#
# Step 3 - Insert Kz(z) into the Jeans equation (equation 12):
#   d/dz [nu(z) * sigma_z^2(z)] = nu(z) * Kz(z)
#
# Step 4 - Solve for sigma_z_model(z) (equation 13):
#   sigma_z^2(z) = (1/nu(z)) * integral_z^z_max nu(z') * |Kz(z')| dz'
#
#   This outer integral form follows from integrating the Jeans equation
#   outward, applying the boundary condition sigma_z^2 -> 0 at large |z|.
#   Physically: the dispersion at height z is supported by the weight of
#   all material above it, which is why larger rho_DM (more force at
#   large z) raises sigma_z at large z.

# Fine evaluation grid for numerical integration
z_grid = np.linspace(0.0, 1500.0, 3000)   # pc

def compute_Kz(z_arr, rho_thin, h_thin, rho_thick, h_thick, rho_DM):
    """
    Vertical gravitational force Kz(z) [(km/s)^2/pc].
    Kz(0) = 0 by midplane symmetry; Kz < 0 for z > 0 (restoring force).
    Obtained by integrating Poisson's equation (equation 11).
    """
    rho      = rho_total(z_arr, rho_thin, h_thin, rho_thick, h_thick, rho_DM)
    dKz_dz   = -4.0 * np.pi * G_grav * rho
    Kz       = cumulative_trapezoid(dKz_dz, z_arr, initial=0.0)
    return Kz   # (km/s)^2 / pc

def compute_sigma_z_model(z_centers, nu_obs, rho_thin, h_thin,
                          rho_thick, h_thick, rho_DM):
    """
    Predict sigma_z [km/s] at each observed bin center z_centers [pc]
    by solving the vertical Jeans equation (equations 12-13).

    Parameters
    ----------
    z_centers : array  observed bin centers [pc]
    nu_obs    : array  observed star counts (proxy for nu(z))
    rho_thin, h_thin, rho_thick, h_thick, rho_DM : floats  model parameters

    Returns
    -------
    sigma_model : array  predicted sigma_z at each z_center [km/s]
    """
    # Step 1 & 2: Kz on fine grid
    Kz_grid = compute_Kz(z_grid, rho_thin, h_thin, rho_thick, h_thick, rho_DM)

    # Interpolate observed nu onto fine grid
    nu_fn   = interp1d(
        z_centers, nu_obs,
        kind="linear", bounds_error=False,
        fill_value=(nu_obs[0], nu_obs[-1]),
    )
    nu_grid = np.clip(nu_fn(z_grid), 1e-6, None)

    # Step 3 & 4: solve Jeans equation via outer integral
    #   sigma_z^2(z) = (1/nu(z)) * integral_z^z_max nu * |Kz| dz'
    integrand    = nu_grid * np.abs(Kz_grid)
    cum_int      = cumulative_trapezoid(integrand, z_grid, initial=0.0)
    outer_int    = cum_int[-1] - cum_int          # integral from z to z_max
    sigma2_grid  = np.clip(outer_int / nu_grid, 0.0, None)

    # Interpolate back onto observed bin centers
    sigma_fn = interp1d(
        z_grid, np.sqrt(sigma2_grid),
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )
    return sigma_fn(z_centers)

# Quick sanity check with literature-motivated parameters

theta_test = (0.10, 300.0, 0.02, 900.0, 0.013)   # (rho_thin, h_thin, rho_thick, h_thick, rho_DM)

valid = (
    np.isfinite(sigma_z) & np.isfinite(sigma_z_err) &
    np.isfinite(nu) & (nu > 0) & (sigma_z_err > 0)
)
z_obs   = z_centers[valid]
nu_obs  = nu[valid]
sig_obs = sigma_z[valid]
sig_err = sigma_z_err[valid]

sig_test = compute_sigma_z_model(z_obs, nu_obs, *theta_test)

print("Forward model sanity check (literature theta):")
for z_i, s_obs, s_mod in zip(z_obs, sig_obs, sig_test):
    print(f"  z = {z_i:6.0f} pc   sigma_obs = {s_obs:.2f}   sigma_model = {s_mod:.2f} km/s")

# Save relevant quantities from coordinate transformation, MC propagation,
# binning, and forward-model sanity check.

summary_file = "[2]_Run_Summary.txt"

finite_zstd = z_std[np.isfinite(z_std)]
finite_vzstd = vz_std[np.isfinite(vz_std)]
finite_nu = nu[np.isfinite(nu)]
finite_sig = sigma_z[np.isfinite(sigma_z)]

with open(summary_file, "w") as f:

    f.write("GAIA DR3 K-DWARF 6D COORDINATE + BINNING SUMMARY\n")
    f.write("="*70 + "\n\n")

    f.write("--Input Sample--\n")
    f.write(f"Input file: [1]_Gaia_KDwarfs.fits\n")
    f.write(f"Number of K-dwarf tracers: {len(kdwarfs):,}\n")
    f.write(f"Median distance: {np.nanmedian(d_pc):.2f} pc\n")
    f.write(f"Distance range: {np.nanmin(d_pc):.2f} to {np.nanmax(d_pc):.2f} pc\n\n")

    f.write("--Galactocentric Transformation--\n")
    f.write(f"R_sun: {R_sun}\n")
    f.write(f"z_sun: {z_sun}\n")
    f.write("v_sun: [11.1, 232.24, 7.25] km/s\n")
    f.write(f"Z range: {np.nanmin(Z):.2f} to {np.nanmax(Z):.2f} pc\n")
    f.write(f"v_z range: {np.nanmin(v_z):.2f} to {np.nanmax(v_z):.2f} km/s\n")
    f.write(f"v_z mean: {np.nanmean(v_z):.4f} km/s\n")
    f.write(f"v_z std: {np.nanstd(v_z):.4f} km/s\n\n")

    f.write("--Monte Carlo Propagation--\n")
    f.write(f"N_MC main run: {N_MC}\n")
    f.write(f"Median sigma_z position uncertainty: {np.nanmedian(finite_zstd):.4f} pc\n")
    f.write(f"Median sigma_v_z uncertainty: {np.nanmedian(finite_vzstd):.4f} km/s\n")
    f.write("MC convergence test:\n")
    for nmc, mz, mvz in zip(mc_trials, conv_z, conv_vz):
        f.write(f"  N_MC={nmc:>4d}: median sigma_z={mz:.4f} pc, median sigma_vz={mvz:.4f} km/s\n")
    f.write("\n")

    f.write("--Density and Dispersion Binning--\n")
    f.write(f"z bin range: {z_bins[0]:.1f} to {z_bins[-1]:.1f} pc\n")
    f.write(f"Number of z bins: {len(z_centers)}\n")
    f.write(f"Bootstrap resamples: {N_boot}\n")
    f.write(f"Distance limit used for Veff: d_max = {d_max:.1f} pc\n")
    f.write(f"Finite bins used: {np.sum(valid)}\n")
    f.write(f"Median volume-corrected tracer density: {np.nanmedian(finite_nu):.6e} stars/pc^3\n")
    f.write(f"Median sigma_z profile value: {np.nanmedian(finite_sig):.4f} km/s\n\n")

    f.write("--Forward Model Sanity Check--\n")
    f.write("Mass model test parameters:\n")
    f.write(f"  rho_thin  = {theta_test[0]:.4f} Msun/pc^3\n")
    f.write(f"  h_thin    = {theta_test[1]:.2f} pc\n")
    f.write(f"  rho_thick = {theta_test[2]:.4f} Msun/pc^3\n")
    f.write(f"  h_thick   = {theta_test[3]:.2f} pc\n")
    f.write(f"  rho_DM    = {theta_test[4]:.4f} Msun/pc^3\n")
    f.write(f"  sigma_model_midplane ≈ {sig_test[0]:.4f} km/s\n\n")

    f.write("Observed vs model sigma_z:\n")
    for z_i, s_obs, s_err, s_mod in zip(z_obs, sig_obs, sig_err, sig_test):
        f.write(
            f"  z={z_i:8.2f} pc   "
            f"sigma_obs={s_obs:8.4f} +/- {s_err:8.4f}   "
            f"sigma_model={s_mod:8.4f} km/s\n"
        )
    f.write("\n")

    f.write("--Files Produced--\n")
    f.write("[2]_Gaia_KDwarfs_6D.fits\n")
    f.write("[2]_Uncertainty_Propagation.png\n")
    f.write("[2]_Density_Dispersion.png\n")
    f.write("[2]_Run_Summary.txt\n\n")

print(f"\nSaved run summary -> '{summary_file}'")

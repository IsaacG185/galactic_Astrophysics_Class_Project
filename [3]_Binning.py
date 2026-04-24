# [Constructing the Likelihood]
#
# This file prepares the binned tracer density, velocity-dispersion profile,
# and likelihood functions used by [4]_Inference.py.
#
# To reduce baryon-DM degeneracy, this version uses the same reduced model as [4]:
#
#   theta = (sigma_top, rho_thin, rho_DM)
#
# with fixed:
#   h_thin = 300 pc
#   rho_thick = 0.02 Msun/pc^3
#   h_thick = 900 pc

import numpy as np
from astropy.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# Reload transformed K-dwarf sample from [2]

kdwarfs = Table.read("[2]_Gaia_KDwarfs_6D.fits")

z  = np.array(kdwarfs["z_mc_med"], dtype=float)
vz = np.array(kdwarfs["vz_mc_med"], dtype=float)
z_abs = np.abs(z)

# Recompute volume-corrected density and dispersion profiles

z_bins = np.linspace(0, 1500, 25)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

nu, sigma_z, nu_err, sigma_z_err = [], [], [], []

N_boot = 200
d_max = 1500.0  # pc

for i in range(len(z_bins) - 1):

    z_lo = z_bins[i]
    z_hi = z_bins[i + 1]
    z_mid = 0.5 * (z_lo + z_hi)
    dz = z_hi - z_lo

    in_bin = (z_abs >= z_lo) & (z_abs < z_hi)
    vz_bin = vz[in_bin]
    N = len(vz_bin)

    if N < 10:
        nu.append(np.nan)
        sigma_z.append(np.nan)
        nu_err.append(np.nan)
        sigma_z_err.append(np.nan)
        continue

    # Approximate effective geometric volume inside d < 1.5 kpc sphere
    Rmax = np.sqrt(max(d_max**2 - z_mid**2, 0.0))
    Veff = np.pi * Rmax**2 * dz

    # Volume-corrected tracer density
    nu.append(N / Veff)

    # Velocity dispersion
    mean_vz = np.mean(vz_bin)
    var_vz = np.mean(vz_bin**2) - mean_vz**2
    sigma_z.append(np.sqrt(var_vz))

    # Bootstrap uncertainties
    nu_boot = []
    sig_boot = []

    for _ in range(N_boot):
        s = np.random.choice(vz_bin, size=N, replace=True)

        # N is unchanged under bootstrap resampling, so density uncertainty
        # from this simple bootstrap is effectively zero.
        nu_boot.append(N / Veff)

        m = np.mean(s)
        v = np.mean(s**2) - m**2
        sig_boot.append(np.sqrt(v))

    nu_err.append(np.std(nu_boot))
    sigma_z_err.append(np.std(sig_boot))


nu = np.array(nu)
sigma_z = np.array(sigma_z)
nu_err = np.array(nu_err)
sigma_z_err = np.array(sigma_z_err)

# Fit only reliable z range

# The outer bins are more affected by selection-function uncertainty and
# thick-disk contamination, so the baseline likelihood uses |z| <= 400 pc.

fit_mask = z_centers <= 400

valid = (
    fit_mask
    & np.isfinite(sigma_z)
    & np.isfinite(sigma_z_err)
    & np.isfinite(nu)
    & (nu > 0)
    & (sigma_z_err > 0)
)

z_obs = z_centers[valid]
nu_obs = nu[valid]
sig_obs = sigma_z[valid]
sig_err = sigma_z_err[valid]

print(f"Valid bins used in likelihood: {len(z_obs)}")
print(f"Fit z range: {np.nanmin(z_obs):.2f} to {np.nanmax(z_obs):.2f} pc")
print(f"Median observed sigma_z: {np.nanmedian(sig_obs):.2f} km/s")

# Fixed mass model parameters

G_grav = 4.3009e-3   # pc Msun^-1 (km/s)^2

RHO_GAS0 = 0.04      # Msun/pc^3
H_GAS = 150.0        # pc

# Fixed baryonic disk structure
H_THIN_FIXED = 300.0          # pc
RHO_THICK_FIXED = 0.02        # Msun/pc^3
H_THICK_FIXED = 900.0         # pc

z_fit_max = np.nanmax(z_obs)
z_grid = np.linspace(0.0, z_fit_max, 1500)


def rho_sech2(z_arr, rho0, h):
    return rho0 / np.cosh(z_arr / (2.0 * h))**2


def rho_total(z_arr, rho_thin, rho_DM):
    rho_star = (
        rho_sech2(z_arr, rho_thin, H_THIN_FIXED)
        + rho_sech2(z_arr, RHO_THICK_FIXED, H_THICK_FIXED)
    )

    rho_gas = rho_sech2(z_arr, RHO_GAS0, H_GAS)

    return rho_star + rho_gas + rho_DM


def compute_Kz(z_arr, rho_thin, rho_DM):
    rho = rho_total(z_arr, rho_thin, rho_DM)
    dKz_dz = -4.0 * np.pi * G_grav * rho
    return cumulative_trapezoid(dKz_dz, z_arr, initial=0.0)


def compute_sigma_z_model(
    z_centers,
    nu_obs,
    sigma_top,
    rho_thin,
    rho_DM,
):
    """
    Jeans forward model with finite top-boundary pressure.

    theta = (sigma_top, rho_thin, rho_DM)

    sigma_z^2(z) =
      [nu(zmax) sigma_top^2 + integral_z^zmax nu(z') |Kz(z')| dz'] / nu(z)
    """

    Kz_grid = compute_Kz(
        z_grid,
        rho_thin,
        rho_DM,
    )

    nu_fn = interp1d(
        z_centers,
        nu_obs,
        kind="linear",
        bounds_error=False,
        fill_value=(nu_obs[0], nu_obs[-1]),
    )

    nu_grid = np.clip(nu_fn(z_grid), 1e-12, None)

    # Physically correct Jeans integrand
    integrand = nu_grid * np.abs(Kz_grid)

    cum_int = cumulative_trapezoid(
        integrand,
        z_grid,
        initial=0.0,
    )

    outer_int = cum_int[-1] - cum_int

    boundary_pressure = nu_grid[-1] * sigma_top**2

    sigma2_grid = (boundary_pressure + outer_int) / nu_grid

    if np.any(~np.isfinite(sigma2_grid)) or np.any(sigma2_grid <= 0):
        return np.full_like(z_centers, np.nan, dtype=float)

    sigma_fn = interp1d(
        z_grid,
        np.sqrt(sigma2_grid),
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    return sigma_fn(z_centers)

# Likelihood

def log_likelihood(theta):
    sigma_top, rho_thin, rho_DM = theta

    try:
        sigma_model = compute_sigma_z_model(
            z_obs,
            nu_obs,
            sigma_top,
            rho_thin,
            rho_DM,
        )
    except Exception:
        return -np.inf

    if np.any(~np.isfinite(sigma_model)):
        return -np.inf

    residuals = sig_obs - sigma_model

    return -0.5 * np.sum(
        (residuals / sig_err)**2
        + np.log(2.0 * np.pi * sig_err**2)
    )

# Prior

def log_prior(theta):
    sigma_top, rho_thin, rho_DM = theta

    if not (5.0 < sigma_top < 45.0):
        return -np.inf

    if not (0.08 < rho_thin < 0.14):
        return -np.inf

    if not (0.0 < rho_DM < 0.03):
        return -np.inf

    # Weak Gaussian prior around literature-like local thin-disk density
    lp_rho_thin = -0.5 * ((rho_thin - 0.10) / 0.02)**2

    return lp_rho_thin


def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

# Sanity check

sigma_top_test = sig_obs[-1]

theta_test = (
    sigma_top_test,
    0.10,
    0.013,
)

ll_test = log_likelihood(theta_test)

print("\nLikelihood sanity check")
print("----------------------")
print(f"sigma_top_test = {sigma_top_test:.2f} km/s")
print(f"rho_thin_test  = {theta_test[1]:.4f} Msun/pc^3")
print(f"rho_DM_test    = {theta_test[2]:.4f} Msun/pc^3")
print(f"log-likelihood at literature theta: {ll_test:.2f}")
print("(Should be finite and negative; more negative = worse fit)")

# Run Summary Output

summary_file = "[3]_Run_Summary.txt"

with open(summary_file, "w") as f:

    f.write("BINNING + LIKELIHOOD SETUP SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("--Input--\n")
    f.write("Input file: [2]_Gaia_KDwarfs_6D.fits\n")
    f.write(f"Number of stars loaded: {len(kdwarfs):,}\n\n")

    f.write("--Binning--\n")
    f.write(f"z bin range: {z_bins[0]:.1f} to {z_bins[-1]:.1f} pc\n")
    f.write(f"Number of z bins: {len(z_centers)}\n")
    f.write(f"Fit mask: z_centers <= 400 pc\n")
    f.write(f"Number of valid likelihood bins: {len(z_obs)}\n")
    f.write(f"Fit z range: {np.nanmin(z_obs):.2f} to {np.nanmax(z_obs):.2f} pc\n")
    f.write(f"Bootstrap resamples: {N_boot}\n")
    f.write(f"Distance limit for Veff: d_max = {d_max:.1f} pc\n")
    f.write("Tracer density method: nu(z) = N(z) / Veff(z)\n")
    f.write("Veff(z) approximated using geometric volume inside d < 1.5 kpc sphere.\n\n")

    f.write("--Observed Profiles--\n")
    f.write(f"Median nu(z): {np.nanmedian(nu_obs):.6e} stars/pc^3\n")
    f.write(f"Median sigma_z: {np.nanmedian(sig_obs):.4f} km/s\n")
    f.write(f"Median sigma_z uncertainty: {np.nanmedian(sig_err):.4f} km/s\n")
    f.write(f"z_obs min/max: {np.nanmin(z_obs):.2f} to {np.nanmax(z_obs):.2f} pc\n\n")

    f.write("--Fixed Model Assumptions--\n")
    f.write(f"H_THIN_FIXED: {H_THIN_FIXED:.2f} pc\n")
    f.write(f"RHO_THICK_FIXED: {RHO_THICK_FIXED:.4f} Msun/pc^3\n")
    f.write(f"H_THICK_FIXED: {H_THICK_FIXED:.2f} pc\n")
    f.write(f"RHO_GAS0: {RHO_GAS0:.4f} Msun/pc^3\n")
    f.write(f"H_GAS: {H_GAS:.2f} pc\n\n")

    f.write("--Forward Model--\n")
    f.write("Mass model:\n")
    f.write("rho_total = rho_thin sech^2(z/2H_THIN_FIXED)\n")
    f.write("          + RHO_THICK_FIXED sech^2(z/2H_THICK_FIXED)\n")
    f.write("          + rho_gas + rho_DM\n")
    f.write("Jeans solver uses finite top-boundary pressure sigma_top.\n")
    f.write("MCMC parameter vector should be:\n")
    f.write("  theta = (sigma_top, rho_thin, rho_DM)\n\n")

    f.write("--Priors--\n")
    f.write("sigma_top: 5.0 < sigma_top < 45.0 km/s\n")
    f.write("rho_thin: 0.08 < rho_thin < 0.14 Msun/pc^3\n")
    f.write("rho_DM: 0.0 < rho_DM < 0.03 Msun/pc^3\n")
    f.write("Weak Gaussian prior: rho_thin = 0.10 +/- 0.02 Msun/pc^3\n\n")

    f.write("--Sanity Check--\n")
    f.write(f"sigma_top_test = {sigma_top_test:.4f} km/s\n")
    f.write("theta_test = (\n")
    f.write(f"  sigma_top = {theta_test[0]:.4f} km/s,\n")
    f.write(f"  rho_thin  = {theta_test[1]:.4f} Msun/pc^3,\n")
    f.write(f"  rho_DM    = {theta_test[2]:.4f} Msun/pc^3\n")
    f.write(")\n")
    f.write(f"log-likelihood at theta_test: {ll_test:.4f}\n\n")

    f.write("--Files Produced--\n")
    f.write("[3]_Run_Summary.txt\n\n")

    f.write("--Notes--\n")
    f.write("This file prepares the likelihood functions for [4]_Inference.py.\n")
    f.write("This [3] version matches the reduced three-parameter [4] model.\n")
    f.write("The six-parameter midplane-boundary version should not be used with the current [4].\n")

print(f"\nSaved run summary -> '{summary_file}'")
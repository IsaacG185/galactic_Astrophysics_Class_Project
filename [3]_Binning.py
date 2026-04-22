# [Constructing the Likelihood]

import numpy as np
from astropy.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# ── Reload binned profiles saved at the end of [2] ──────────────────────────
# (z_centers, nu, sigma_z, nu_err, sigma_z_err were computed in [2] and
#  need to be recomputed here from the saved 6D fits file)

kdwarfs = Table.read("[2]_Gaia_KDwarfs_6D.fits")

z  = np.array(kdwarfs["z_mc_med"], dtype=float)
vz = np.array(kdwarfs["vz_mc_med"], dtype=float)
z_abs = np.abs(z)

z_bins    = np.linspace(0, 1500, 25)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

nu, sigma_z, nu_err, sigma_z_err = [], [], [], []
N_boot = 200

for i in range(len(z_bins) - 1):
    in_bin = (z_abs >= z_bins[i]) & (z_abs < z_bins[i+1])
    vz_bin = vz[in_bin]
    N = len(vz_bin)
    if N < 10:
        nu.append(np.nan); sigma_z.append(np.nan)
        nu_err.append(np.nan); sigma_z_err.append(np.nan)
        continue
    nu.append(N)
    mean_vz = np.mean(vz_bin)
    sigma_z.append(np.sqrt(np.mean(vz_bin**2) - mean_vz**2))
    nu_boot, sig_boot = [], []
    for _ in range(N_boot):
        s = np.random.choice(vz_bin, size=N, replace=True)
        nu_boot.append(len(s))
        m = np.mean(s)
        sig_boot.append(np.sqrt(np.mean(s**2) - m**2))
    nu_err.append(np.std(nu_boot))
    sigma_z_err.append(np.std(sig_boot))

nu         = np.array(nu)
sigma_z    = np.array(sigma_z)
nu_err     = np.array(nu_err)
sigma_z_err = np.array(sigma_z_err)

valid   = (
    np.isfinite(sigma_z) & np.isfinite(sigma_z_err) &
    np.isfinite(nu) & (nu > 0) & (sigma_z_err > 0)
)
z_obs   = z_centers[valid]
nu_obs  = nu[valid]
sig_obs = sigma_z[valid]
sig_err = sigma_z_err[valid]

# ── Re-define forward model (originally in [2]) ──────────────────────────────
G_grav   = 4.3009e-3
RHO_GAS0 = 0.04
H_GAS    = 150.0
z_grid   = np.linspace(0.0, 1500.0, 3000)

def rho_sech2(z, rho0, h):
    return rho0 / np.cosh(z / (2.0 * h))**2

def rho_total(z_arr, rho_thin, h_thin, rho_thick, h_thick, rho_DM):
    rho_star = (rho_sech2(z_arr, rho_thin,  h_thin) +
                rho_sech2(z_arr, rho_thick, h_thick))
    rho_gas  =  rho_sech2(z_arr, RHO_GAS0, H_GAS)
    return rho_star + rho_gas + rho_DM

def compute_Kz(z_arr, rho_thin, h_thin, rho_thick, h_thick, rho_DM):
    rho    = rho_total(z_arr, rho_thin, h_thin, rho_thick, h_thick, rho_DM)
    dKz_dz = -4.0 * np.pi * G_grav * rho
    return cumulative_trapezoid(dKz_dz, z_arr, initial=0.0)

def compute_sigma_z_model(z_centers, nu_obs, rho_thin, h_thin,
                          rho_thick, h_thick, rho_DM):
    Kz_grid = compute_Kz(z_grid, rho_thin, h_thin, rho_thick, h_thick, rho_DM)
    nu_fn   = interp1d(z_centers, nu_obs, kind="linear",
                       bounds_error=False, fill_value=(nu_obs[0], nu_obs[-1]))
    nu_grid = np.clip(nu_fn(z_grid), 1e-6, None)
    integrand   = nu_grid * np.abs(Kz_grid)
    cum_int     = cumulative_trapezoid(integrand, z_grid, initial=0.0)
    outer_int   = cum_int[-1] - cum_int
    sigma2_grid = np.clip(outer_int / nu_grid, 0.0, None)
    sigma_fn    = interp1d(z_grid, np.sqrt(sigma2_grid),
                           kind="linear", bounds_error=False,
                           fill_value="extrapolate")
    return sigma_fn(z_centers)

# Likelihood
#
# Assuming Gaussian measurement errors (equation 16):
#   ln L(theta) = -0.5 * sum_i [ (sigma_obs_i - sigma_model_i)^2
#                                 / delta_sigma_i^2 ]

def log_likelihood(theta):
    rho_thin, h_thin, rho_thick, h_thick, rho_DM = theta
    try:
        sigma_model = compute_sigma_z_model(
            z_obs, nu_obs, rho_thin, h_thin, rho_thick, h_thick, rho_DM
        )
    except Exception:
        return -np.inf
    residuals = sig_obs - sigma_model
    return -0.5 * np.sum((residuals / sig_err)**2)

# Sanity check
theta_test = (0.10, 300.0, 0.02, 900.0, 0.013)
ll_test    = log_likelihood(theta_test)
print(f"Log-likelihood at literature theta: {ll_test:.2f}")
print("(Should be finite and negative; more negative = worse fit)")
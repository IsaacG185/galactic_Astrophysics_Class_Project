# [How rho_DM Is Inferred]
#
# Final reduced Jeans model:
#   theta = (sigma_top, rho_thin, rho_DM)
#
# The thin-disk density is allowed to vary, but with a tight literature-motivated prior
# to prevent the baryon-DM degeneracy from dominating the inference.

import numpy as np
from astropy.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import emcee
import corner

# Plot style

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.framealpha": 1,
})

# Reload data

kdwarfs = Table.read("[2]_Gaia_KDwarfs_6D.fits")

z = np.array(kdwarfs["z_mc_med"], dtype=float)
vz = np.array(kdwarfs["vz_mc_med"], dtype=float)
z_abs = np.abs(z)

# Volume-corrected binning

z_bins = np.linspace(0, 1500, 25)
z_centers = 0.5 * (z_bins[:-1] + z_bins[1:])

nu, sigma_z, nu_err, sigma_z_err = [], [], [], []

N_boot = 200
d_max = 1500.0

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

    Rmax = np.sqrt(max(d_max**2 - z_mid**2, 0.0))
    Veff = np.pi * Rmax**2 * dz

    nu.append(N / Veff)

    mean_vz = np.mean(vz_bin)
    var_vz = np.mean(vz_bin**2) - mean_vz**2
    sigma_z.append(np.sqrt(var_vz))

    sig_boot = []

    for _ in range(N_boot):
        s = np.random.choice(vz_bin, size=N, replace=True)
        m = np.mean(s)
        v = np.mean(s**2) - m**2
        sig_boot.append(np.sqrt(v))

    nu_err.append(0.0)
    sigma_z_err.append(np.std(sig_boot))

nu = np.array(nu)
sigma_z = np.array(sigma_z)
nu_err = np.array(nu_err)
sigma_z_err = np.array(sigma_z_err)

# Fit only reliable z range

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

print(f"Valid bins used: {len(z_obs)}")
print(f"Fit range: {np.nanmin(z_obs):.1f} to {np.nanmax(z_obs):.1f} pc")
print(f"Median observed sigma_z: {np.nanmedian(sig_obs):.2f} km/s")

# Fixed mass model parameters

G_grav = 4.3009e-3

RHO_GAS0 = 0.04
H_GAS = 150.0

H_THIN_FIXED = 300.0

RHO_THICK_FIXED = 0.02
H_THICK_FIXED = 900.0

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

# Likelihood and posterior

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

def log_prior(theta):
    sigma_top, rho_thin, rho_DM = theta

    if not (5.0 < sigma_top < 45.0):
        return -np.inf

    if not (0.085 < rho_thin < 0.115):
        return -np.inf

    if not (0.0 < rho_DM < 0.06):
        return -np.inf

    # Tight Gaussian prior on baryonic thin-disk density
    lp_rho_thin = -0.5 * ((rho_thin - 0.10) / 0.005)**2

    return lp_rho_thin

def log_posterior(theta):
    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta)

    if not np.isfinite(ll):
        return -np.inf

    return lp + ll

# MCMC sampling

N_DIM = 3
N_WALKERS = 36
N_STEPS = 6000
N_BURN = 1200
THIN = 15

sigma_top_init = sig_obs[-1]

p0_center = np.array([
    sigma_top_init,
    0.10,
    0.013,
])

rng_init = np.random.default_rng(42)

p0_scale = np.array([
    0.03 * sigma_top_init,
    0.003,
    0.003,
])

p0 = p0_center + p0_scale * rng_init.standard_normal((N_WALKERS, N_DIM))

p0[:, 0] = np.clip(p0[:, 0], 5.1, 44.9)
p0[:, 1] = np.clip(p0[:, 1], 0.086, 0.114)
p0[:, 2] = np.clip(p0[:, 2], 0.001, 0.059)

print(f"\nRunning MCMC: {N_WALKERS} walkers x {N_STEPS} steps...")
sampler = emcee.EnsembleSampler(N_WALKERS, N_DIM, log_posterior)
sampler.run_mcmc(p0, N_STEPS, progress=True)
print("MCMC complete.")

flat_samples = sampler.get_chain(
    discard=N_BURN,
    thin=THIN,
    flat=True,
)

print(f"Flat chain shape: {flat_samples.shape}")

try:
    tau = sampler.get_autocorr_time(discard=N_BURN)
    print(f"\nAutocorrelation times: {np.round(tau, 1)}")
except emcee.autocorr.AutocorrError as e:
    print(f"\nAutocorrelation estimate did not converge: {e}")

# ------------------------------------------------------------
# Posterior summary
# ------------------------------------------------------------

sigma_top_samples = flat_samples[:, 0]
rho_thin_samples = flat_samples[:, 1]
rho_DM_samples = flat_samples[:, 2]

rho_DM_med = np.median(rho_DM_samples)
rho_DM_lo = np.percentile(rho_DM_samples, 16)
rho_DM_hi = np.percentile(rho_DM_samples, 84)

sigma_top_med = np.median(sigma_top_samples)
rho_thin_med = np.median(rho_thin_samples)

theta_med_full = np.median(flat_samples, axis=0)

print("\nInferred local dark matter density:")
print(
    f"rho_DM = {rho_DM_med:.4f} + {rho_DM_hi-rho_DM_med:.4f}"
    f" - {rho_DM_med-rho_DM_lo:.4f} Msun/pc^3"
)

print(f"sigma_top median = {sigma_top_med:.2f} km/s")
print(f"rho_thin median  = {rho_thin_med:.4f} Msun/pc^3")

# Physical interpretation plot

fig_interp, ax_interp = plt.subplots(figsize=(8, 5))

rho_thin_fixed_for_plot = rho_thin_med
sigma_anchor = sig_obs[-1]

for rho_DM_trial, label, color in [
    (0.005, r"$\rho_\mathrm{DM}=0.005$", "tab:red"),
    (rho_DM_med, r"$\rho_\mathrm{DM}=" + f"{rho_DM_med:.3f}$ inferred", "black"),
    (0.030, r"$\rho_\mathrm{DM}=0.030$", "tab:blue"),
]:
    theta_trial = (
        sigma_anchor,
        rho_thin_fixed_for_plot,
        rho_DM_trial,
    )

    sig_trial = compute_sigma_z_model(
        z_obs,
        nu_obs,
        *theta_trial,
    )

    ax_interp.plot(
        z_obs,
        sig_trial,
        lw=2.0,
        color=color,
        label=label,
    )

ax_interp.errorbar(
    z_obs,
    sig_obs,
    yerr=sig_err,
    fmt="o",
    color="gray",
    ecolor="gray",
    markersize=5,
    capsize=3,
    label=r"Observed $\sigma_z$",
)

ax_interp.set_xlabel(r"$|z|$ [pc]")
ax_interp.set_ylabel(r"$\sigma_z$ [km s$^{-1}$]")
ax_interp.set_title(
    r"Effect of $\rho_\mathrm{DM}$ on $\sigma_z(z)$",
    fontweight="bold",
)
ax_interp.grid(alpha=0.3)
ax_interp.legend()

plt.tight_layout()
plt.savefig("[4]_Physical_Interpretation.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved -> [4]_Physical_Interpretation.png")

# Posterior predictive check

N_PPC = min(200, len(flat_samples))
ppc_idx = np.random.choice(len(flat_samples), size=N_PPC, replace=False)

fig_ppc, ax_ppc = plt.subplots(figsize=(8, 5))

for idx in ppc_idx:
    sig_pred = compute_sigma_z_model(
        z_obs,
        nu_obs,
        *flat_samples[idx],
    )

    ax_ppc.plot(
        z_obs,
        sig_pred,
        color="cornflowerblue",
        alpha=0.05,
        lw=1,
    )

sig_med_model = compute_sigma_z_model(
    z_obs,
    nu_obs,
    *theta_med_full,
)

ax_ppc.plot(
    z_obs,
    sig_med_model,
    color="black",
    lw=2.0,
    label="Posterior median model",
)

ax_ppc.errorbar(
    z_obs,
    sig_obs,
    yerr=sig_err,
    fmt="o",
    color="gray",
    ecolor="gray",
    markersize=5,
    capsize=3,
    label=r"Observed $\sigma_z$",
)

ax_ppc.set_xlabel(r"$|z|$ [pc]")
ax_ppc.set_ylabel(r"$\sigma_z$ [km s$^{-1}$]")
ax_ppc.set_title("Posterior Predictive Check", fontweight="bold")
ax_ppc.grid(alpha=0.3)
ax_ppc.legend()

plt.tight_layout()
plt.savefig("[4]_Posterior_Predictive_Check.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved -> [4]_Posterior_Predictive_Check.png")

# Trace plot

param_labels = [
    r"$\sigma_\mathrm{top}$",
    r"$\rho_\mathrm{thin}$",
    r"$\rho_\mathrm{DM}$",
]

samples_chain = sampler.get_chain()

fig_trace, axes_trace = plt.subplots(N_DIM, 1, figsize=(10, 7), sharex=True)

for i, (ax_t, label) in enumerate(zip(axes_trace, param_labels)):
    ax_t.plot(
        samples_chain[:, :, i],
        color="black",
        alpha=0.25,
        lw=0.5,
    )

    ax_t.axvline(
        N_BURN,
        color="red",
        lw=1.2,
        linestyle="--",
        label="End burn-in" if i == 0 else None,
    )

    ax_t.set_ylabel(label)
    ax_t.grid(alpha=0.25)

axes_trace[0].legend()
axes_trace[-1].set_xlabel("MCMC step")

fig_trace.suptitle("MCMC Chain Traces", fontweight="bold")

plt.tight_layout()
plt.savefig("[4]_MCMC_Traces.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved -> [4]_MCMC_Traces.png")

# Corner plot

fig_corner = corner.corner(
    flat_samples,
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".4f",
    title_kwargs={"fontsize": 10},
)

fig_corner.suptitle(
    "Posterior Corner Plot — Final Reduced Jeans Model",
    fontsize=12,
    fontweight="bold",
    y=1.01,
)

plt.savefig("[4]_MCMC_Corner.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved -> [4]_MCMC_Corner.png")

# rho_DM marginal posterior

fig_dm, ax_dm = plt.subplots(figsize=(7, 4))

ax_dm.hist(
    rho_DM_samples,
    bins=60,
    color="cornflowerblue",
    edgecolor="black",
    linewidth=0.3,
    alpha=0.85,
    density=True,
)

ax_dm.axvline(
    rho_DM_med,
    color="black",
    lw=2.0,
    label=f"Median = {rho_DM_med:.4f}",
)

ax_dm.axvline(
    rho_DM_lo,
    color="black",
    lw=1.2,
    linestyle="--",
    label=f"16th = {rho_DM_lo:.4f}",
)

ax_dm.axvline(
    rho_DM_hi,
    color="black",
    lw=1.2,
    linestyle=":",
    label=f"84th = {rho_DM_hi:.4f}",
)

ax_dm.axvspan(
    rho_DM_lo,
    rho_DM_hi,
    color="gray",
    alpha=0.15,
)

ax_dm.set_xlabel(r"$\rho_\mathrm{DM}$ [M$_\odot$ pc$^{-3}$]")
ax_dm.set_ylabel("Posterior density")
ax_dm.set_title(r"Marginal Posterior of $\rho_\mathrm{DM}$", fontweight="bold")
ax_dm.grid(alpha=0.3)
ax_dm.legend()

plt.tight_layout()
plt.savefig("[4]_rho_DM_Marginal.png", dpi=300, bbox_inches="tight")
plt.show()

print("Saved -> [4]_rho_DM_Marginal.png")

# Summary file

summary_file = "[4]_Run_Summary.txt"

with open(summary_file, "w") as f:

    f.write("MCMC INFERENCE SUMMARY\n")
    f.write("=" * 70 + "\n\n")

    f.write("--Input--\n")
    f.write("Input file: [2]_Gaia_KDwarfs_6D.fits\n")
    f.write(f"Number of stars loaded: {len(kdwarfs):,}\n")
    f.write(f"Number of likelihood bins: {len(z_obs)}\n")
    f.write(f"Fit z range: {np.nanmin(z_obs):.2f} to {np.nanmax(z_obs):.2f} pc\n\n")

    f.write("--Fixed Model Assumptions--\n")
    f.write(f"H_THIN_FIXED: {H_THIN_FIXED:.2f} pc\n")
    f.write(f"RHO_THICK_FIXED: {RHO_THICK_FIXED:.4f} Msun/pc^3\n")
    f.write(f"H_THICK_FIXED: {H_THICK_FIXED:.2f} pc\n")
    f.write(f"RHO_GAS0: {RHO_GAS0:.4f} Msun/pc^3\n")
    f.write(f"H_GAS: {H_GAS:.2f} pc\n\n")

    f.write("--Priors--\n")
    f.write("sigma_top: 5.0 < sigma_top < 45.0 km/s\n")
    f.write("rho_thin: 0.085 < rho_thin < 0.115 Msun/pc^3\n")
    f.write("rho_DM: 0.0 < rho_DM < 0.06 Msun/pc^3\n")
    f.write("Gaussian prior: rho_thin = 0.10 +/- 0.005 Msun/pc^3\n\n")

    f.write("--MCMC Settings--\n")
    f.write(f"N_DIM: {N_DIM}\n")
    f.write(f"N_WALKERS: {N_WALKERS}\n")
    f.write(f"N_STEPS: {N_STEPS}\n")
    f.write(f"N_BURN: {N_BURN}\n")
    f.write(f"THIN: {THIN}\n")
    f.write(f"Flat sample shape: {flat_samples.shape}\n\n")

    f.write("--Parameter Vector--\n")
    f.write("theta = (sigma_top, rho_thin, rho_DM)\n\n")

    f.write("--Posterior Medians--\n")
    f.write(f"sigma_top: {theta_med_full[0]:.6f} km/s\n")
    f.write(f"rho_thin:  {theta_med_full[1]:.6f} Msun/pc^3\n")
    f.write(f"rho_DM:    {theta_med_full[2]:.6f} Msun/pc^3\n")

    f.write("\n--rho_DM Result--\n")
    f.write(
        f"rho_DM = {rho_DM_med:.6f} "
        f"+ {rho_DM_hi-rho_DM_med:.6f} "
        f"- {rho_DM_med-rho_DM_lo:.6f} Msun/pc^3\n"
    )

    f.write("68% credible interval from 16th-84th percentiles.\n\n")

    f.write("--Files Produced--\n")
    f.write("[4]_Physical_Interpretation.png\n")
    f.write("[4]_Posterior_Predictive_Check.png\n")
    f.write("[4]_MCMC_Traces.png\n")
    f.write("[4]_MCMC_Corner.png\n")
    f.write("[4]_rho_DM_Marginal.png\n")
    f.write("[4]_Run_Summary.txt\n")

print(f"\nSaved run summary -> {summary_file}")

print("\nFinal result:")
print(
    f"rho_DM = {rho_DM_med:.4f} + {rho_DM_hi-rho_DM_med:.4f}"
    f" - {rho_DM_med-rho_DM_lo:.4f} Msun/pc^3"
)
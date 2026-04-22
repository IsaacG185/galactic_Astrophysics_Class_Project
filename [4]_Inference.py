# [How rho_DM Is Inferred]
#
# We sample the posterior distribution (equation 17):
#   p(theta | data) ∝ L(theta) * p(theta)
#
# using emcee, an affine-invariant MCMC ensemble sampler.
# The marginal posterior of rho_DM (equation 18) is obtained by
# integrating over all other parameters — in practice, by simply
# reading off the rho_DM column of the flattened chain.

# First, do pip install emcee and pip install corner

import numpy as np
from astropy.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import emcee
import corner

# ── Reload data and re-define forward model (from [2] and [3]) ───────────────

kdwarfs = Table.read("[2]_Gaia_KDwarfs_6D.fits")

z     = np.array(kdwarfs["z_mc_med"], dtype=float)
vz    = np.array(kdwarfs["vz_mc_med"], dtype=float)
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

nu          = np.array(nu)
sigma_z     = np.array(sigma_z)
nu_err      = np.array(nu_err)
sigma_z_err = np.array(sigma_z_err)

valid   = (
    np.isfinite(sigma_z) & np.isfinite(sigma_z_err) &
    np.isfinite(nu) & (nu > 0) & (sigma_z_err > 0)
)
z_obs   = z_centers[valid]
nu_obs  = nu[valid]
sig_obs = sigma_z[valid]
sig_err = sigma_z_err[valid]

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

def log_prior(theta):
    rho_thin, h_thin, rho_thick, h_thick, rho_DM = theta
    if not (0.01  < rho_thin  < 0.5  ): return -np.inf
    if not (100   < h_thin    < 500  ): return -np.inf
    if not (0.001 < rho_thick < 0.1  ): return -np.inf
    if not (500   < h_thick   < 2000 ): return -np.inf
    if not (0.0   < rho_DM    < 1.0  ): return -np.inf
    return 0.0

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta)

# ── MCMC sampling ─────────────────────────────────────────────────────────────
#
# Initial positions: small Gaussian ball around literature-motivated values
#   rho_thin  ~ 0.10 Msun/pc^3  (Bovy & Rix 2013)
#   h_thin    ~ 300 pc
#   rho_thick ~ 0.02 Msun/pc^3
#   h_thick   ~ 900 pc
#   rho_DM    ~ 0.013 Msun/pc^3  (McKee et al. 2015)

N_DIM     = 5
N_WALKERS = 32
N_STEPS   = 2000
N_BURN    = 500

p0_center = np.array([0.10, 300.0, 0.02, 900.0, 0.013])
rng_init  = np.random.default_rng(42)
p0 = p0_center + 1e-3 * p0_center * rng_init.standard_normal((N_WALKERS, N_DIM))

print(f"Running MCMC: {N_WALKERS} walkers x {N_STEPS} steps ...")
sampler = emcee.EnsembleSampler(N_WALKERS, N_DIM, log_posterior)
sampler.run_mcmc(p0, N_STEPS, progress=True)
print("MCMC complete.")

flat_samples = sampler.get_chain(discard=N_BURN, thin=15, flat=True)
print(f"Flat chain shape: {flat_samples.shape}")

# Autocorrelation convergence check
try:
    tau = sampler.get_autocorr_time(discard=N_BURN)
    print(f"\nAutocorrelation times (steps): {np.round(tau, 1)}")
    print(f"Effective samples: {flat_samples.shape[0]}")
except emcee.autocorr.AutocorrError as e:
    print(f"Autocorrelation estimate did not converge: {e}")

# Marginal posterior of rho_DM (equation 18)
rho_DM_samples = flat_samples[:, 4]
rho_DM_med = np.median(rho_DM_samples)
rho_DM_lo  = np.percentile(rho_DM_samples, 16)
rho_DM_hi  = np.percentile(rho_DM_samples, 84)

print(f"\nInferred local dark matter density:")
print(f"  rho_DM = {rho_DM_med:.4f} + {rho_DM_hi - rho_DM_med:.4f}"
      f" - {rho_DM_med - rho_DM_lo:.4f}  [Msun/pc^3]")
print(f"  (16th-84th percentile credible interval)")

# [Physical Interpretation]
#
# Increasing rho_DM increases the total vertical restoring force at
# large |z|, because dark matter contributes a roughly constant density
# that accumulates in the Poisson integral beyond the thin disk scale
# height. This steepens the rise of sigma_z(z) with height relative to
# a purely baryonic model. The observed shape and amplitude of the
# dispersion profile therefore directly constrain rho_DM through the
# likelihood: too little DM -> model sigma_z falls below the data at
# large |z|; too much DM -> model sigma_z overshoots.
#
# We visualize this by plotting sigma_z_model for three values of rho_DM
# (low, best-fit, high) against the data.

fig_interp, ax_interp = plt.subplots(figsize=(8, 5))
fig_interp.patch.set_facecolor("#0d1117")
ax_interp.set_facecolor("#0d1117")

theta_base = list(p0_center)
for rho_DM_trial, label, color in [
    (0.005, r"$\rho_\mathrm{DM}=0.005$", "#FF6B6B"),
    (rho_DM_med, r"$\rho_\mathrm{DM}=" + f"{rho_DM_med:.3f}$" + " (inferred)", "#FFC107"),
    (0.030, r"$\rho_\mathrm{DM}=0.030$", "#4C8BF5"),
]:
    theta_trial = theta_base[:4] + [rho_DM_trial]
    sig_trial   = compute_sigma_z_model(z_obs, nu_obs, *theta_trial)
    ax_interp.plot(z_obs, sig_trial, lw=2.0, color=color, label=label)

ax_interp.errorbar(
    z_obs, sig_obs, yerr=sig_err,
    fmt="o", color="white", markersize=5, elinewidth=1.2,
    capsize=3, label=r"Observed $\sigma_z$", zorder=6,
)
ax_interp.set_xlabel(r"$|z|$  [pc]", fontsize=13, color="white")
ax_interp.set_ylabel(r"$\sigma_z$  [km/s]", fontsize=13, color="white")
ax_interp.set_title(
    r"Physical Interpretation: effect of $\rho_\mathrm{DM}$ on $\sigma_z(z)$",
    fontsize=13, fontweight="bold", color="white",
)
ax_interp.tick_params(colors="white")
for sp in ax_interp.spines.values():
    sp.set_edgecolor("#444")
ax_interp.legend(fontsize=10, facecolor="#1a1f2b", edgecolor="#444", labelcolor="white")
ax_interp.grid(which="major", linestyle="--", linewidth=0.4, color="#333")
ax_interp.xaxis.set_minor_locator(ticker.MultipleLocator(50))
plt.tight_layout()
plt.savefig("[4]_Physical_Interpretation.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Saved -> '[4]_Physical_Interpretation.png'")

# [Systematics and Validation]
#
# We test two systematic variations and quantify how rho_DM shifts:
#
#   Systematic 1: RUWE threshold
#     Default: RUWE < 1.4.  Variation: RUWE < 1.2 (stricter).
#     A stricter cut removes more potentially problematic astrometry,
#     testing whether marginally poor fits bias z and vz.
#
#   Systematic 2: Number of z bins
#     Default: 25 bins over 0-1500 pc.  Variation: 15 bins (coarser).
#     Tests sensitivity to binning choice in the dispersion profile.
#
# For each variation we re-run the full forward model at the posterior
# median theta and refit rho_DM via a simple grid search over rho_DM
# (holding other parameters fixed at their posterior medians), which is
# sufficient to quantify the shift without re-running full MCMC.

theta_med_full = np.median(flat_samples, axis=0)

def quick_rho_DM_fit(z_obs_s, nu_obs_s, sig_obs_s, sig_err_s, theta_med):
    """
    Grid search over rho_DM holding other parameters fixed.
    Returns best-fit rho_DM for the given data subset.
    """
    rho_DM_grid = np.linspace(0.001, 0.10, 200)
    ll_grid = []
    for rho_DM_trial in rho_DM_grid:
        theta_trial = list(theta_med[:4]) + [rho_DM_trial]
        try:
            sig_mod = compute_sigma_z_model(
                z_obs_s, nu_obs_s, *theta_trial
            )
            res = sig_obs_s - sig_mod
            ll_grid.append(-0.5 * np.sum((res / sig_err_s)**2))
        except Exception:
            ll_grid.append(-np.inf)
    ll_grid = np.array(ll_grid)
    return rho_DM_grid[np.argmax(ll_grid)]

# ── Systematic 1: RUWE threshold ─────────────────────────────────────────────
from astropy.table import Table as ATable

clean_ruwe_strict = ATable.read("[1]_Gaia_Ruwe_Clean.fits")

# Re-apply stricter RUWE < 1.2 cut

ruwe_strict_mask = np.array(clean_ruwe_strict["ruwe"], dtype=float) < 1.2
clean_strict     = clean_ruwe_strict[ruwe_strict_mask]
print(f"\nSystematic 1 — strict RUWE < 1.2: {len(clean_strict):,} stars "
      f"(removed {len(clean_ruwe_strict) - len(clean_strict):,})")

# Re-apply K-dwarf selection
plx_s  = np.array(clean_strict["parallax"],       dtype=float)
G_s    = np.array(clean_strict["phot_g_mean_mag"], dtype=float)
BPRP_s = np.array(clean_strict["bp_rp"],           dtype=float)
M_G_s  = G_s + 5.0 * np.log10(plx_s / 1000.0) + 5.0
kd_mask_s = (
    (BPRP_s >= 1.0) & (BPRP_s <= 1.8) &
    (M_G_s  >= 5.0) & (M_G_s  <= 7.5) &
    (plx_s  >  0.0)
)
kd_strict = clean_strict[kd_mask_s]
print(f"K dwarfs after strict RUWE cut: {len(kd_strict):,}")

# Need 6D coordinates — recompute z, vz from scratch for this subsample
from astropy.coordinates import SkyCoord, Galactocentric
import astropy.coordinates as coord
import astropy.units as u

R_sun = 8.122 * u.kpc
z_sun = 20.8  * u.pc
v_sun = coord.CartesianDifferential([11.1, 232.24, 7.25] * u.km / u.s)
galactocentric_frame = Galactocentric(
    galcen_distance=R_sun, z_sun=z_sun, galcen_v_sun=v_sun
)

def compute_z_vz(tbl):
    """Return z [pc] and vz [km/s] arrays for a table with Gaia columns."""
    sc = SkyCoord(
        ra=np.array(tbl["ra"],               dtype=float) * u.deg,
        dec=np.array(tbl["dec"],             dtype=float) * u.deg,
        distance=(1000.0 / np.array(tbl["parallax"], dtype=float)) * u.pc,
        pm_ra_cosdec=np.array(tbl["pmra"],   dtype=float) * u.mas/u.yr,
        pm_dec=np.array(tbl["pmdec"],        dtype=float) * u.mas/u.yr,
        radial_velocity=np.array(tbl["radial_velocity"], dtype=float) * u.km/u.s,
        frame="icrs",
    )
    gc = sc.transform_to(galactocentric_frame)
    return gc.z.to(u.pc).value, gc.v_z.to(u.km/u.s).value

z_s1, vz_s1 = compute_z_vz(kd_strict)
z_abs_s1    = np.abs(z_s1)

def bin_profile(z_abs_arr, vz_arr, z_bins, N_boot=200):
    """Re-bin dispersion profile for any subsample."""
    nu_b, sig_b, nu_e, sig_e = [], [], [], []
    for i in range(len(z_bins) - 1):
        in_bin = (z_abs_arr >= z_bins[i]) & (z_abs_arr < z_bins[i+1])
        vz_bin = vz_arr[in_bin]
        N = len(vz_bin)
        if N < 10:
            nu_b.append(np.nan); sig_b.append(np.nan)
            nu_e.append(np.nan); sig_e.append(np.nan)
            continue
        nu_b.append(N)
        m = np.mean(vz_bin)
        sig_b.append(np.sqrt(np.mean(vz_bin**2) - m**2))
        nb, sb = [], []
        for _ in range(N_boot):
            s = np.random.choice(vz_bin, size=N, replace=True)
            nb.append(len(s)); m2 = np.mean(s)
            sb.append(np.sqrt(np.mean(s**2) - m2**2))
        nu_e.append(np.std(nb)); sig_e.append(np.std(sb))
    return (np.array(nu_b), np.array(sig_b),
            np.array(nu_e), np.array(sig_e))

nu_s1, sig_s1, nue_s1, sige_s1 = bin_profile(z_abs_s1, vz_s1, z_bins)
valid_s1 = (
    np.isfinite(sig_s1) & np.isfinite(sige_s1) &
    np.isfinite(nu_s1) & (nu_s1 > 0) & (sige_s1 > 0)
)
rho_DM_s1 = quick_rho_DM_fit(
    z_centers[valid_s1], nu_s1[valid_s1],
    sig_s1[valid_s1], sige_s1[valid_s1], theta_med_full
)
print(f"\nSystematic 1 result (RUWE < 1.2): rho_DM = {rho_DM_s1:.4f} Msun/pc^3")
print(f"  Shift from baseline: {rho_DM_s1 - rho_DM_med:+.4f} Msun/pc^3")

# ── Systematic 2: Coarser binning ────────────────────────────────────────────
z_bins_coarse    = np.linspace(0, 1500, 15)
z_centers_coarse = 0.5 * (z_bins_coarse[:-1] + z_bins_coarse[1:])

nu_s2, sig_s2, nue_s2, sige_s2 = bin_profile(
    np.abs(z), vz, z_bins_coarse
)
valid_s2 = (
    np.isfinite(sig_s2) & np.isfinite(sige_s2) &
    np.isfinite(nu_s2) & (nu_s2 > 0) & (sige_s2 > 0)
)
rho_DM_s2 = quick_rho_DM_fit(
    z_centers_coarse[valid_s2], nu_s2[valid_s2],
    sig_s2[valid_s2], sige_s2[valid_s2], theta_med_full
)
print(f"\nSystematic 2 result (15 bins): rho_DM = {rho_DM_s2:.4f} Msun/pc^3")
print(f"  Shift from baseline: {rho_DM_s2 - rho_DM_med:+.4f} Msun/pc^3")

# ── Synthetic dataset recovery test ──────────────────────────────────────────
#
# Generate a synthetic sigma_z(z) from a known rho_DM_true, add Gaussian
# noise at the level of sig_err, then run the grid search to verify
# recovery of rho_DM_true.

rho_DM_true    = 0.013   # Msun/pc^3  (injected value)
theta_synth    = list(theta_med_full[:4]) + [rho_DM_true]
sig_synth_true = compute_sigma_z_model(z_obs, nu_obs, *theta_synth)

rng_synth  = np.random.default_rng(7)
sig_synth  = sig_synth_true + rng_synth.normal(0.0, sig_err)

rho_DM_recovered = quick_rho_DM_fit(
    z_obs, nu_obs, sig_synth, sig_err, theta_med_full
)
print(f"\nSynthetic recovery test:")
print(f"  Injected  rho_DM_true     = {rho_DM_true:.4f} Msun/pc^3")
print(f"  Recovered rho_DM          = {rho_DM_recovered:.4f} Msun/pc^3")
print(f"  Residual                  = {rho_DM_recovered - rho_DM_true:+.4f} Msun/pc^3")

# Summary table
print(f"\n{'Variant':<30} {'rho_DM [Msun/pc^3]':>20} {'Shift':>10}")
print("-" * 62)
print(f"{'Baseline (default)':<30} {rho_DM_med:>20.4f} {'—':>10}")
print(f"{'RUWE < 1.2':<30} {rho_DM_s1:>20.4f} {rho_DM_s1-rho_DM_med:>+10.4f}")
print(f"{'15 z-bins':<30} {rho_DM_s2:>20.4f} {rho_DM_s2-rho_DM_med:>+10.4f}")
print(f"{'Synthetic recovery':<30} {rho_DM_recovered:>20.4f} {rho_DM_recovered-rho_DM_true:>+10.4f}")

# [Model Checking]
#
# Four diagnostics are produced:
#   1. Chain trace plot      — visual convergence check per parameter
#   2. Corner plot           — full posterior with correlations
#   3. Posterior predictive  — model realizations vs observed sigma_z
#   4. rho_DM marginal       — 1D histogram with credible interval

param_labels = [
    r"$\rho_\mathrm{thin}$",
    r"$h_\mathrm{thin}$",
    r"$\rho_\mathrm{thick}$",
    r"$h_\mathrm{thick}$",
    r"$\rho_\mathrm{DM}$",
]

# ── 1. Chain traces ───────────────────────────────────────────────────────────
samples_chain = sampler.get_chain()   # (N_STEPS, N_WALKERS, N_DIM)

fig_trace, axes_trace = plt.subplots(N_DIM, 1, figsize=(10, 9), sharex=True)
fig_trace.patch.set_facecolor("#0d1117")
for i, (ax_t, label) in enumerate(zip(axes_trace, param_labels)):
    ax_t.set_facecolor("#0d1117")
    ax_t.plot(samples_chain[:, :, i], color="#4C8BF5", alpha=0.3, lw=0.5)
    ax_t.axvline(N_BURN, color="#FF4C4C", lw=1.2, linestyle="--",
                 label="End of burn-in" if i == 0 else "")
    ax_t.set_ylabel(label, color="white", fontsize=10)
    ax_t.tick_params(colors="white")
    for sp in ax_t.spines.values():
        sp.set_edgecolor("#444")
axes_trace[0].legend(fontsize=9, facecolor="#1a1f2b",
                     edgecolor="#444", labelcolor="white")
axes_trace[-1].set_xlabel("MCMC step", fontsize=12, color="white")
fig_trace.suptitle("MCMC Chain Traces — Convergence Diagnostic",
                   fontsize=13, fontweight="bold", color="white")
plt.tight_layout()
plt.savefig("[4]_MCMC_Traces.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Saved -> '[4]_MCMC_Traces.png'")

# ── 2. Corner plot ────────────────────────────────────────────────────────────
fig_corner = corner.corner(
    flat_samples,
    labels=param_labels,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_fmt=".4f",
    title_kwargs={"fontsize": 10},
    color="#4C8BF5",
)
fig_corner.suptitle(
    r"Posterior corner plot — Jeans model parameters",
    fontsize=12, fontweight="bold", y=1.01,
)
plt.savefig("[4]_MCMC_Corner.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved -> '[4]_MCMC_Corner.png'")

# ── 3. Posterior predictive check ────────────────────────────────────────────
N_PPC     = 200
ppc_idx   = np.random.choice(len(flat_samples), size=N_PPC, replace=False)

fig_ppc, ax_ppc = plt.subplots(figsize=(8, 5))
fig_ppc.patch.set_facecolor("#0d1117")
ax_ppc.set_facecolor("#0d1117")

for idx in ppc_idx:
    sig_pred = compute_sigma_z_model(z_obs, nu_obs, *flat_samples[idx])
    ax_ppc.plot(z_obs, sig_pred, color="#4C8BF5", alpha=0.05, lw=1)

sig_med_model = compute_sigma_z_model(z_obs, nu_obs, *theta_med_full)
ax_ppc.plot(z_obs, sig_med_model, color="#FFC107", lw=2.0,
            label="Posterior median model", zorder=5)
ax_ppc.errorbar(
    z_obs, sig_obs, yerr=sig_err,
    fmt="o", color="white", markersize=5, elinewidth=1.2,
    capsize=3, label=r"Observed $\sigma_z$", zorder=6,
)
ax_ppc.set_xlabel(r"$|z|$  [pc]", fontsize=13, color="white")
ax_ppc.set_ylabel(r"$\sigma_z$  [km/s]", fontsize=13, color="white")
ax_ppc.set_title("Posterior Predictive Check — Vertical Velocity Dispersion",
                 fontsize=13, fontweight="bold", color="white")
ax_ppc.tick_params(colors="white")
for sp in ax_ppc.spines.values():
    sp.set_edgecolor("#444")
ax_ppc.legend(fontsize=10, facecolor="#1a1f2b",
              edgecolor="#444", labelcolor="white")
ax_ppc.grid(which="major", linestyle="--", linewidth=0.4, color="#333")
ax_ppc.xaxis.set_minor_locator(ticker.MultipleLocator(50))
plt.tight_layout()
plt.savefig("[4]_Posterior_Predictive_Check.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Saved -> '[4]_Posterior_Predictive_Check.png'")

# ── 4. rho_DM marginal posterior ─────────────────────────────────────────────
fig_dm, ax_dm = plt.subplots(figsize=(7, 4))
fig_dm.patch.set_facecolor("#0d1117")
ax_dm.set_facecolor("#0d1117")

ax_dm.hist(rho_DM_samples, bins=60, color="#4C8BF5",
           edgecolor="none", alpha=0.85, density=True)
ax_dm.axvline(rho_DM_med, color="#FFC107", lw=2.0, linestyle="-",
              label=f"Median = {rho_DM_med:.4f}")
ax_dm.axvline(rho_DM_lo,  color="#FFC107", lw=1.2, linestyle="--",
              label=f"16th pct = {rho_DM_lo:.4f}")
ax_dm.axvline(rho_DM_hi,  color="#FFC107", lw=1.2, linestyle=":",
              label=f"84th pct = {rho_DM_hi:.4f}")
ax_dm.axvspan(rho_DM_lo, rho_DM_hi, color="#FFC107", alpha=0.10)
ax_dm.set_xlabel(r"$\rho_\mathrm{DM}$  [M$_\odot$ pc$^{-3}$]",
                 fontsize=13, color="white")
ax_dm.set_ylabel("Posterior density", fontsize=12, color="white")
ax_dm.set_title(r"Marginal posterior of $\rho_\mathrm{DM}$",
                fontsize=13, fontweight="bold", color="white")
ax_dm.tick_params(colors="white")
for sp in ax_dm.spines.values():
    sp.set_edgecolor("#444")
ax_dm.legend(fontsize=10, facecolor="#1a1f2b",
             edgecolor="#444", labelcolor="white")
ax_dm.grid(which="major", linestyle="--", linewidth=0.4, color="#333")
plt.tight_layout()
plt.savefig("[4]_rho_DM_Marginal.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Saved -> '[4]_rho_DM_Marginal.png'")

print(f"\nFinal result:")
print(f"  rho_DM = {rho_DM_med:.4f} + {rho_DM_hi-rho_DM_med:.4f}"
      f" - {rho_DM_med-rho_DM_lo:.4f}  Msun/pc^3  (68% credible interval)")
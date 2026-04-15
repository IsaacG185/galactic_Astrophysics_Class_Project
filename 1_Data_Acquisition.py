# [Gaia: The Survey and Its Operation, Physical Framework (no work for this part), Data Acquisition]

# Installed VSCodium from https://vscodium.com/,
# Then install python through VSCode extensions (Ctrl+Shift+X)
# Then ran VSCodium terminal (Ctrl+Shift+'), then ran
# pip install numpy scipy matplotlib astropy astroquery
# Then installed TOPCAT via https://www.star.bris.ac.uk/~mbt/topcat/ (to look at fits files)

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric
from astropy.table import Table
from astroquery.gaia import Gaia

# Query description
gaia_Query = """
SELECT TOP 1000000
    source_id,
    ra,
    ra_error,
    dec,
    dec_error,
    parallax,
    parallax_error,
    pmra,
    pmra_error,
    pmdec,
    pmdec_error,
    radial_velocity,
    radial_velocity_error,
    ruwe,
    phot_g_mean_mag,
    phot_bp_mean_mag,
    phot_rp_mean_mag,
    bp_rp,
    teff_gspphot
FROM
    gaiadr3.gaia_source
WHERE
    radial_velocity IS NOT NULL
    AND parallax_over_error > 5
    AND parallax > 0.667
"""

# Query to Gaia using astroquery.gaia
job = Gaia.launch_job_async(
    query=gaia_Query,
    name="local_Stars", # Optional
    verbose=True # optional
)

# Retrieve results as an astropy Table
result = job.get_results()
print(f"Retrieved {len(result):,} sources")
print(result.colnames)
print(result[:5])

# Save to FITS and read back
result.write("[1]_Gaia_Data.fits", format="fits")
r = Table.read("[1]_Gaia_Data.fits")
print(r)

# Data Cleaning, mask any rows with NaN in critical columns 
cols_required = [
    "parallax", "parallax_error",
   "pmra", "pmra_error",
   "pmdec", "pmdec_error",
   "radial_velocity", "radial_velocity_error",
 ]

mask = np.ones(len(result), dtype=bool)
for col in cols_required:
    mask &= np.isfinite(result[col])
clean = result[mask]
print(f"After NaN mask: {len(clean):,} sources remain")

# There are no NaN, but this sanity check is left in just in case.

# [Astrometric Quality and RUWE]

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ruwe_vals = clean["ruwe"].data.astype(float)

fig, ax = plt.subplots(figsize=(9, 5))

# Histogram - log-y so the rare high-RUWE tail is visible
counts, edges, patches = ax.hist(
    ruwe_vals,
    bins=200,
    range=(0, 5),
    color="#4C8BF5",
    edgecolor="none",
    alpha=0.85,
    log=True,
    label="All stars",
)

# Shade the rejected region
ax.axvspan(1.4, 5.0, color="#FF4C4C", alpha=0.12, label="Rejected (RUWE ≥ 1.4)")

# Threshold line
ax.axvline(1.4, color="#FF4C4C", linewidth=1.8, linestyle="--", label="RUWE = 1.4 threshold")

# Ideal-fit reference
ax.axvline(1.0, color="#FFC107", linewidth=1.2, linestyle=":", label="Ideal RUWE = 1.0")

ax.set_xlabel("RUWE", fontsize=13)
ax.set_ylabel("Number of stars (log scale)", fontsize=13)
ax.set_title("RUWE Distribution - Gaia DR3 Local Sample", fontsize=14, fontweight="bold")
ax.set_xlim(0, 5)
ax.set_ylim(bottom=1)
ax.legend(fontsize=10)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plot_path = "[1]_Ruwe_Distribution.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved to '{plot_path}'")

# Saves new .fits file using the RUWE < 1.4 threshold criteria
RUWE_THRESHOLD = 1.4

ruwe_mask = clean["ruwe"] < RUWE_THRESHOLD
clean_ruwe = clean[ruwe_mask]

n_before = len(clean)
n_after  = len(clean_ruwe)
n_removed = n_before - n_after

print(f"\nRUWE cut summary")
print(f"  Before : {n_before:>7,} stars")
print(f"  Removed: {n_removed:>7,} stars  ({100 * n_removed / n_before:.2f} %)")
print(f"  After  : {n_after:>7,} stars  (RUWE < {RUWE_THRESHOLD})")

clean_ruwe.write("[1]_Gaia_Ruwe_Clean.fits", format="fits", overwrite=True)
print(f"\nSaved astrometrically clean sample → '[1]_Gaia_Ruwe_Clean.fits'")

# Gaia's astrometric pipeline fits a 5-parameter model (ra, dec,
# parallax, pmra, pmdec) to each star's along-scan observations.
# RUWE = sqrt(chi2 / (N_obs - 5)) * normalization_factor, where
# the normalization removes known dependencies on G magnitude and
# color. A well-behaved single star yields RUWE ≈ 1.
#
# WHY POOR FITS BIAS PARALLAX AND PROPER MOTION (and therefore z, vz):
#
#   1. Unresolved binaries: The photocentre wobbles at the binary
#      orbital period, adding a spurious signal that Gaia's model
#      absorbs partly into parallax and pmra/pmdec. This can inflate
#      OR deflate the inferred parallax, biasing the distance estimate
#      and therefore the computed Galactic height z = d * sin(b).
#
#   2. Crowding / nearby contaminant: Blended flux shifts the
#      photocentre in a direction unrelated to true parallactic
#      motion, introducing correlated noise into all five parameters.
#
#   3. Extended or non-point-like sources: Galaxies or nebulae give
#      a poor PSF fit; their "parallax" is meaningless.
#
#   In all cases the biased parallax propagates directly into the
#   distance d = 1/plx, and hence z = d * sin(b).  The biased proper
#   motions propagate into the transverse velocity components, so vz
#   (which mixes pmra, pmdec, and radial_velocity through the
#   coordinate rotation into Galactocentric frame) is also corrupted.
#   Keeping RUWE < 1.4 (the standard Gaia collaboration threshold)
#   retains stars whose astrometric model residuals are consistent
#   with Poisson noise, giving reliable 6-D phase-space coordinates.

# [Tracer Selection]

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

# Load cleaned data
clean_ruwe = Table.read("[1]_Gaia_Ruwe_Clean.fits")

# Compute absolute G magnitude using parallax in mas
# M_G = G + 5*log10(parallax / 1000) + 5
plx = np.array(clean_ruwe["parallax"].data, dtype=float)   # mas
G   = np.array(clean_ruwe["phot_g_mean_mag"].data, dtype=float)
BP_RP = np.array(clean_ruwe["bp_rp"].data, dtype=float)

M_G = G + 5.0 * np.log10(plx / 1000.0) + 5.0

# K-dwarf selection box
BPRP_MIN, BPRP_MAX = 1.0, 1.8
MG_MIN,   MG_MAX   = 5.0, 7.5

kdwarf_mask = (
    (BP_RP >= BPRP_MIN) & (BP_RP <= BPRP_MAX) &
    (M_G   >= MG_MIN)   & (M_G   <= MG_MAX)
)
n_kdwarf = kdwarf_mask.sum()
print(f"K-dwarf candidates: {n_kdwarf:,}")

# Save K-dwarf subsample as fits file, gaia_Kdwarfs.fits
kdwarfs = clean_ruwe[kdwarf_mask]
kdwarfs["M_G"] = M_G[kdwarf_mask]
kdwarfs.write("[1]_Gaia_Kdwarfs.fits", format="fits", overwrite=True)
print("Saved K-dwarf sample -> '[1]_Gaia_Kdwarfs.fits'")

# Color-Magnitude Diagram
fig, ax = plt.subplots(figsize=(8, 9))
fig.patch.set_facecolor("#0d1117")
ax.set_facecolor("#0d1117")

# All stars as 2D density histogram
h = ax.hist2d(
    BP_RP, M_G,
    bins=[300, 350],
    range=[[-0.6, 4.0], [-3, 17]],
    cmap="YlOrRd",
    norm=LogNorm(vmin=1, vmax=500),
    zorder=1,
)

# Highlight K-dwarf candidates
ax.scatter(
    BP_RP[kdwarf_mask], M_G[kdwarf_mask],
    s=0.8, c="#00CFFF", alpha=0.25, linewidths=0, zorder=2,
    label=f"K-dwarf candidates  ({n_kdwarf:,})",
)

# Selection box
rect = mpatches.FancyBboxPatch(
    (BPRP_MIN, MG_MIN),
    BPRP_MAX - BPRP_MIN,
    MG_MAX   - MG_MIN,
    boxstyle="square,pad=0",
    linewidth=2.2, edgecolor="#00CFFF",
    facecolor="none", linestyle="--",
    zorder=5, label="K-dwarf selection box",
)
ax.add_patch(rect)

# Annotation
ax.annotate(
    "K dwarfs\n"
    r"$1.0 \leq G_{BP}-G_{RP} \leq 1.8$" + "\n"
    r"$5.0 \leq M_G \leq 7.5$",
    xy=(1.40, 6.25), xytext=(2.35, 4.2),
    fontsize=9.5, color="#00CFFF",
    arrowprops=dict(arrowstyle="->", color="#00CFFF", lw=1.4),
    bbox=dict(boxstyle="round,pad=0.3", fc="#0d1117", ec="#00CFFF", lw=1.0, alpha=0.85),
    zorder=6,
)

# Colorbar
cbar = fig.colorbar(h[3],ax=ax,pad=0.02,fraction=0.035)
cbar.set_label("Stars per bin (log scale)",fontsize=10,color="white")
cbar.ax.yaxis.set_tick_params(color="white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

# Labels and styling
ax.set_xlabel(r"$G_{BP} - G_{RP}$  [mag]",fontsize=13,color="white")
ax.set_ylabel(r"$M_G$  [mag]",fontsize=13,color="white")
ax.set_title(
    "Color-Magnitude Diagram - Gaia DR3 Local Sample\nTracer Selection: K Dwarfs",
    fontsize=13, fontweight="bold", color="white", pad=10,
)
ax.invert_yaxis()
ax.set_xlim(-0.6, 4.0)
ax.set_ylim(17, -3)
ax.tick_params(colors="white", which="both")
for spine in ax.spines.values():
    spine.set_edgecolor("#444")
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
ax.grid(which="major", linestyle="--", linewidth=0.4, color="#333", zorder=0)
ax.legend(fontsize=9, facecolor="#1a1f2b", edgecolor="#444", labelcolor="white", loc="upper left")

plt.tight_layout()
plt.savefig("[1]_Tracer_Selection_CMD.png", dpi=180, bbox_inches="tight", facecolor="#0d1117")
plt.show()
print("Plot saved -> '[1]_Tracer_Selection_CMD.png'")

# Need to write justification separately for the report.
# K dwarfs chosen because:\n,  Long-lived (tau >> disk age) -> sample full disk history\n, Numerous -> good statistics per z-bin\n
# Thin-disk confined -> trace disk dynamics well\n
# Avoid subgiants (bluer/brighter) & M-dwarfs\n
# (complex atmosphere, poorer RV)\n
# RUWE-clean parallaxes -> reliable d = 1/plx
# For next push, ensuring everything updates properly.
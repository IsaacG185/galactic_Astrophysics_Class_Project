# Gaia: The Survey and Its Operation, Physical Framework (no work for this part), Data Acquisition

# Installed VSCodium from https://vscodium.com/,
# Then install python through VSCode extensions (Ctrl+Shift+X)
# Then ran VSCodium terminal (Ctrl+Shift+'), then ran
# pip install numpy scipy matplotlib astropy astroquery
# Then installed TOPCAT via https://www.star.bris.ac.uk/~mbt/topcat/

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric
from astropy.table import Table
from astroquery.gaia import Gaia

# ------------------------------------------------------------------
# ADQL Query
# ------------------------------------------------------------------
ADQL_QUERY = """
SELECT TOP 100000
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
    bp_rp
FROM
    gaiadr3.gaia_source
WHERE
    radial_velocity IS NOT NULL
    AND parallax_over_error > 5
    AND parallax > 0.667
"""

# ------------------------------------------------------------------
# Submit the query to the Gaia archive
# ------------------------------------------------------------------
job = Gaia.launch_job_async(
    query=ADQL_QUERY,
    name="local_stars", # Optional
    verbose=True # optional
)

# Retrieve results as an astropy Table
result = job.get_results()
print(f"Retrieved {len(result):,} sources")
print(result.colnames)
print(result[:5])

# Save to FITS and read back
result.write("gaia_Data.fits", format="fits")
r = Table.read("gaia_Data.fits")
print(r)

# ------------------------------------------------------------------
# Basic data hygiene after retrieval
# ------------------------------------------------------------------
# Mask any rows with NaN in critical columns (paranoia check —
# the WHERE clause should handle most of this)
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

# Astrometric Quality and RUWE
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

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ruwe_vals = clean["ruwe"].data.astype(float)

fig, ax = plt.subplots(figsize=(9, 5))

# Histogram — log-y so the rare high-RUWE tail is visible
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
ax.set_title("RUWE Distribution — Gaia DR3 Local Sample", fontsize=14, fontweight="bold")
ax.set_xlim(0, 5)
ax.set_ylim(bottom=1)
ax.legend(fontsize=10)
ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(axis="y", which="major", linestyle="--", linewidth=0.5, alpha=0.4)

plt.tight_layout()
plot_path = "ruwe_Distribution.png"
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"Plot saved to '{plot_path}'")

# ------------------------------------------------------------------
# Apply the RUWE < 1.4 cut and save a new FITS file
# ------------------------------------------------------------------
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

clean_ruwe.write("gaia_Ruwe_Clean.fits", format="fits", overwrite=True)
print(f"\nSaved astrometrically clean sample → 'gaia_Ruwe_Clean.fits'")
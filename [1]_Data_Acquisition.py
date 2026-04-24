# [Gaia: The Survey and Its Operation, Physical Framework (no work for this part), Data Acquisition]

# Installed VSCodium from https://vscodium.com/
# Then install python through VSCode extensions (Ctrl+Shift+X)
# Then ran VSCodium terminal (Ctrl+Shift+'), then ran
# pip install numpy scipy matplotlib astropy astroquery
# Then installed TOPCAT via https://www.star.bris.ac.uk/~mbt/topcat/ (to look at fits files)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

from matplotlib.colors import LogNorm
from astropy.table import Table
from astroquery.gaia import Gaia

# Global plotting style for publication-quality bright figures

plt.rcParams.update({
    "figure.facecolor":"white",
    "axes.facecolor":"white",
    "savefig.facecolor":"white",
    "font.size":11,
    "axes.labelsize":13,
    "axes.titlesize":14,
    "legend.framealpha":1,
})

# Query description
# random_index gives a random representative sample rather than sky-order bias.
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
    pmra_pmdec_corr,
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
    AND radial_velocity_error < 5
    AND parallax_over_error > 10
    AND parallax > 0.667
ORDER BY random_index
"""

# Query to Gaia using astroquery.gaia
job = Gaia.launch_job_async(
    query=gaia_Query,
    name="local_Stars",
    verbose=True
)

# Retrieve results as an astropy Table
result = job.get_results()
print(f"Retrieved {len(result):,} sources")
print(result.colnames)
print(result[:5])

# Save to FITS and read back
result.write("[1]_Gaia_Data.fits", format="fits", overwrite=True)
r = Table.read("[1]_Gaia_Data.fits")
print(r)

# Data Cleaning
cols_required = [
    "parallax","parallax_error",
    "pmra","pmra_error",
    "pmdec","pmdec_error",
    "radial_velocity","radial_velocity_error",
    "ruwe"
]

mask = np.ones(len(result),dtype=bool)
for col in cols_required:
    mask &= np.isfinite(result[col])

clean = result[mask]
print(f"After NaN mask: {len(clean):,} sources remain")

# There are no NaN, but this sanity check is left in just in case.

# [Astrometric Quality and RUWE]

ruwe_vals = clean["ruwe"].data.astype(float)

fig, ax = plt.subplots(figsize=(9,5))

# Linear-scale histogram (NOT log)
counts, edges, patches = ax.hist(
    ruwe_vals,
    bins=180,
    range=(0.7,3.0),
    color="cornflowerblue",
    edgecolor="white",
    linewidth=0.35,
    alpha=0.9,
    label="All stars"
)

# Shade rejected region
ax.axvspan(
    1.4,
    3.0,
    color="salmon",
    alpha=0.18,
    label="Rejected (RUWE ≥ 1.4)"
)

# Threshold line
ax.axvline(
    1.4,
    color="red",
    linestyle="--",
    linewidth=2,
    label="RUWE threshold"
)

# Ideal single-star reference
ax.axvline(
    1.0,
    color="goldenrod",
    linestyle=":",
    linewidth=1.7,
    label="Ideal RUWE = 1"
)

ax.set_xlim(.7,3)
ax.set_xlabel("RUWE")
ax.set_ylabel("Number of stars")
ax.set_title("RUWE Distribution — Gaia DR3 Local Sample",fontweight="bold")

ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax.grid(axis="y",linestyle="--",alpha=.35)
ax.legend()

plt.tight_layout()
plot_path="[1]_Ruwe_Distribution.png"
plt.savefig(plot_path,dpi=300)
plt.show()
print(f"Plot saved to '{plot_path}'")

# Saves new .fits file using the RUWE < 1.4 threshold criteria
RUWE_THRESHOLD=1.4

ruwe_mask = clean["ruwe"] < RUWE_THRESHOLD
clean_ruwe = clean[ruwe_mask]

n_before=len(clean)
n_after=len(clean_ruwe)
n_removed=n_before-n_after

print("\nRUWE cut summary")
print(f" Before : {n_before:,}")
print(f" Removed: {n_removed:,} ({100*n_removed/n_before:.2f}%)")
print(f" After  : {n_after:,}")

clean_ruwe.write(
    "[1]_Gaia_Ruwe_Clean.fits",
    format="fits",
    overwrite=True
)

print("Saved astrometrically clean sample")


# [Tracer Selection]

clean_ruwe = Table.read("[1]_Gaia_Ruwe_Clean.fits")

# Compute absolute G magnitude using parallax in mas
# M_G = G + 5log10(parallax/1000)+5
plx=np.array(clean_ruwe["parallax"],dtype=float)
G=np.array(clean_ruwe["phot_g_mean_mag"],dtype=float)
BP_RP=np.array(clean_ruwe["bp_rp"],dtype=float)

M_G = G + 5*np.log10(plx/1000.0)+5

# K-dwarf selection box (tightened)
BPRP_MIN,BPRP_MAX = 1.1,1.6
MG_MIN,MG_MAX = 5.5,7.3

kdwarf_mask=(
 (BP_RP>=BPRP_MIN)&(BP_RP<=BPRP_MAX)&
 (M_G>=MG_MIN)&(M_G<=MG_MAX)
)

n_kdwarf=kdwarf_mask.sum()
print(f"K-dwarf candidates: {n_kdwarf:,}")

kdwarfs=clean_ruwe[kdwarf_mask]
kdwarfs["M_G"] = M_G[kdwarf_mask]
kdwarfs.write(
    "[1]_Gaia_KDwarfs.fits",
    format="fits",
    overwrite=True
)

print("Saved K-dwarf sample")

# Color-Magnitude Diagram (light paper style)
fig,ax=plt.subplots(figsize=(8,9))

h=ax.hist2d(
    BP_RP,
    M_G,
    bins=[300,350],
    range=[[-0.6,4],[-3,17]],
    cmap="YlOrRd",
    norm=LogNorm(vmin=1,vmax=500)
)

# Highlight selected tracers
ax.scatter(
    BP_RP[kdwarf_mask],
    M_G[kdwarf_mask],
    s=.8,
    c="deepskyblue",
    alpha=.25,
    linewidths=0,
    label=f"K dwarfs ({n_kdwarf:,})"
)

rect=mpatches.Rectangle(
    (BPRP_MIN,MG_MIN),
    BPRP_MAX-BPRP_MIN,
    MG_MAX-MG_MIN,
    fill=False,
    edgecolor="blue",
    linestyle="--",
    linewidth=2,
    label="Selection box"
)
ax.add_patch(rect)

ax.annotate(
    "K dwarfs\n1.1 ≤ BP-RP ≤ 1.6\n5.5 ≤ M_G ≤ 7.3",
    xy=(1.35,6.2),
    xytext=(2.25,4.5),
    arrowprops=dict(arrowstyle='->'),
    bbox=dict(boxstyle='round',fc='white')
)

cbar=fig.colorbar(h[3],ax=ax,pad=.02)
cbar.set_label("Stars per bin (log scale)")

ax.set_xlabel(r"$G_{BP}-G_{RP}$ [mag]")
ax.set_ylabel(r"$M_G$ [mag]")
ax.set_title(
"Color-Magnitude Diagram - Gaia DR3\nTracer Selection: K Dwarfs",
fontweight='bold'
)

ax.invert_yaxis()
ax.set_xlim(-.6,4)
ax.set_ylim(17,-3)

ax.xaxis.set_minor_locator(ticker.MultipleLocator(.1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(.5))
ax.grid(ls='--',alpha=.25)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(
    "[1]_Tracer_Selection_CMD.png",
    dpi=300,
    bbox_inches='tight'
)
plt.show()

print("Plot saved -> [1]_Tracer_Selection_CMD.png")

# Summary file generated at end so Gaia query need not rerun
summary_file="[1]_Run_Summary.txt"

frac_ruwe_kept=n_after/n_before
frac_kdwarf_full=n_kdwarf/n_after
frac_kdwarf_orig=n_kdwarf/len(result)

ruwe_median=np.median(clean['ruwe'])
ruwe_95=np.percentile(clean['ruwe'],95)

parallax_med=np.median(clean_ruwe['parallax'])
parallax_min=np.min(clean_ruwe['parallax'])
parallax_max=np.max(clean_ruwe['parallax'])

distance_med_pc=1000/parallax_med

bp_rp_med=np.median(BP_RP[kdwarf_mask])
Mg_med=np.median(M_G[kdwarf_mask])

with open(summary_file,'w') as f:

    f.write('GAIA DR3 LOCAL SAMPLE SUMMARY\n')
    f.write('='*60+'\n\n')

    f.write('--Query--\n')
    f.write(f'Initial stars queried: {len(result):,}\n')
    f.write('Selection:\n')
    f.write(' radial_velocity IS NOT NULL\n')
    f.write(' radial_velocity_error < 5 km/s\n')
    f.write(' parallax_over_error > 10\n')
    f.write(' parallax > 0.667 mas\n')
    f.write(' ORDER BY random_index\n\n')

    f.write('--RUWE Cleaning--\n')
    f.write(f'RUWE threshold: {RUWE_THRESHOLD}\n')
    f.write(f'Before cut: {n_before:,}\n')
    f.write(f'After cut : {n_after:,}\n')
    f.write(f'Removed   : {n_removed:,}\n')
    f.write(f'Fraction retained: {frac_ruwe_kept:.4f}\n')
    f.write(f'Median RUWE: {ruwe_median:.4f}\n')
    f.write(f'95th percentile RUWE: {ruwe_95:.4f}\n\n')

    f.write('--Distance Diagnostics--\n')
    f.write(f'Median parallax: {parallax_med:.4f} mas\n')
    f.write(f'Min parallax   : {parallax_min:.4f}\n')
    f.write(f'Max parallax   : {parallax_max:.4f}\n')
    f.write(f'Median distance: {distance_med_pc:.1f} pc\n\n')

    f.write('--K-Dwarf Tracers--\n')
    f.write(f'Count: {n_kdwarf:,}\n')
    f.write(f'RUWE-clean fraction: {frac_kdwarf_full:.4f}\n')
    f.write(f'Original fraction  : {frac_kdwarf_orig:.4f}\n')
    f.write(f'Median BP-RP: {bp_rp_med:.4f}\n')
    f.write(f'Median M_G : {Mg_med:.4f}\n\n')

    f.write('--Files Produced--\n')
    f.write('[1]_Gaia_Data.fits\n')
    f.write('[1]_Gaia_Ruwe_Clean.fits\n')
    f.write('[1]_Gaia_KDwarfs.fits\n')
    f.write('[1]_Ruwe_Distribution.png\n')
    f.write('[1]_Tracer_Selection_CMD.png\n')

print(f"Saved run summary -> {summary_file}")
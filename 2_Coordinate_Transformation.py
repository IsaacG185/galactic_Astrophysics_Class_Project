# ------------------------------------------------------------------
# Coordinate transformation: ICRS → Galactocentric 6D phase space
# ------------------------------------------------------------------
# Distance in parsecs from the naive parallax inversion.
# For science results, replace with Bayesian distances.
distance_pc = 1000.0 / clean["parallax"]   # parallax in mas → dist in pc

coords = SkyCoord(
    ra=clean["ra"] * u.deg,
    dec=clean["dec"] * u.deg,
    distance=distance_pc * u.pc,
    pm_ra_cosdec=clean["pmra"] * u.mas / u.yr,
    pm_dec=clean["pmdec"] * u.mas / u.yr,
    radial_velocity=clean["radial_velocity"] * u.km / u.s,
    frame="icrs",
)

# Galactocentric frame using the Astropy default parameters:
#   R_sun  = 8.122 kpc  (Gravity Collaboration 2018)
#   z_sun  = 20.8 pc    (Bennett & Bovy 2019)
#   v_circ = 229 km/s   (Eilers et al. 2019, via astropy defaults)
# These can be customised via Galactocentric(galcen_distance=..., etc.)
galcen = coords.galactocentric

X_kpc = galcen.x.to(u.kpc).value
Y_kpc = galcen.y.to(u.kpc).value
Z_kpc = galcen.z.to(u.kpc).value
vx    = galcen.v_x.to(u.km / u.s).value
vy    = galcen.v_y.to(u.km / u.s).value
vz    = galcen.v_z.to(u.km / u.s).value

print("\nSample 6D phase-space output (first 5 rows):")
for i in range(min(5, len(clean))):
    print(
        f"  source {clean['source_id'][i]:20d}  "
        f"X={X_kpc[i]:+7.3f} Y={Y_kpc[i]:+7.3f} Z={Z_kpc[i]:+7.3f} kpc  "
        f"vx={vx[i]:+7.1f} vy={vy[i]:+7.1f} vz={vz[i]:+7.1f} km/s"
    )


# ------------------------------------------------------------------
# Quick sanity-check plots (Toomre diagram + sky distribution)
# ------------------------------------------------------------------
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Sky map (equatorial) ---
ax = axes[0]
ax.scatter(
    clean["ra"], clean["dec"],
    s=1, alpha=0.3, c=clean["phot_g_mean_mag"],
    cmap="viridis_r", rasterized=True
)
ax.set_xlabel("RA (deg)")
ax.set_ylabel("Dec (deg)")
ax.set_title("Sky distribution — Gaia DR3 RVS sample (d < 1.5 kpc)")
ax.invert_xaxis()

# --- Toomre diagram: sqrt(vx^2 + vz^2) vs vy ---
ax = axes[1]
v_perp = np.sqrt(vx**2 + vz**2)

# LSR reference: vy ≈ 0, v_perp ≈ 0  (disk population clumps here)
ax.scatter(vy, v_perp, s=1, alpha=0.3, rasterized=True)
ax.axhline(0, color="gray", lw=0.5, ls="--")
ax.axvline(0, color="gray", lw=0.5, ls="--")

# Add velocity circle guides (50, 100, 150, 210 km/s)
theta = np.linspace(0, 2 * np.pi, 500)
for r in [50, 100, 150, 210]:
    ax.plot(r * np.cos(theta), r * np.sin(theta),
            color="gray", lw=0.4, ls=":")

ax.set_xlabel(r"$v_y$ (km s$^{-1}$) — azimuthal velocity")
ax.set_ylabel(r"$\sqrt{v_x^2 + v_z^2}$ (km s$^{-1}$)")
ax.set_title("Toomre diagram")
ax.set_xlim(-350, 350)
ax.set_ylim(0, 350)

plt.tight_layout()
plt.savefig("gaia_dr3_rvs_sanity_check.pdf", dpi=150, bbox_inches="tight")
plt.show()
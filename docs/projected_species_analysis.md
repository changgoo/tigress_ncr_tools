# H I and emission-measure projected statistics

This analysis applies the existing face-on one-point PDF and shear-aware power-
spectrum pipelines to atomic hydrogen and emission measure. It reads only
`proj2d/theta0`; it does not reconstruct either quantity from VTK snapshots.
The estimator and diagnostics are identical to the total-gas analysis, so the
three tracers can be compared without changing the numerical method.

## 1. Projected fields

| quantity key | `proj2d` field | map represented | output stem |
|---|---|---|---|
| `gas` | `nH` | total hydrogen-nucleus column, \(\Sigma\) | `density` |
| `hi` | `nHI` | atomic-hydrogen column, \(\Sigma_{\rm HI}\) | `hi` |
| `em` | `ne_sq` | emission measure, \({\rm EM}=\int n_e^2\,d\ell\) | `em` |

The native `nH` and `nHI` line integrals acquire units of H nuclei cm\(^{-2}\)
after multiplying the stored pc path length by one pc in cm. `ne_sq` is stored
as pc cm\(^{-6}\). These constant conversions cancel from both normalized
variables,

\[
\delta_Q=\frac{Q}{\langle Q\rangle_A}-1,
\qquad
s_Q=\ln\left(\frac{Q}{\langle Q\rangle_A}\right),
\]

and therefore are not applied before the PDF or FFT. The native mean remains in
the archives as `mean_sigma_code`; `field`, `quantity`, and
`projected_physical_unit` identify its interpretation.

As in the total-gas analysis, corrupted projection model
`R8_8pc_NCR_row0000` is excluded. The comparison contains the same clean 31
models, ranked and colored by mean `sfr10` over 200--600 Myr.

## 2. PDFs

Run the two species with

```bash
plot-suite-density-pdf /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --quantity hi --workers 8
plot-suite-density-pdf /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --quantity em --workers 8
```

Every pixel is area weighted. Instantaneous \(\sigma_\delta\) and \(\sigma_s\)
are measured directly from all pixels, independent of the finite histogram
range. Because H I can be extremely small on ionized sightlines and EM can
have a strong high tail, the automatic histogram limits differ by tracer:

| quantity | default \(\delta\) range | default \(s\) range |
|---|---:|---:|
| total gas | \([-1,30]\) | \([-6,4]\) |
| H I | \([-1,30]\) | \([-24,5]\) |
| EM | \([-1,300]\) | \([-12,8]\) |

`--delta-min`, `--delta-max`, `--s-min`, and `--s-max` override individual
limits. Histogram normalization retains all pixels in the denominator, so an
integral smaller than unity explicitly indicates probability outside the
stored range. The direct width measurements are unaffected.

The H I products are written beneath `SUITE/hi_pdf_theta0/` as
`hi_pdfs.npz`, `hi_pdf_widths.csv`, and quantity-labeled figures. The EM
products use `SUITE/em_pdf_theta0/` and the corresponding `em_*` names. See
[density_pdf.md](density_pdf.md) for the estimator, temporal summaries,
Gaussian comparison, and correlation products.

### Low-column H I bump and neutral-fraction censoring

The very low-column component is not well described by a log-normal. A direct
full-VTK check was made for `R8_8pc_NCR_row0010` at \(t=380.0019\) Myr (VTK
0041), an epoch with a prominent secondary peak at \(s=-7.775\). The diagnostic
reconstructs

\[
\Sigma_{\rm HI}=\int n_{\rm H}x_{\rm HI}\,dz
\]

and repeats the projection after setting the contribution from cells with
\(x_{\rm HI}<0.01\) to zero. Run it with

```bash
plot-hi-fraction-mask \
  /tigress/changgoo/anvil/TIGRESS-NCR-suite/R8_8pc_NCR_row0010 41 \
  --xhi-min 0.01 --overwrite
```

The raw VTK reconstruction agrees with the stored proj2d map to a median
relative error of \(2.1\times10^{-8}\). Sightlines within
\(|s+7.775|<0.15\), which contain 2.44% of the map area, lose 100% of their H I
column under the cut. The cut removes only 0.0973% of the global H I column but
makes 22.29% of all sightlines exactly zero. Thus the prominent bump is made
by tiny residual H I columns in gas with \(x_{\rm HI}<0.01\); it is not a
separate neutral structure. This sensitivity does not by itself prove that
the small neutral fractions are numerically wrong--they may be physical
residual fractions--but it does show that the low-column PDF component is a
censoring/floor-sensitive tracer feature.

For H I, report the zero/censored area fraction separately and compare the
conditional PDF of detected positive columns. Robust scalar alternatives to
\(\sigma_s\) include log-column interpercentile widths above a stated column
threshold and linear-space fractional widths; an observational
\(N_{\rm HI}\) threshold is generally easier to interpret than a cell-level
\(x_{\rm HI}\) cut. The diagnostic products are written to
`SUITE/hi_low_fraction_mask_test/`.

## 3. Power spectra

Run the shear-aware spectra with

```bash
plot-suite-density-spectrum /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --quantity hi --workers 8
plot-suite-density-spectrum /tigress/changgoo/anvil/TIGRESS-NCR-suite \
  --quantity em --workers 8
```

For each tracer, the map is remapped to periodic shearing coordinates and the
physical shearing-wave relation
\(k_x=k_{x,0}+q\Omega t_{\rm remap}k_y\) is used before annular averaging.
The per-model 2D caches are deliberately separate:

- total gas: `proj2d/theta0/density_power_2d/density_power_2d.npz`;
- H I: `proj2d/theta0/hi_power_2d/density_power_2d.npz`;
- EM: `proj2d/theta0/em_power_2d/density_power_2d.npz`.

Suite products are written beneath `hi_power_spectrum_theta0/` and
`em_power_spectrum_theta0/`. Each contains the complete time-dependent 1D
archive, the physical-grid mean 2D spectra, integral scale, fixed-band slope,
quadrupole amplitude and angle, temporal summaries, and environmental-
parameter correlations. Add `--movie` only when the full time evolution movie
is needed. See [density_power_spectrum.md](density_power_spectrum.md) for the
normalization and exact diagnostic definitions.

## 4. Interpretation

The three spectra measure morphology of *fractional* fluctuations, not the
absolute tracer luminosity or column. H I suppresses ionized and molecular
material, while EM weights electron density squared and therefore emphasizes
compact dense ionized structures. Similar spectral slopes need not imply
similar absolute columns or filling factors.

Within the fixed-size NCR suite, temporal ranges and environmental trends in
\(\sigma_s\), \(\sigma_\delta\), \(L_{\rm in}\), \(\alpha\), and \(A_2\) can
be compared consistently. Across different box sizes, retain the convergence
cautions from the total-gas study: integral scale and dimensional power
normalization are intrinsically sensitive to newly available large scales;
the slope is meaningful only when the same resolved physical fit interval is
available; and a band-limited variance or shape-normalized spectrum is often a
more robust cross-box diagnostic.

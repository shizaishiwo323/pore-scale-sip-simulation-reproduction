# Figure 8 Mechanism-Resolved Comparison

## Figure Contract

- Claim: the reproduced 350^3 GPU-FFT simulations can be separated into operational interfacial, pore, and membrane polarization contributions and compared directly with the paper's experimental Figure 8 data.
- Evidence: panel a compares imaginary conductivity; panel b compares the derived real relative permittivity.
- Scope: the paper simulation curves are intentionally excluded from this figure.

## Results

| mechanism | peak_frequency_hz | peak_abs_imag_s_m | hardest_frequency_hz | max_iterations | max_residual |
|---|---:|---:|---:|---:|---:|
| Interfacial polarization | 1e+09 | 8.727e-01 | 10000 | 391 | 9.975e-06 |
| Pore polarization | 1 | 2.176e-05 | 0.001 | 168 | 9.701e-06 |
| Membrane polarization | 10 | 5.116e-04 | 0.001 | 164 | 9.957e-06 |
| All polarization | 1e+09 | 8.727e-01 | 10000 | 418 | 9.958e-06 |

## Interpretation

- The membrane-polarization run dominates the low- to mid-frequency response in the reproduced component split, which is consistent with the full all-polarization curve being largely controlled by this term around the main dispersion band.
- The pore-polarization run is smaller over most frequencies, but it still contributes a resolvable low-frequency imaginary-conductivity shoulder.
- The interfacial-only run is weak at low frequency and becomes important mainly at high frequency where dielectric contrast terms dominate.
- The all-polarization curve is plotted as the direct full material response; use it as the primary comparison to experiment, while the three mechanism curves show which enabled term controls each band.

## Review Risks

- This is an operational decomposition based on selectively enabling terms in the current material spectrum, not a guaranteed one-to-one reconstruction of the paper authors' internal Figure 8 mechanism files.
- The plotted y-values are magnitudes because the interfacial-only low-frequency imaginary response can be slightly negative under this sign convention and numerical setup.
- All three component sweeps used the x-direction, complex64, full 350^3 Berea grid, GPU BiCGSTAB, FFT-Poisson preconditioning, and `rtol = 1e-5`; tensor-direction averaging remains a separate validation step.

- Signed negative imaginary values occurred for: Interfacial polarization: 4 low-frequency point(s).

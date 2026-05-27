# Figure 6-8 Three-Source Comparison: Results and Discussion

## Result Summary

- The full-grid GPU-FFT sweep contains 12 representative frequencies; 12/12 reached `info = 0`.
- The maximum final relative residual is `9.958e-06`, so the plotted curve is no longer dominated by linear-solver non-convergence.
- At `1 Hz`, the reproduced effective conductivity is `0.00359059 + 0.000478144 i S/m`.

## Hardest Frequencies

| frequency_hz | iterations | relative_residual_norm |
|---:|---:|---:|
| 10000 | 418 | 9.355e-06 |
| 100000 | 205 | 9.254e-06 |
| 0.001 | 172 | 8.529e-06 |

## Interpretation

- The reproduced real conductivity follows the same broad frequency-increasing trend as the paper data, but its magnitude remains offset from both the paper simulation and experiment over several bands.
- The imaginary conductivity and derived real permittivity show the largest deviations, especially where polarization mechanisms dominate. This suggests the remaining mismatch is more likely controlled by material-parameter assumptions, polarization-mechanism partitioning, tensor-direction averaging, or phase assignment than by Krylov convergence.
- The paper Figure 8 simulation decomposes pore, membrane, and interfacial terms; the current reproduced curve corresponds to the available all-polarization material model in a single imposed-field direction, so it should be interpreted as a first full-grid reproduction curve rather than a final one-to-one mechanism-resolved reproduction.

## Next Checks

- Add `y` and `z` imposed-field directions and compare the diagonal/tensor average with the paper curves.
- Run key frequencies in `complex128` or verify final residuals with a higher-precision checkpoint.
- Reproduce the paper's separate pore, membrane, and interfacial polarization input sets before interpreting Figure 8 mechanism-level discrepancies.

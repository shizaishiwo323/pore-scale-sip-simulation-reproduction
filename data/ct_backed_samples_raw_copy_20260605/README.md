# CT-backed carbonate samples raw-copy bundle

Created: 2026-06-05

This directory separates the four samples that have copied raw CT slice stacks from the larger, mixed `Carbonates_Fan` and `James_Geophysics` archives. It is a copied working bundle only; the original data directories were not modified.

Original data roots:

- `C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\James_Geophysics`
- `C:\Users\imgw\Documents\Codex\论文复现\pore-scale-simulation-reproduction\Carbonates_Fan`

## Samples included

| Sample | Lithology | Group | Included data |
| --- | --- | --- | --- |
| `16` | Grainstone | LKC | CT slices, SIP, NMR original, NMR combined, CT rendering videos |
| `89` | Grainstone | LKC | CT slices, SIP, NMR original replicates `89-1` and `89-2`, NMR combined |
| `13-18` | Packstone | Willington/Wellington | CT slices, SIP, NMR original, NMR combined |
| `16-15` | Mudstone | Willington/Wellington | CT slices, SIP, NMR original, NMR combined, CT rendering videos |

Each sample is organized as:

```text
sample_<sample_id>_<lithology>/
  CT_slices/       copied raw TIFF slice stack from James_Geophysics
  SIP/             copied PSIP CSV file for the same sample id
  NMR_original/    copied original T2CPMG measurement directory/directories
  NMR_combined/    copied combined/processed NMR data directory
  imaging_videos/  copied CT rendering videos, where available
```

## Physical properties

The Pulse Decay K workbook has also been copied here:

- `physical_properties/ZhangFan core samples.xls`

The extracted target-sample rows are in:

- `physical_properties/pulse_decay_target_samples.csv`

Rows found in the workbook sheet `Cores_ZhangFan`:

| Sample | Diameter (cm) | Length (cm) | Bulk volume (cc) | Density (g/cc) | Porosity | Permeability SS (mD) | Permeability US (mD) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `89` | 2.54 | 5.01 | 25.38 | 2.50 | 0.061 | 0.036 | 0.052 |
| `13-18` | 3.81 | 3.24 | 36.92 | 2.61 | 0.066 | 9.784 |  |
| `16-15` | 3.81 | 3.10 | 35.36 | 2.65 | 0.064 | 0.044 | 0.044 |

Sample `16` was not found in this Pulse Decay workbook. The workbook note defines `SS` as steady-state measurement and `US` as unsteady-state / pulse-decay measurement.

## Important interpretation

The copied SIP and NMR data correspond to the same sample id at the whole-plug/sample scale. They are not tied to a single two-dimensional CT slice. The CT TIFF files are raw slice stacks used to reconstruct a 3D digital rock volume.

Use `sample_manifest.csv` as the quick index for the copied bundle.

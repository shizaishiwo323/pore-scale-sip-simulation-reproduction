# CT Core Physical Properties

## Raw Voxel Core

- Shape: `350 x 350 x 350` voxels.
- Dtype: `<u2`.
- Label mapping used here: `1` = pore/water, `2` = solid.
- Porosity from voxel counts: `0.2314697143`.
- Voxel size used for physical scaling: `2.800000e-06 m`.
- Physical side lengths: `9.800000e-04, 9.800000e-04, 9.800000e-04 m`.
- Bulk volume: `9.411920e-10 m^3`.

## pnextract Network

- Pores: `2124`.
- Throats: `3849`.
- Volume-weighted mean pore radius: `2.484921e-05 m`.
- Volume-weighted mean throat length: `3.870017e-05 m`.

## Algorithm Note

- The pnextract README identifies the extractor as a rewrite of the Dong and Blunt (2009) maximal-ball network extraction algorithm.
- Therefore the extracted pore network is a pore-throat, ball-and-stick representation based on maximal balls.
- The AC3D electrical simulation in this project is not solved on that network; it is solved on the original two-phase voxel grid, while the pnextract network supplies pore/throat statistics for polarization spectra.

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import tifffile
from scipy.io import netcdf_file


SCRIPT = Path(__file__).parents[1] / "code" / "scripts" / "pore_network" / "convert_netcdf_phase_to_binary_tiff.py"
SPEC = spec_from_file_location("convert_netcdf_phase_to_binary_tiff", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_convert_preserves_stack_and_maps_labels(tmp_path: Path) -> None:
    source = tmp_path / "phase.nc"
    output = tmp_path / "binary.tiff"
    manifest = tmp_path / "manifest.json"
    phase = np.array([[[1, 2], [3, 1]], [[4, 5], [1, 3]]], dtype=np.int8)

    with netcdf_file(source, "w") as dataset:
        dataset.createDimension("z", 2)
        dataset.createDimension("y", 2)
        dataset.createDimension("x", 2)
        dataset.voxel_size_xyz = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        dataset.voxel_unit = "um"
        variable = dataset.createVariable("phase", "b", ("z", "y", "x"))
        variable[:] = phase

    result = MODULE.convert(source, output, manifest, variable="phase", pore_labels=[1], solid_labels=[2, 3, 4, 5])

    converted = tifffile.imread(output)
    assert converted.shape == phase.shape
    assert np.array_equal(converted, np.where(phase == 1, 0, 255).astype(np.uint8))
    assert result["source_label_counts"] == {"1": 3, "2": 1, "3": 2, "4": 1, "5": 1}
    assert result["macro_porosity"] == 3 / 8

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    parser = argparse.ArgumentParser(description="Count porosity from an ilastik Simple Segmentation export.")
    parser.add_argument("--input", required=True, help="Segmentation file: .h5/.hdf5/.tif/.tiff")
    parser.add_argument("--dataset", default="exported_data", help="HDF5 dataset path")
    parser.add_argument("--pore-label", type=int, default=2, help="Label value that represents pore")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    return parser.parse_args()


def load_array(path: Path, dataset: str):
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(path, "r") as h5:
            if dataset not in h5:
                available = []
                h5.visititems(lambda name, obj: available.append(name) if hasattr(obj, "shape") else None)
                raise KeyError(f"Dataset '{dataset}' not found. Available datasets: {available}")
            array = h5[dataset][...]
    elif suffix in {".tif", ".tiff"}:
        import tifffile

        array = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported file extension: {path.suffix}")

    return np.asarray(array)


def squeeze_trailing_single_channel(array):
    if array.ndim >= 1 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    array = squeeze_trailing_single_channel(load_array(input_path, args.dataset))
    total_voxels = int(array.size)
    pore_voxels = int(np.count_nonzero(array == args.pore_label))
    porosity = pore_voxels / total_voxels if total_voxels else float("nan")

    labels, counts = np.unique(array, return_counts=True)
    label_counts = {str(int(label)): int(count) for label, count in zip(labels, counts)}

    result = {
        "input": str(input_path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "pore_label": args.pore_label,
        "pore_voxels": pore_voxels,
        "total_voxels": total_voxels,
        "porosity_fraction": porosity,
        "porosity_percent": porosity * 100.0,
        "label_counts": label_counts,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Input: {result['input']}")
        print(f"Shape: {tuple(result['shape'])}, dtype: {result['dtype']}")
        print("Label counts:")
        for label in sorted(label_counts, key=lambda value: int(value)):
            print(f"  label {label}: {label_counts[label]:,}")
        print("")
        print(f"Pore label: {args.pore_label}")
        print(f"Pore voxels: {pore_voxels:,}")
        print(f"Total voxels: {total_voxels:,}")
        print(f"Porosity: {porosity * 100.0:.6f}%")


if __name__ == "__main__":
    main()

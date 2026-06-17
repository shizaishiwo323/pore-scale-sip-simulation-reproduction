from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "code" / "scripts" / "pore_network" / "contact_patch_split_diagnostic.py"


def load_module():
    spec = importlib.util.spec_from_file_location("contact_patch_split_diagnostic", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_contact_patches_splits_disconnected_faces_for_same_pore_pair():
    module = load_module()
    labels = -np.ones((5, 5, 5), dtype=np.int32)

    # Two pore labels touch at two disconnected y-z face patches across x.
    labels[1, 1:3, 1:3] = 2
    labels[2, 1:3, 1:3] = 3
    labels[1, 3:5, 3:5] = 2
    labels[2, 3:5, 3:5] = 3

    patches = module.find_contact_patches(labels, pore_ids={2, 3})

    assert len(patches) == 2
    assert {patch.pore1_id for patch in patches} == {2}
    assert {patch.pore2_id for patch in patches} == {3}
    assert sorted(patch.face_count for patch in patches) == [4, 4]


def test_summarize_patch_split_conserves_aggregated_throat_volume():
    module = load_module()
    labels = -np.ones((5, 5, 5), dtype=np.int32)
    labels[1, 1:3, 1:3] = 2
    labels[2, 1:3, 1:3] = 3
    labels[1, 3:5, 3:5] = 2
    labels[2, 3:5, 3:5] = 3

    pores = {
        2: module.PoreGeometry(pore_id=2, center_voxels=np.array([1.0, 2.0, 2.0]), radius_voxels=2.0),
        3: module.PoreGeometry(pore_id=3, center_voxels=np.array([2.0, 2.0, 2.0]), radius_voxels=2.0),
    }
    throat = module.AggregatedThroat(
        pore1_id=2,
        pore2_id=3,
        length_voxels=10.0,
        radius_voxels=1.0,
        volume_voxels3=8.0,
    )

    split = module.summarize_patch_split(
        module.find_contact_patches(labels, pore_ids={2, 3}),
        pores=pores,
        throats={(2, 3): throat},
        voxel_size_um=2.0,
    )

    assert len(split) == 2
    assert np.isclose(sum(row["split_volume_voxels3"] for row in split), 8.0)
    assert all(row["patch_count_for_pair"] == 2 for row in split)
    assert all(row["split_length_um"] > 0 for row in split)

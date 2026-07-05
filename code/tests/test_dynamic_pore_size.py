from __future__ import annotations

import numpy as np
import pytest

from pore_scale_electrical.dynamic_pore_size import (
    compute_dynamic_pore_size_from_fields,
    fields_from_periodic_potential,
    physical_direction_to_array_axis,
)


def test_physical_direction_to_array_axis_records_zyx_mapping():
    assert physical_direction_to_array_axis("x", "zyx") == 2
    assert physical_direction_to_array_axis("y", "zyx") == 1
    assert physical_direction_to_array_axis("z", "zyx") == 0


def test_flat_pore_channel_lambda_equals_channel_thickness():
    h = 2.5e-6
    channel_voxels = 5
    pore = np.zeros((channel_voxels + 2, 4, 3), dtype=bool)
    pore[1:-1, :, :] = True
    fields = np.zeros((3, *pore.shape), dtype=float)
    fields[1, pore] = 1.0

    result = compute_dynamic_pore_size_from_fields(pore, fields, h)

    assert result.dynamic_pore_size_m == pytest.approx(channel_voxels * h, rel=1e-12)
    assert result.uniform_field_proxy_m == pytest.approx(channel_voxels * h, rel=1e-12)


def test_dynamic_pore_size_scales_with_voxel_size_but_not_field_strength():
    pore = np.zeros((6, 3, 3), dtype=bool)
    pore[1:-1, :, :] = True
    fields = np.zeros((3, *pore.shape), dtype=float)
    fields[2, pore] = 2.0

    base = compute_dynamic_pore_size_from_fields(pore, fields, 1.0e-6)
    scaled_voxel = compute_dynamic_pore_size_from_fields(pore, fields, 4.0e-6)
    scaled_field = compute_dynamic_pore_size_from_fields(pore, fields * 10.0, 1.0e-6)

    assert scaled_voxel.dynamic_pore_size_m == pytest.approx(4.0 * base.dynamic_pore_size_m)
    assert scaled_field.dynamic_pore_size_m == pytest.approx(base.dynamic_pore_size_m)


def test_fields_from_periodic_potential_ignores_solid_potential_for_interface_energy():
    pore = np.zeros((5, 4, 3), dtype=bool)
    pore[1:-1, :, :] = True
    potential = np.zeros(pore.shape, dtype=float)
    potential[~pore] = 1.0e9

    fields = fields_from_periodic_potential(
        potential,
        physical_direction="y",
        axis_order="zyx",
        field_strength_v_m=1.0,
        voxel_size_m=1.0,
        pore_mask=pore,
    )
    result = compute_dynamic_pore_size_from_fields(pore, fields, 1.0)

    assert result.dynamic_pore_size_m == pytest.approx(3.0)

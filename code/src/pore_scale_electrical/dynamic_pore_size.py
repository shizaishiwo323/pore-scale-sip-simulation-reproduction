"""Field-weighted dynamic pore size helpers for voxelized AC3D domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


PhysicalDirection = Literal["x", "y", "z"]


@dataclass(frozen=True)
class DynamicPoreSizeIntegral:
    volume_integral_v2_m: float
    surface_integral_v2: float
    dynamic_pore_size_m: float
    interface_face_count: int
    pore_voxel_count: int
    pore_volume_m3: float
    voxel_surface_area_m2: float
    uniform_field_proxy_m: float


def physical_direction_to_array_axis(direction: str, axis_order: str) -> int:
    """Map a physical direction name to the array axis recorded by ``axis_order``."""

    normalized_direction = direction.lower()
    normalized_axis_order = axis_order.lower()
    if sorted(normalized_axis_order) != ["x", "y", "z"] or len(normalized_axis_order) != 3:
        raise ValueError("axis_order must be a permutation of x, y, z")
    if normalized_direction not in {"x", "y", "z"}:
        raise ValueError("direction must be one of x, y, or z")
    return normalized_axis_order.index(normalized_direction)


def _field_energy(field_components: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(field_components) ** 2, axis=0)


def compute_dynamic_pore_size_from_fields(
    pore_mask: np.ndarray,
    field_components: np.ndarray,
    voxel_size_m: float,
) -> DynamicPoreSizeIntegral:
    """Compute Lambda from pore-center fields and pore-side tangential interface fields."""

    pore = np.asarray(pore_mask, dtype=bool)
    fields = np.asarray(field_components)
    if pore.ndim != 3:
        raise ValueError("pore_mask must be 3-D")
    if fields.shape != (3, *pore.shape):
        raise ValueError("field_components must have shape (3, *pore_mask.shape)")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")

    pore_voxel_count = int(np.count_nonzero(pore))
    if pore_voxel_count == 0:
        raise ValueError("pore_mask must contain at least one pore voxel")

    volume_energy = _field_energy(fields)
    volume_integral = float(np.sum(volume_energy[pore]) * voxel_size_m**3)

    surface_energy_sum = 0.0
    interface_face_count = 0
    for axis in range(3):
        neighbor_pore = np.roll(pore, -1, axis=axis)
        pore_to_solid = pore & ~neighbor_pore
        solid_to_pore = ~pore & neighbor_pore

        if np.any(pore_to_solid):
            tangential = volume_energy[pore_to_solid] - np.abs(fields[axis][pore_to_solid]) ** 2
            surface_energy_sum += float(np.sum(tangential))
            interface_face_count += int(np.count_nonzero(pore_to_solid))
        if np.any(solid_to_pore):
            pore_side_components = np.roll(fields, -1, axis=axis + 1)
            pore_side_energy = _field_energy(pore_side_components)
            tangential = pore_side_energy[solid_to_pore] - np.abs(pore_side_components[axis][solid_to_pore]) ** 2
            surface_energy_sum += float(np.sum(tangential))
            interface_face_count += int(np.count_nonzero(solid_to_pore))

    if interface_face_count == 0:
        raise ValueError("pore_mask has no pore-solid interface faces")
    surface_integral = float(surface_energy_sum * voxel_size_m**2)
    if surface_integral <= np.finfo(float).tiny:
        raise ValueError("surface integral is zero; tangential interface field is absent")

    pore_volume = float(pore_voxel_count * voxel_size_m**3)
    voxel_surface_area = float(voxel_size_m**2)
    uniform_proxy = float(2.0 * pore_volume / (interface_face_count * voxel_surface_area))
    return DynamicPoreSizeIntegral(
        volume_integral_v2_m=volume_integral,
        surface_integral_v2=surface_integral,
        dynamic_pore_size_m=float(2.0 * volume_integral / surface_integral),
        interface_face_count=interface_face_count,
        pore_voxel_count=pore_voxel_count,
        pore_volume_m3=pore_volume,
        voxel_surface_area_m2=voxel_surface_area,
        uniform_field_proxy_m=uniform_proxy,
    )


def fields_from_periodic_potential(
    potential: np.ndarray,
    *,
    physical_direction: PhysicalDirection,
    axis_order: str,
    field_strength_v_m: float,
    voxel_size_m: float,
    pore_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct pore-side center fields from the periodic correction potential."""

    u = np.asarray(potential)
    if u.ndim != 3:
        raise ValueError("potential must be 3-D")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")
    pore = np.ones(u.shape, dtype=bool) if pore_mask is None else np.asarray(pore_mask, dtype=bool)
    if pore.shape != u.shape:
        raise ValueError("pore_mask must match potential shape")

    fields = np.zeros((3, *u.shape), dtype=np.result_type(u, np.float64))
    drive_axis = physical_direction_to_array_axis(physical_direction, axis_order)
    for axis in range(3):
        forward_pore = np.roll(pore, -1, axis=axis)
        backward_pore = np.roll(pore, 1, axis=axis)
        forward_u = np.roll(u, -1, axis=axis)
        backward_u = np.roll(u, 1, axis=axis)
        central = pore & forward_pore & backward_pore
        forward_only = pore & forward_pore & ~backward_pore
        backward_only = pore & ~forward_pore & backward_pore

        grad = np.zeros(u.shape, dtype=fields.dtype)
        grad[central] = (forward_u[central] - backward_u[central]) / (2.0 * voxel_size_m)
        grad[forward_only] = (forward_u[forward_only] - u[forward_only]) / voxel_size_m
        grad[backward_only] = (u[backward_only] - backward_u[backward_only]) / voxel_size_m
        fields[axis] = -grad

    fields[drive_axis, pore] += field_strength_v_m
    fields[:, ~pore] = 0.0
    return fields

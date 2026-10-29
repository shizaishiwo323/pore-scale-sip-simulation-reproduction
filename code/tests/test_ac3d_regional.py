import numpy as np
import pytest

from pore_scale_electrical.ac3d_gpu import cupy_available
from pore_scale_electrical.ac3d_regional import (
    build_region_map,
    local_thickness_gpu,
    regional_dynamic_pore_sizes,
    solve_voxel_conductivity_gpu,
    voxel_sensitivities,
)
from pore_scale_electrical.ac3d_solver import solve_ac3d
from pore_scale_electrical.dynamic_pore_size import (
    compute_dynamic_pore_size_from_fields,
    fields_from_periodic_potential,
)


def _channel_volume():
    """Insulating block with a straight x-channel whose middle part is narrower."""

    pore = np.zeros((12, 9, 9), dtype=bool)
    pore[:, 2:7, 2:7] = True
    pore[4:8, :, :] = False
    pore[4:8, 4:5, 4:5] = True
    return pore


def test_region_sums_recover_global_dynamic_pore_size():
    pore = _channel_volume()
    # direct LU needs a non-singular matrix: 1e-12 S/m stands in for the insulating solid
    sigma = np.where(pore, 0.05, 1.0e-12)
    potential = solve_ac3d(sigma, "x").potential.real
    fields = fields_from_periodic_potential(
        potential, physical_direction="x", axis_order="xyz", field_strength_v_m=1.0, voxel_size_m=1.0, pore_mask=pore
    )
    labels = np.zeros(pore.shape, dtype=np.int32)
    labels[pore] = 1
    labels[4:8][pore[4:8]] = 2
    regional = regional_dynamic_pore_sizes(labels, fields, 1.0, n_regions=2)
    reference = compute_dynamic_pore_size_from_fields(pore, fields, 1.0)
    assert regional.global_dynamic_pore_size_m == pytest.approx(reference.dynamic_pore_size_m, rel=1e-12)
    assert regional.volume_integral_v2_m.sum() == pytest.approx(reference.volume_integral_v2_m, rel=1e-12)
    # the 1x1 constriction concentrates the field: smaller field-weighted size than the wide body
    assert regional.dynamic_pore_size_m[2] < regional.dynamic_pore_size_m[1]


def test_build_region_map_assigns_throat_segment_within_half_length():
    pore = np.ones((1, 1, 20), dtype=bool)
    velems = np.full(pore.shape, 2, dtype=np.int32)  # pore id 1
    velems[..., 10:] = 3  # pore id 2
    velems[..., 0] = -1  # unassigned -> nearest pore (id 1)
    throat_voxels = np.zeros(pore.shape, dtype=np.int32)
    throat_voxels[..., 9:11] = 1
    region = build_region_map(
        pore, velems, throat_voxels,
        throat_pore1=np.array([1]), throat_pore2=np.array([2]), throat_length_m=np.array([6.0]),
        voxel_size_m=1.0, n_pores=2,
    )
    labels = region.labels[0, 0]
    assert labels[0] == 1
    assert np.all(labels[6:14] == 3)  # within 3 voxels of the interface voxels 9 and 10
    assert labels[5] == 1 and labels[14] == 2
    assert region.stats["pnextract_unassigned_pore_voxels_filled_by_nearest_pore"] == 1


def test_voxel_sensitivities_predict_small_regional_increments():
    pore = _channel_volume()
    labels = np.where(pore, 1, 0)
    labels[4:8][pore[4:8]] = 2
    sigma0 = np.where(pore, 0.12, 1.5e-3)  # conductive matrix (two-scale carbonate case)
    base = solve_ac3d(sigma0, "x")
    sens = voxel_sensitivities(sigma0, base.potential.real, drive_axis=0)
    per_region = np.bincount(labels.ravel(), weights=sens.ravel())
    increments = np.array([2.0e-5 + 4.0e-6j, 1.0e-3 + 5.0e-4j, 4.0e-3 + 1.0e-3j])  # matrix, body, constriction
    perturbed = solve_ac3d(sigma0 + increments[labels], "x").effective_conductivity_s_m
    predicted = base.effective_conductivity_s_m + np.sum(increments * per_region) / labels.size
    change = perturbed - base.effective_conductivity_s_m
    assert abs(predicted - perturbed) < 0.02 * abs(change)
    # homogeneity: scaling every conductivity by (1 + e) scales sigma_eff by (1 + e)
    assert np.sum(sigma0 * sens) / labels.size == pytest.approx(base.effective_conductivity_s_m.real, rel=1e-6)


@pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA is not available")
@pytest.mark.parametrize("solid_value", [0.0, 2.0e-4 + 1.0e-4j])
def test_voxel_gpu_solve_matches_direct_solver_on_regional_map(solid_value):
    rng = np.random.default_rng(3)
    pore = _channel_volume()
    sigma = np.where(pore, 0.043 + 0.0j, solid_value).astype(np.complex128)
    sigma[pore] += rng.uniform(0.0, 0.01, pore.sum()) + 1j * rng.uniform(0.0, 1.0e-3, pore.sum())
    reference_sigma = np.where(pore, sigma, solid_value if solid_value != 0 else 1.0e-12)
    reference = solve_ac3d(reference_sigma, "x").effective_conductivity_s_m
    result = solve_voxel_conductivity_gpu(sigma, drive_axis=0, fft_reference_s_m=0.043, rtol=1e-10, maxiter=2000)
    assert result.true_residual_passed
    assert result.effective_conductivity_s_m == pytest.approx(reference, rel=1e-8)
    assert result.energy_form_conductivity_s_m == pytest.approx(reference, rel=1e-8)


@pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA is not available")
def test_local_thickness_of_a_slab_equals_its_width():
    phase = np.zeros((40, 12, 12), dtype=bool)
    phase[5:15] = True  # 10-voxel slab between pore layers
    phase[20:36] = True  # 16-voxel slab
    thickness = local_thickness_gpu(phase)
    assert np.all(np.abs(thickness[5:15] - 10) <= 1)
    assert np.all(np.abs(thickness[20:36] - 16) <= 1)
    assert np.all(thickness[~phase] == 0)

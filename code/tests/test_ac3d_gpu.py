import numpy as np
import pytest

from pore_scale_electrical.ac3d_gpu import (
    _gpu_bicgstab,
    _wrap_preconditioner_with_gauge_rows,
    cp,
    cupy_available,
    face_type_conductivity_gpu,
    make_fft_poisson_preconditioner_gpu,
    matrix_free_matvec_face_types_gpu,
    normalize_info_with_true_residual,
    solve_ac3d_bicgstab_compact_gpu_face_types,
    solve_ac3d_cocg_gpu_face_types,
    solve_ac3d_red_black_sor_gpu_face_types,
    solve_ac3d_matrix_free_gpu_face_types,
    solve_ac3d_weighted_jacobi_gpu_face_types,
)
from pore_scale_electrical.ac3d_solver import (
    face_type_conductivity,
    matrix_free_matvec_face_types,
    phase_conductivity_grid,
    solve_ac3d,
)


pytestmark = pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA is not available")


def test_normalize_info_with_true_residual_marks_recursive_only_convergence_failed():
    assert normalize_info_with_true_residual(0, True) == 0
    assert normalize_info_with_true_residual(0, False) == -20
    assert normalize_info_with_true_residual(500, False) == 500


def test_gpu_face_type_matvec_matches_cpu():
    rng = np.random.default_rng(13)
    labels = rng.choice(np.array([1, 2], dtype=np.uint16), size=(3, 4, 5))
    water_sigma = 0.4 + 0.03j
    solid_sigma = 0.02 + 0.001j
    vector = rng.random(labels.size) + 1j * rng.random(labels.size)

    cpu_face_data = face_type_conductivity(labels, 1, 2, water_sigma, solid_sigma)
    gpu_face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma)

    cpu_result = matrix_free_matvec_face_types(cpu_face_data, vector)
    gpu_result = cp.asnumpy(matrix_free_matvec_face_types_gpu(gpu_face_data, cp.asarray(vector)))

    assert np.allclose(gpu_result, cpu_result, rtol=1e-12, atol=1e-12)


def test_gpu_solver_matches_direct_small_grid():
    labels = np.ones((4, 4, 3), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma)
    face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma)

    direct = solve_ac3d(grid, direction="x")
    gpu = solve_ac3d_matrix_free_gpu_face_types(face_data, direction="x", rtol=1e-10, maxiter=200)

    assert gpu.info == 0
    assert gpu.residual_norm < 1e-8
    assert gpu.true_residual_norm < 1e-8
    assert np.isclose(gpu.recursive_residual_norm, gpu.residual_history[-1][1])
    assert gpu.true_residual_passed is True
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-8, atol=1e-10)
    assert gpu.residual_history


def test_gpu_solver_active_domain_gauge_handles_zero_solid_rows():
    labels = np.full((4, 4, 3), 2, dtype=np.uint16)
    labels[0, :, :] = 1
    labels[2, :, :] = 1
    face_data = face_type_conductivity_gpu(labels, 1, 2, 1.0 + 0.1j, 0.0j)

    gpu = solve_ac3d_matrix_free_gpu_face_types(
        face_data,
        direction="y",
        rtol=1e-8,
        maxiter=200,
        gauge_mode="active-domain",
    )

    assert gpu.info == 0
    assert gpu.true_residual_passed is True
    assert gpu.true_residual_norm < 1e-8


def test_gpu_bicgstab_continues_after_recursive_residual_false_convergence():
    rng = np.random.default_rng(1)
    labels = rng.choice(np.array([1, 2], dtype=np.uint16), size=(4, 4, 4), p=[0.5, 0.5])
    face_data = face_type_conductivity_gpu(labels, 1, 2, 0.043 + 1e-12j, 0.0j, dtype=np.complex64)

    gpu = solve_ac3d_matrix_free_gpu_face_types(
        face_data,
        direction="x",
        rtol=1e-7,
        maxiter=500,
        preconditioner="jacobi",
        gauge_mode="auto",
    )

    assert gpu.info == 0
    assert gpu.true_residual_passed is True
    assert gpu.true_residual_norm <= 1e-7


def test_gpu_bicgstab_refreshes_true_residual_at_history_checkpoints():
    class CountingDiagonalOperator:
        def __init__(self) -> None:
            self.calls = 0
            self.diagonal = cp.asarray([2.0, 3.0, 4.0, 5.0], dtype=cp.complex64)

        def matvec(self, vector):
            self.calls += 1
            return self.diagonal * vector

    operator = CountingDiagonalOperator()
    rhs = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=cp.complex64)
    _, _, iterations, _, _, _, history = _gpu_bicgstab(
        operator,
        rhs,
        x0=None,
        rtol=1.0e-30,
        atol=0.0,
        maxiter=2,
        preconditioner=None,
        iteration_callback=None,
        residual_every=1,
        residual_callback=None,
    )

    assert iterations == 2
    assert len(history) == 2
    assert operator.calls >= 8


def test_fft_preconditioner_wrapper_preserves_gauge_rows():
    vector = cp.asarray([1.0, 2.0, 3.0, 4.0], dtype=cp.complex128)
    identity_mask = cp.asarray([True, False, False, False])
    gauge_indices = cp.asarray([2], dtype=cp.int64)

    def double_preconditioner(values):
        return values * 2.0

    wrapped = _wrap_preconditioner_with_gauge_rows(
        double_preconditioner,
        identity_mask=identity_mask,
        gauge_indices=gauge_indices,
    )
    out = cp.asnumpy(wrapped(vector))

    assert np.allclose(out, [1.0, 4.0, 3.0, 8.0])


def test_gpu_fft_preconditioner_solver_matches_direct_small_grid():
    labels = np.ones((4, 4, 3), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma)
    face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma)

    apply_fft, reference = make_fft_poisson_preconditioner_gpu(face_data)
    vector = cp.asarray(np.arange(labels.size, dtype=np.float64) + 1j)
    preconditioned = apply_fft(vector)

    assert abs(reference) > 0
    assert preconditioned.shape == vector.shape
    assert bool(cp.all(cp.isfinite(preconditioned)).get())

    direct = solve_ac3d(grid, direction="x")
    gpu = solve_ac3d_matrix_free_gpu_face_types(
        face_data,
        direction="x",
        rtol=1e-10,
        maxiter=200,
        preconditioner="fft",
    )

    assert gpu.info == 0
    assert gpu.residual_norm < 1e-8
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-8, atol=1e-10)


def test_gpu_weighted_jacobi_solver_matches_direct_small_grid():
    labels = np.ones((4, 4, 3), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)
    face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)

    direct = solve_ac3d(grid, direction="y")
    gpu = solve_ac3d_weighted_jacobi_gpu_face_types(
        face_data,
        direction="y",
        rtol=1e-6,
        maxiter=10000,
        omega=0.65,
        residual_every=1000,
    )

    assert gpu.residual_norm < 1e-5
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-4, atol=1e-6)


def test_gpu_red_black_sor_solver_matches_direct_even_periodic_grid():
    labels = np.ones((4, 4, 4), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)
    face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)

    direct = solve_ac3d(grid, direction="y")
    gpu = solve_ac3d_red_black_sor_gpu_face_types(
        face_data,
        direction="y",
        rtol=1e-6,
        maxiter=10000,
        omega=1.25,
        residual_every=100,
    )

    assert gpu.residual_norm < 1e-5
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-4, atol=1e-6)


def test_gpu_cocg_solver_matches_direct_small_grid():
    labels = np.ones((4, 4, 3), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)
    face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)

    direct = solve_ac3d(grid, direction="y")
    gpu = solve_ac3d_cocg_gpu_face_types(
        face_data,
        direction="y",
        rtol=1e-7,
        maxiter=200,
        residual_every=1,
    )

    assert gpu.info == 0
    assert gpu.residual_norm < 1e-6
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-5, atol=1e-7)


def test_gpu_compact_bicgstab_solver_matches_direct_small_grid():
    labels = np.ones((4, 4, 3), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)
    face_data = face_type_conductivity_gpu(labels, 1, 2, water_sigma, solid_sigma, dtype=np.complex64)

    direct = solve_ac3d(grid, direction="y")
    gpu = solve_ac3d_bicgstab_compact_gpu_face_types(
        face_data,
        direction="y",
        rtol=1e-7,
        maxiter=200,
        residual_every=1,
    )

    assert gpu.info == 0
    assert gpu.residual_norm < 1e-6
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-5, atol=1e-7)

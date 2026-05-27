import numpy as np
import pytest

from pore_scale_electrical.ac3d_gpu import (
    cp,
    cupy_available,
    face_type_conductivity_gpu,
    make_fft_poisson_preconditioner_gpu,
    matrix_free_matvec_face_types_gpu,
    solve_ac3d_matrix_free_gpu_face_types,
)
from pore_scale_electrical.ac3d_solver import (
    face_type_conductivity,
    matrix_free_matvec_face_types,
    phase_conductivity_grid,
    solve_ac3d,
)


pytestmark = pytest.mark.skipif(not cupy_available(), reason="CuPy/CUDA is not available")


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
    assert np.isclose(gpu.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-8, atol=1e-10)
    assert gpu.residual_history


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

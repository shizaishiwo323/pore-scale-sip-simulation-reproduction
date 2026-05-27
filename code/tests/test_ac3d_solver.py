import numpy as np

from pore_scale_electrical.ac3d_solver import (
    build_face_type_codes,
    face_conductivity_arrays_from_types,
    face_type_conductivity,
    harmonic_face_conductivity,
    matrix_free_matvec,
    matrix_free_matvec_faces,
    matrix_free_matvec_face_types,
    phase_conductivity_grid,
    solve_ac3d,
    solve_ac3d_matrix_free,
    solve_ac3d_matrix_free_face_types,
)


def test_harmonic_face_conductivity_matches_half_cell_series():
    left = np.array([1.0 + 0.5j])
    right = np.array([3.0 + 0.25j])
    expected = 1.0 / (0.5 / left + 0.5 / right)

    assert np.allclose(harmonic_face_conductivity(left, right), expected)


def test_uniform_complex_medium_returns_input_conductivity():
    sigma0 = 0.043 + 2.0e-4j
    grid = np.full((4, 3, 5), sigma0, dtype=np.complex128)

    for direction in ("x", "y", "z"):
        result = solve_ac3d(grid, direction=direction)
        assert np.isclose(result.effective_conductivity_s_m, sigma0, rtol=1e-11, atol=1e-13)
        assert result.residual_norm < 1e-10


def test_parallel_layers_match_arithmetic_mean():
    sigma1 = 1.0 + 0.1j
    sigma2 = 4.0 + 0.2j
    grid = np.empty((4, 4, 3), dtype=np.complex128)
    grid[:, :2, :] = sigma1
    grid[:, 2:, :] = sigma2

    result = solve_ac3d(grid, direction="x")
    expected = 0.5 * (sigma1 + sigma2)

    assert np.isclose(result.effective_conductivity_s_m, expected, rtol=1e-10, atol=1e-12)


def test_series_layers_match_harmonic_mean():
    sigma1 = 1.0 + 0.1j
    sigma2 = 4.0 + 0.2j
    grid = np.empty((4, 3, 3), dtype=np.complex128)
    grid[:2, :, :] = sigma1
    grid[2:, :, :] = sigma2

    result = solve_ac3d(grid, direction="x")
    expected = 1.0 / (0.5 / sigma1 + 0.5 / sigma2)

    assert np.isclose(result.effective_conductivity_s_m, expected, rtol=1e-10, atol=1e-12)


def test_matrix_free_operator_matches_explicit_matrix():
    rng = np.random.default_rng(42)
    sigma = 0.1 + rng.random((3, 4, 2)) + 1j * rng.random((3, 4, 2)) * 0.01
    vector = rng.random(sigma.size) + 1j * rng.random(sigma.size)

    from pore_scale_electrical.ac3d_solver import build_periodic_system

    matrix, _rhs = build_periodic_system(sigma, direction="x")
    assert np.allclose(matrix @ vector, matrix_free_matvec(sigma, vector))


def test_face_type_operator_matches_roll_operator_for_two_phase_grid():
    rng = np.random.default_rng(7)
    labels = rng.choice(np.array([1, 2], dtype=np.uint16), size=(3, 4, 5))
    water_sigma = 0.4 + 0.03j
    solid_sigma = 0.02 + 0.001j
    sigma = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma)
    vector = rng.random(sigma.size) + 1j * rng.random(sigma.size)

    face_data = face_type_conductivity(labels, 1, 2, water_sigma, solid_sigma)
    face_arrays = face_conductivity_arrays_from_types(face_data)

    assert np.allclose(matrix_free_matvec(sigma, vector), matrix_free_matvec_faces(face_arrays, vector))
    assert np.allclose(matrix_free_matvec(sigma, vector), matrix_free_matvec_face_types(face_data, vector))


def test_face_type_codes_are_compact_uint8_arrays():
    labels = np.array([[[1, 1, 2], [2, 1, 2]]], dtype=np.uint16)
    codes = build_face_type_codes(labels, 1, 2)

    assert len(codes) == 3
    assert all(code.dtype == np.uint8 for code in codes)
    assert all(code.shape == labels.shape for code in codes)


def test_matrix_free_solver_matches_direct_solver():
    grid = np.empty((4, 4, 3), dtype=np.complex128)
    grid[:, :2, :] = 1.0 + 0.1j
    grid[:, 2:, :] = 4.0 + 0.2j

    direct = solve_ac3d(grid, direction="x")
    iterative = solve_ac3d_matrix_free(grid, direction="x", rtol=1e-10, maxiter=200)

    assert iterative.info == 0
    assert iterative.residual_norm < 1e-8
    assert np.isclose(iterative.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-8, atol=1e-10)


def test_matrix_free_solver_accepts_x0_and_records_residual_history():
    grid = np.empty((4, 4, 3), dtype=np.complex128)
    grid[:, :2, :] = 1.0 + 0.1j
    grid[:, 2:, :] = 4.0 + 0.2j
    x0 = np.zeros(grid.shape, dtype=np.complex128)

    iterative = solve_ac3d_matrix_free(grid, direction="x", rtol=1e-10, maxiter=200, x0=x0, residual_every=1)

    assert iterative.info == 0
    assert iterative.residual_history
    assert iterative.residual_history[-1][0] == iterative.iterations
    assert np.isclose(iterative.residual_history[-1][1], iterative.residual_norm)


def test_face_type_solver_matches_direct_solver():
    labels = np.ones((4, 4, 3), dtype=np.uint16)
    labels[:, 2:, :] = 2
    water_sigma = 1.0 + 0.1j
    solid_sigma = 4.0 + 0.2j
    grid = phase_conductivity_grid(labels, 1, 2, water_sigma, solid_sigma)
    face_data = face_type_conductivity(labels, 1, 2, water_sigma, solid_sigma)

    direct = solve_ac3d(grid, direction="x")
    iterative = solve_ac3d_matrix_free_face_types(face_data, direction="x", rtol=1e-10, maxiter=200)

    assert iterative.info == 0
    assert iterative.residual_norm < 1e-8
    assert np.isclose(iterative.effective_conductivity_s_m, direct.effective_conductivity_s_m, rtol=1e-8, atol=1e-10)


def test_phase_label_mapping_rejects_unknown_labels():
    labels = np.array([[[1, 2], [1, 3]]], dtype=np.uint16)

    try:
        phase_conductivity_grid(labels, pore_label=1, solid_label=2, water_conductivity_s_m=1.0, solid_conductivity_s_m=0.0)
    except ValueError as exc:
        assert "unexpected labels" in str(exc)
    else:
        raise AssertionError("unknown labels should raise ValueError")

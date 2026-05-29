import numpy as np

from pore_scale_electrical.ac2d_solver import phase_conductivity_grid_2d, solve_ac2d_dirichlet


def test_uniform_complex_medium_returns_input_conductivity_2d():
    sigma0 = 0.013 + 2.0e-5j
    grid = np.full((7, 9), sigma0, dtype=np.complex128)

    result = solve_ac2d_dirichlet(grid, dx_m=3.5e-6, dy_m=3.5e-6)

    assert np.isclose(result.effective_conductivity_s_m, sigma0, rtol=1e-11, atol=1e-13)
    assert result.residual_norm < 1e-10


def test_parallel_layers_match_arithmetic_mean_2d():
    sigma1 = 1.0 + 0.1j
    sigma2 = 4.0 + 0.2j
    grid = np.empty((8, 10), dtype=np.complex128)
    grid[:4, :] = sigma1
    grid[4:, :] = sigma2

    result = solve_ac2d_dirichlet(grid, dx_m=1.0, dy_m=1.0)
    expected = 0.5 * (sigma1 + sigma2)

    assert np.isclose(result.effective_conductivity_s_m, expected, rtol=1e-10, atol=1e-12)


def test_series_layers_match_harmonic_mean_2d():
    sigma1 = 1.0 + 0.1j
    sigma2 = 4.0 + 0.2j
    grid = np.empty((6, 10), dtype=np.complex128)
    grid[:, :5] = sigma1
    grid[:, 5:] = sigma2

    result = solve_ac2d_dirichlet(grid, dx_m=1.0, dy_m=1.0)
    expected = 1.0 / (0.5 / sigma1 + 0.5 / sigma2)

    assert np.isclose(result.effective_conductivity_s_m, expected, rtol=1e-10, atol=1e-12)


def test_phase_label_mapping_rejects_background_2d():
    labels = np.array([[1, 2], [1, 0]], dtype=np.uint8)

    try:
        phase_conductivity_grid_2d(labels, 1, 2, 1.0, 0.0)
    except ValueError as exc:
        assert "unexpected labels" in str(exc)
    else:
        raise AssertionError("background labels should be rejected")

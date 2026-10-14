import numpy as np
import pytest

from pore_scale_electrical import polarization as niu
from pore_scale_electrical.ac3d_solver import solve_ac3d
from pore_scale_electrical_2d import (
    Grid2D,
    RegionSpec,
    SIP2DParameters,
    TwoPoreChain2D,
    cell_fields,
    conductivity_map,
    debye_decomposition,
    dynamic_pore_size_2d,
    graded_axis,
    membrane_increments,
    phase_values,
    pore_increments,
    radius_from_relaxation_time,
    solve_ac2d_periodic,
)
from pore_scale_electrical_2d.polarization import stern_conductance, titov_bracket


def graded_grid(nx=12, ny=10):
    rng = np.random.default_rng(3)
    return Grid2D(rng.uniform(0.5, 2.0, nx) * 1e-6, rng.uniform(0.5, 2.0, ny) * 1e-6)


def test_uniform_medium_on_graded_grid_returns_input_conductivity():
    grid = graded_grid()
    sigma0 = 0.043 + 3.0e-6j
    result = solve_ac2d_periodic(np.full(grid.shape, sigma0), grid)
    assert np.isclose(result.effective_conductivity_s_m, sigma0, rtol=1e-12)
    assert abs(result.transverse_conductivity_s_m) < 1e-14
    assert result.residual_norm < 1e-10


def test_parallel_and_series_layers_on_graded_grid():
    grid = graded_grid()
    s1, s2 = 1.0 + 0.1j, 4.0 + 0.3j
    rows = np.arange(grid.ny) < 4
    parallel = np.where(rows[:, None], s1, s2) * np.ones(grid.shape)
    weight = grid.dy_m[rows].sum() / grid.length_y_m
    expected_parallel = weight * s1 + (1 - weight) * s2
    assert np.isclose(solve_ac2d_periodic(parallel, grid).effective_conductivity_s_m, expected_parallel, rtol=1e-12)

    cols = np.arange(grid.nx) < 5
    series = np.where(cols[None, :], s1, s2) * np.ones(grid.shape)
    weight = grid.dx_m[cols].sum() / grid.length_x_m
    expected_series = 1.0 / (weight / s1 + (1 - weight) / s2)
    assert np.isclose(solve_ac2d_periodic(series, grid).effective_conductivity_s_m, expected_series, rtol=1e-12)


def test_straight_channel_in_insulator_is_exact_and_has_lambda_equal_width():
    dy = graded_axis(10e-6, 4e-6, 5e-6, 0.125e-6, 1e-6)
    grid = Grid2D(np.full(20, 0.5e-6), dy)
    yc = grid.y_centers_m
    channel = (yc > 4e-6) & (yc < 5e-6)
    sigma = np.where(channel[:, None], 0.043, 0.0) * np.ones(grid.shape)
    result = solve_ac2d_periodic(sigma, grid)
    assert np.isclose(result.effective_conductivity_s_m.real, 0.043 * 1e-6 / grid.length_y_m, rtol=1e-12)
    fields = cell_fields(result, sigma, grid)
    lam = dynamic_pore_size_2d(sigma != 0, fields, grid)
    assert np.isclose(lam.dynamic_pore_size_m, 1e-6, rtol=1e-10)
    assert np.isclose(lam.geometric_size_m, 1e-6, rtol=1e-10)


def test_isolated_inclusion_carries_no_dc_current():
    grid = Grid2D.uniform(30, 30, 1e-6)
    x, y = np.meshgrid(grid.x_centers_m, grid.y_centers_m)
    disk = (x - 15e-6) ** 2 + (y - 15e-6) ** 2 < (8e-6) ** 2
    result = solve_ac2d_periodic(np.where(disk, 0.043, 0.0), grid)
    assert result.n_components == 1
    assert abs(result.effective_conductivity_s_m) < 1e-15


def test_matches_existing_ac3d_solver_on_random_two_phase_slice():
    rng = np.random.default_rng(7)
    labels = rng.random((9, 11)) < 0.55
    sigma = np.where(labels, 0.043 + 2e-5j, 1e-4 + 5e-7j)
    voxel = 2.8e-6
    ours_x = solve_ac2d_periodic(sigma, Grid2D.uniform(11, 9, voxel), "x").effective_conductivity_s_m
    ours_y = solve_ac2d_periodic(sigma, Grid2D.uniform(11, 9, voxel), "y").effective_conductivity_s_m
    volume = sigma.T[:, :, None]  # AC3D axis 0 = x, axis 1 = y, single z layer
    ref_x = solve_ac3d(volume, "x", voxel_size_m=voxel).effective_conductivity_s_m
    ref_y = solve_ac3d(volume, "y", voxel_size_m=voxel).effective_conductivity_s_m
    assert np.isclose(ours_x, ref_x, rtol=1e-10)
    assert np.isclose(ours_y, ref_y, rtol=1e-10)


def test_kernels_equal_niu_3d_polarization_module():
    freq = np.logspace(-4, 5, 37)
    params = SIP2DParameters(stern_diffusion_m2_s=1.3e-9, membrane_diffusion_m2_s=1.3e-9, membrane_polarizability=0.01)
    reference = niu.PolarizationParameters(membrane_polarizability=0.01)
    radius, length, zdc = 3e-6, 15e-6, 1.7e9
    expected_pore = niu.elementary_pore_conductance(freq, np.array([radius]), reference)[:, 0]
    assert np.allclose(stern_conductance(freq, radius, params), expected_pore, rtol=1e-13, atol=0)
    expected_bracket = niu.elementary_membrane_impedance(freq, np.array([length]), np.array([zdc]), reference)[:, 0] / zdc
    assert np.allclose(titov_bracket(freq, length, params), expected_bracket, rtol=1e-13, atol=0)


def test_regional_and_uniform_pore_increments():
    freq = np.logspace(-3, 3, 7)
    params = SIP2DParameters(stern_diffusion_m2_s=3.8e-12)
    regions = {
        1: RegionSpec(1, "body", "pore_body", 5e-6, 4e-6),
        2: RegionSpec(2, "throat", "throat", 0.05e-6, 0.1e-6, membrane_length_m=15e-6),
    }
    regional = pore_increments(freq, regions, params, pore_mode="regional")
    assert np.allclose(regional[2], 2 * stern_conductance(freq, 0.05e-6, params) / 0.1e-6)
    areas = {1: 3.0, 2: 1.0}
    uniform = pore_increments(freq, regions, params, pore_mode="uniform", region_areas_m2=areas, global_dynamic_pore_size_m=2e-7)
    expected = 2 * (0.75 * stern_conductance(freq, 5e-6, params) + 0.25 * stern_conductance(freq, 0.05e-6, params)) / 2e-7
    assert np.allclose(uniform[1], expected) and np.allclose(uniform[2], expected)
    membrane = membrane_increments(freq, regions, params)
    assert np.all(membrane[1] == 0)
    assert np.allclose(membrane[2], params.water_conductivity_s_m * (1 / titov_bracket(freq, 15e-6, params) - 1))


def test_mechanism_isolation_rules():
    freq = np.array([1.0, 1e3])
    params = SIP2DParameters(stern_diffusion_m2_s=3.8e-12)
    regions = {1: RegionSpec(1, "throat", "throat", 0.05e-6, 0.1e-6, membrane_length_m=15e-6)}
    pore = pore_increments(freq, regions, params, pore_mode="regional")
    mem = membrane_increments(freq, regions, params)
    omega = 2 * np.pi * freq
    solid, water = phase_values("pore", freq, regions, params, pore=pore, membrane=mem)
    assert np.all(solid == 0) and np.allclose(water[1], params.water_conductivity_s_m + pore[1])
    solid, water = phase_values("membrane", freq, regions, params, pore=pore, membrane=mem)
    assert np.all(solid == 0) and np.allclose(water[1], params.water_conductivity_s_m + mem[1])
    solid, water = phase_values("interfacial", freq, regions, params, pore=pore, membrane=mem)
    assert np.allclose(solid, 1j * omega * params.solid_permittivity_f_m)
    assert np.allclose(water[1], params.water_conductivity_s_m + 1j * omega * params.water_permittivity_f_m)
    solid, water = phase_values("all", freq, regions, params, pore=pore, membrane=mem)
    assert np.allclose(water[1], params.water_conductivity_s_m + 1j * omega * params.water_permittivity_f_m + pore[1] + mem[1])
    with pytest.raises(ValueError):
        conductivity_map(np.array([[0, 2]]), 0.0, {1: 1.0})


def test_debye_decomposition_recovers_two_relaxations_and_permittivity():
    freq = np.logspace(-4, 5, 91)
    omega = 2 * np.pi * freq
    tau_true = np.array([2.0, 2e-4])
    m_true = np.array([1e-4, 4e-4])
    sigma = 0.01 + sum(m * 1j * omega * t / (1 + 1j * omega * t) for m, t in zip(m_true, tau_true)) + 1j * omega * 6e-10
    tau_grid = np.logspace(-7, 3, 101)
    fit = debye_decomposition(freq, sigma, tau_grid, regularization=1e-4)
    assert fit.success
    assert np.isclose(fit.sigma_0_s_m, 0.01, rtol=1e-3)
    assert np.isclose(fit.epsilon_inf_f_m, 6e-10, rtol=0.05)
    log_tau = np.log10(tau_grid)
    for t, m in zip(tau_true, m_true):
        window = np.abs(log_tau - np.log10(t)) < 0.6
        assert np.isclose(fit.amplitudes_s_m[window].sum(), m, rtol=0.05)
        assert abs(np.average(log_tau[window], weights=fit.amplitudes_s_m[window]) - np.log10(t)) < 0.05
    assert fit.imag_rmse_s_m < 1e-3 * np.max(sigma.imag)
    assert np.isclose(radius_from_relaxation_time(0.5 / 3.8e-12 * 1e-12, 3.8e-12), 1e-6)


def test_graded_axis_and_two_pore_chain_geometry():
    axis = graded_axis(44e-6, 21.9e-6, 22.1e-6, 0.0125e-6, 0.25e-6)
    assert np.isclose(axis.sum(), 44e-6, rtol=1e-12)
    edges = np.r_[0, np.cumsum(axis)]
    assert np.min(np.abs(edges - 21.9e-6)) < 1e-15 and np.min(np.abs(edges - 22.1e-6)) < 1e-15

    chain = TwoPoreChain2D(20e-6, 5e-6, 0.1e-6, 15e-6, 2e-6)
    grid = chain.grid(coarse_step_m=0.25e-6)
    labels = chain.region_map(grid)
    assert set(np.unique(labels)) == {0, 1, 2, 3, 4}
    area = grid.cell_area_m2
    exact = chain.analytic_properties()
    for label in (1, 2, 3, 4):
        assert np.isclose(area[labels == label].sum(), exact[label]["area_m2"], rtol=0.02)
    result = solve_ac2d_periodic(np.where(labels > 0, 0.043, 0.0), grid)
    assert result.n_components == 1
    assert result.effective_conductivity_s_m.real > 0

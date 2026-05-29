"""2-D complex finite-volume solver for microfluidic SIP simulations.

The solver uses cell-centered unknowns, Dirichlet potentials on the left and
right channel boundaries, and no-flux boundaries on the top and bottom. Face
conductances include the pixel aspect ratio, so rectangular pixels from image
calibration can be used directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pore_scale_electrical.ac3d_solver import harmonic_face_conductivity


Direction2D = Literal["x"]


@dataclass(frozen=True)
class AC2DResult:
    """Result for one 2-D imposed-field solve."""

    effective_conductivity_s_m: complex
    total_current_a_per_m: complex
    voltage_drop_v: float
    field_strength_v_m: float
    residual_norm: float
    potential: np.ndarray
    solver: str


def phase_conductivity_grid_2d(
    labels: np.ndarray,
    water_label: int,
    solid_label: int,
    water_conductivity_s_m: complex,
    solid_conductivity_s_m: complex,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    """Map a two-phase 2-D label image to a complex conductivity grid."""

    label_grid = np.asarray(labels)
    if label_grid.ndim != 2:
        raise ValueError("labels must be a 2-D array")
    complex_dtype = np.dtype(dtype)
    if complex_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("dtype must be complex64 or complex128")

    water_mask = label_grid == water_label
    solid_mask = label_grid == solid_label
    if not np.all(water_mask | solid_mask):
        bad = np.unique(label_grid[~(water_mask | solid_mask)])
        raise ValueError(f"unexpected labels in 2-D domain: {bad[:10]}")

    sigma = np.empty(label_grid.shape, dtype=complex_dtype)
    sigma[water_mask] = water_conductivity_s_m
    sigma[solid_mask] = solid_conductivity_s_m
    return sigma


def build_dirichlet_system_2d(
    conductivity_s_m: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
    voltage_left_v: float = 1.0,
    voltage_right_v: float = 0.0,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build the sparse 2-D finite-volume system.

    Left and right electrodes are represented as Dirichlet boundaries outside
    the first and last cell columns. Their half-cell conductances make a
    uniform grid recover the input conductivity exactly for ``Lx = nx * dx``.
    """

    sigma = np.asarray(conductivity_s_m, dtype=np.complex128)
    if sigma.ndim != 2:
        raise ValueError("conductivity_s_m must be a 2-D array")
    if sigma.shape[1] < 2:
        raise ValueError("conductivity_s_m must have at least two columns")
    if dx_m <= 0 or dy_m <= 0:
        raise ValueError("dx_m and dy_m must be positive")

    ny, nx = sigma.shape
    n_cells = ny * nx
    indices = np.arange(n_cells, dtype=np.int64).reshape(sigma.shape)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    rhs = np.zeros(n_cells, dtype=np.complex128)
    diagonal = np.zeros(n_cells, dtype=np.complex128)

    # x-neighbor faces, area per unit thickness = dy, distance = dx.
    g_x = harmonic_face_conductivity(sigma[:, :-1], sigma[:, 1:]) * (dy_m / dx_m)
    left = indices[:, :-1].ravel()
    right = indices[:, 1:].ravel()
    g = g_x.ravel()
    diagonal[left] += g
    diagonal[right] += g
    rows.extend([left, right])
    cols.extend([right, left])
    data.extend([-g, -g])

    # y-neighbor faces, area per unit thickness = dx, distance = dy.
    if ny > 1:
        g_y = harmonic_face_conductivity(sigma[:-1, :], sigma[1:, :]) * (dx_m / dy_m)
        top = indices[:-1, :].ravel()
        bottom = indices[1:, :].ravel()
        gy = g_y.ravel()
        diagonal[top] += gy
        diagonal[bottom] += gy
        rows.extend([top, bottom])
        cols.extend([bottom, top])
        data.extend([-gy, -gy])

    # Dirichlet half-cell boundary conductances at the left and right ends.
    g_left = 2.0 * sigma[:, 0] * (dy_m / dx_m)
    g_right = 2.0 * sigma[:, -1] * (dy_m / dx_m)
    left_cells = indices[:, 0].ravel()
    right_cells = indices[:, -1].ravel()
    diagonal[left_cells] += g_left
    diagonal[right_cells] += g_right
    rhs[left_cells] += g_left * voltage_left_v
    rhs[right_cells] += g_right * voltage_right_v

    cell_indices = indices.ravel()
    rows.append(cell_indices)
    cols.append(cell_indices)
    data.append(diagonal)

    matrix = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_cells, n_cells))
    return matrix.tocsr(), rhs


def boundary_current_left_a_per_m(
    conductivity_s_m: np.ndarray,
    potential: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
    voltage_left_v: float = 1.0,
) -> complex:
    """Return total current entering through the left boundary per unit depth."""

    sigma = np.asarray(conductivity_s_m)
    u = np.asarray(potential).reshape(sigma.shape)
    g_left = 2.0 * sigma[:, 0] * (dy_m / dx_m)
    return complex(np.sum(g_left * (voltage_left_v - u[:, 0])))


def solve_ac2d_dirichlet(
    conductivity_s_m: np.ndarray,
    *,
    dx_m: float,
    dy_m: float,
    voltage_left_v: float = 1.0,
    voltage_right_v: float = 0.0,
    solver: Literal["direct"] = "direct",
) -> AC2DResult:
    """Solve a 2-D complex conductivity problem with left-right drive."""

    if solver != "direct":
        raise ValueError("only direct sparse solve is implemented for AC2D v1")
    sigma = np.asarray(conductivity_s_m, dtype=np.complex128)
    matrix, rhs = build_dirichlet_system_2d(
        sigma,
        dx_m=dx_m,
        dy_m=dy_m,
        voltage_left_v=voltage_left_v,
        voltage_right_v=voltage_right_v,
    )
    potential_flat = spla.spsolve(matrix, rhs)
    residual = matrix @ potential_flat - rhs
    residual_norm = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).eps))
    potential = potential_flat.reshape(sigma.shape)
    total_current = boundary_current_left_a_per_m(
        sigma,
        potential,
        dx_m=dx_m,
        dy_m=dy_m,
        voltage_left_v=voltage_left_v,
    )
    ny, nx = sigma.shape
    width_m = nx * dx_m
    height_m = ny * dy_m
    voltage_drop = voltage_left_v - voltage_right_v
    field_strength = voltage_drop / width_m
    effective = total_current * width_m / (voltage_drop * height_m)
    return AC2DResult(
        effective_conductivity_s_m=complex(effective),
        total_current_a_per_m=total_current,
        voltage_drop_v=float(voltage_drop),
        field_strength_v_m=float(field_strength),
        residual_norm=residual_norm,
        potential=potential,
        solver=solver,
    )

"""Complex periodic finite-volume solver for AC3D-style conductivity problems.

The solver computes the periodic correction potential ``u`` for a prescribed
macroscopic electric field ``E``:

``div(sigma * (E - grad(u))) = 0``.

This is the cell-centered counterpart of the AC3D equation used by Niu et al.
(2020). Face conductivities use the half-cell series, or harmonic, average
listed in the reproduction plan.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


Direction = Literal["x", "y", "z"] | int
MatvecBackend = Literal["roll", "face-types"]

FACE_PORE_PORE = np.uint8(0)
FACE_SOLID_SOLID = np.uint8(1)
FACE_PORE_SOLID = np.uint8(2)


@dataclass(frozen=True)
class AC3DResult:
    """Result for one imposed-field direction."""

    direction: str
    effective_conductivity_s_m: complex
    mean_current_density_a_m2: complex
    field_strength_v_m: float
    residual_norm: float
    potential: np.ndarray
    solver: str
    iterations: int | None = None


@dataclass(frozen=True)
class AC3DIterativeResult:
    """Result from the matrix-free Krylov solver."""

    direction: str
    effective_conductivity_s_m: complex
    mean_current_density_a_m2: complex
    field_strength_v_m: float
    residual_norm: float
    potential: np.ndarray | None
    solver: str
    iterations: int
    info: int
    residual_history: tuple[tuple[int, float], ...] = ()
    recursive_residual_norm: float | None = None
    true_residual_norm: float | None = None
    true_residual_passed: bool | None = None


@dataclass(frozen=True)
class FaceTypeConductivity:
    """Precomputed phase face types and conductance values for a two-phase grid."""

    face_types: tuple[np.ndarray, np.ndarray, np.ndarray]
    conductance_values: np.ndarray
    shape: tuple[int, int, int]
    dtype: np.dtype


def _complex_dtype(*arrays_or_dtypes: object) -> np.dtype:
    dtype = np.result_type(*arrays_or_dtypes)
    if dtype == np.dtype(np.complex64):
        return np.dtype(np.complex64)
    return np.dtype(np.complex128)


def _bicgstab(
    operator: spla.LinearOperator,
    rhs: np.ndarray,
    *,
    x0: np.ndarray | None,
    rtol: float,
    atol: float,
    maxiter: int | None,
    preconditioner: spla.LinearOperator | None,
    callback: Callable[[np.ndarray], None],
) -> tuple[np.ndarray, int]:
    """Call SciPy bicgstab across old ``tol`` and newer ``rtol`` APIs."""

    parameters = inspect.signature(spla.bicgstab).parameters
    kwargs: dict[str, object] = {
        "x0": x0,
        "maxiter": maxiter,
        "M": preconditioner,
        "callback": callback,
    }
    if "rtol" in parameters:
        kwargs["rtol"] = rtol
        kwargs["atol"] = atol
    else:
        kwargs["tol"] = rtol
        if "atol" in parameters:
            kwargs["atol"] = atol
    return spla.bicgstab(operator, rhs, **kwargs)


def direction_to_axis(direction: Direction) -> int:
    if isinstance(direction, int):
        if direction not in (0, 1, 2):
            raise ValueError("integer direction must be 0, 1, or 2")
        return direction
    mapping = {"x": 0, "y": 1, "z": 2}
    try:
        return mapping[direction.lower()]
    except KeyError as exc:
        raise ValueError("direction must be one of x, y, z, 0, 1, or 2") from exc


def axis_to_direction(axis: int) -> str:
    return ("x", "y", "z")[axis]


def harmonic_face_conductivity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Half-cell series average: ``1 / (0.5/sigma_i + 0.5/sigma_j)``."""

    dtype = _complex_dtype(left, right)
    left = np.asarray(left, dtype=dtype)
    right = np.asarray(right, dtype=dtype)
    denominator = left + right
    out = np.zeros(np.broadcast_shapes(left.shape, right.shape), dtype=dtype)
    mask = np.abs(denominator) > np.finfo(float).tiny
    out[mask] = 2.0 * left[mask] * right[mask] / denominator[mask]
    return out


def phase_conductivity_grid(
    labels: np.ndarray,
    pore_label: int,
    solid_label: int,
    water_conductivity_s_m: complex,
    solid_conductivity_s_m: complex,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    """Map a two-phase label volume to complex conductivity."""

    labels = np.asarray(labels)
    complex_dtype = np.dtype(dtype)
    if complex_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("dtype must be complex64 or complex128")
    sigma = np.empty(labels.shape, dtype=complex_dtype)
    pore_mask = labels == pore_label
    solid_mask = labels == solid_label
    if not np.all(pore_mask | solid_mask):
        bad = np.unique(labels[~(pore_mask | solid_mask)])
        raise ValueError(f"unexpected labels in volume: {bad[:10]}")
    sigma[pore_mask] = water_conductivity_s_m
    sigma[solid_mask] = solid_conductivity_s_m
    return sigma


def build_face_type_codes(labels: np.ndarray, pore_label: int, solid_label: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode positive-axis periodic face phase pairs as compact uint8 arrays."""

    label_grid = np.asarray(labels)
    if label_grid.ndim != 3:
        raise ValueError("labels must be a 3-D array")
    pore_mask = label_grid == pore_label
    solid_mask = label_grid == solid_label
    if not np.all(pore_mask | solid_mask):
        bad = np.unique(label_grid[~(pore_mask | solid_mask)])
        raise ValueError(f"unexpected labels in volume: {bad[:10]}")

    face_types: list[np.ndarray] = []
    for axis in range(3):
        neighbor_pore = np.roll(pore_mask, -1, axis=axis)
        neighbor_solid = np.roll(solid_mask, -1, axis=axis)
        codes = np.empty(label_grid.shape, dtype=np.uint8)
        codes[pore_mask & neighbor_pore] = FACE_PORE_PORE
        codes[solid_mask & neighbor_solid] = FACE_SOLID_SOLID
        codes[(pore_mask & neighbor_solid) | (solid_mask & neighbor_pore)] = FACE_PORE_SOLID
        face_types.append(codes)
    return tuple(face_types)  # type: ignore[return-value]


def face_type_conductance_values(
    water_conductivity_s_m: complex,
    solid_conductivity_s_m: complex,
    dtype: np.dtype | type = np.complex128,
) -> np.ndarray:
    """Return conductance values ordered by FACE_* code constants."""

    complex_dtype = np.dtype(dtype)
    if complex_dtype not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("dtype must be complex64 or complex128")
    water = complex_dtype.type(water_conductivity_s_m)
    solid = complex_dtype.type(solid_conductivity_s_m)
    mixed = harmonic_face_conductivity(np.array([water], dtype=complex_dtype), np.array([solid], dtype=complex_dtype))[0]
    return np.array([water, solid, mixed], dtype=complex_dtype)


def face_type_conductivity(
    labels: np.ndarray,
    pore_label: int,
    solid_label: int,
    water_conductivity_s_m: complex,
    solid_conductivity_s_m: complex,
    dtype: np.dtype | type = np.complex128,
) -> FaceTypeConductivity:
    """Build compact face-type conductance data for a two-phase grid."""

    face_types = build_face_type_codes(labels, pore_label, solid_label)
    conductance_values = face_type_conductance_values(water_conductivity_s_m, solid_conductivity_s_m, dtype)
    return FaceTypeConductivity(
        face_types=face_types,
        conductance_values=conductance_values,
        shape=tuple(int(n) for n in np.asarray(labels).shape),
        dtype=np.dtype(dtype),
    )


def face_conductivity_arrays_from_types(face_data: FaceTypeConductivity) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand compact face-type codes to per-face conductance arrays for one frequency."""

    values = np.asarray(face_data.conductance_values, dtype=face_data.dtype)
    return tuple(values[codes] for codes in face_data.face_types)  # type: ignore[return-value]


def build_periodic_system(
    conductivity_s_m: np.ndarray,
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build the sparse linear system for the periodic correction potential."""

    sigma = np.asarray(conductivity_s_m, dtype=np.complex128)
    if sigma.ndim != 3:
        raise ValueError("conductivity_s_m must be a 3-D array")
    if field_strength_v_m == 0:
        raise ValueError("field_strength_v_m must be non-zero")
    if voxel_size_m <= 0:
        raise ValueError("voxel_size_m must be positive")

    drive_axis = direction_to_axis(direction)
    n_cells = int(np.prod(sigma.shape))
    indices = np.arange(n_cells, dtype=np.int64).reshape(sigma.shape)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    macro_divergence = np.zeros(n_cells, dtype=np.complex128)

    for axis in range(3):
        if sigma.shape[axis] <= 1:
            continue

        neighbor_indices = np.roll(indices, -1, axis=axis).ravel()
        cell_indices = indices.ravel()
        face_sigma = harmonic_face_conductivity(sigma, np.roll(sigma, -1, axis=axis)).ravel()

        rows.extend([cell_indices, cell_indices, neighbor_indices, neighbor_indices])
        cols.extend([cell_indices, neighbor_indices, neighbor_indices, cell_indices])
        data.extend([face_sigma, -face_sigma, face_sigma, -face_sigma])

        if axis == drive_axis:
            drive = face_sigma * field_strength_v_m * voxel_size_m
            np.add.at(macro_divergence, cell_indices, drive)
            np.add.at(macro_divergence, neighbor_indices, -drive)

    matrix = sp.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=(n_cells, n_cells))
    rhs = -macro_divergence

    matrix = matrix.tolil()
    matrix[0, :] = 0.0
    matrix[0, 0] = 1.0
    rhs[0] = 0.0
    return matrix.tocsr(), rhs


def face_conductivity_forward(conductivity_s_m: np.ndarray, axis: int) -> np.ndarray:
    """Conductivity on positive-axis periodic faces."""

    return harmonic_face_conductivity(conductivity_s_m, np.roll(conductivity_s_m, -1, axis=axis))


def matrix_free_matvec(conductivity_s_m: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Apply the periodic AC3D operator without building a sparse matrix."""

    dtype = _complex_dtype(conductivity_s_m, vector)
    sigma = np.asarray(conductivity_s_m, dtype=dtype)
    x = np.asarray(vector, dtype=dtype).reshape(sigma.shape)
    y = np.zeros_like(x)
    for axis in range(3):
        if sigma.shape[axis] <= 1:
            continue
        g_forward = face_conductivity_forward(sigma, axis)
        g_backward = np.roll(g_forward, 1, axis=axis)
        y += g_forward * (x - np.roll(x, -1, axis=axis))
        y += g_backward * (x - np.roll(x, 1, axis=axis))
    y = y.ravel()
    y[0] = vector[0]
    return y


def _accumulate_axis_flux(y: np.ndarray, x: np.ndarray, g_forward: np.ndarray, axis: int) -> None:
    if x.shape[axis] <= 1:
        return
    if axis == 0:
        flux = g_forward[:-1, :, :] * (x[:-1, :, :] - x[1:, :, :])
        y[:-1, :, :] += flux
        y[1:, :, :] -= flux
        boundary_flux = g_forward[-1, :, :] * (x[-1, :, :] - x[0, :, :])
        y[-1, :, :] += boundary_flux
        y[0, :, :] -= boundary_flux
    elif axis == 1:
        flux = g_forward[:, :-1, :] * (x[:, :-1, :] - x[:, 1:, :])
        y[:, :-1, :] += flux
        y[:, 1:, :] -= flux
        boundary_flux = g_forward[:, -1, :] * (x[:, -1, :] - x[:, 0, :])
        y[:, -1, :] += boundary_flux
        y[:, 0, :] -= boundary_flux
    elif axis == 2:
        flux = g_forward[:, :, :-1] * (x[:, :, :-1] - x[:, :, 1:])
        y[:, :, :-1] += flux
        y[:, :, 1:] -= flux
        boundary_flux = g_forward[:, :, -1] * (x[:, :, -1] - x[:, :, 0])
        y[:, :, -1] += boundary_flux
        y[:, :, 0] -= boundary_flux
    else:
        raise ValueError("axis must be 0, 1, or 2")


def matrix_free_matvec_faces(face_conductivities: tuple[np.ndarray, np.ndarray, np.ndarray], vector: np.ndarray) -> np.ndarray:
    """Apply the periodic AC3D operator using precomputed positive-face conductivities."""

    shape = face_conductivities[0].shape
    dtype = _complex_dtype(*(g.dtype for g in face_conductivities), vector)
    x = np.asarray(vector, dtype=dtype).reshape(shape)
    y = np.zeros_like(x)
    for axis, g_forward in enumerate(face_conductivities):
        _accumulate_axis_flux(y, x, np.asarray(g_forward, dtype=dtype), axis)
    y = y.ravel()
    y[0] = np.asarray(vector, dtype=dtype)[0]
    return y


def matrix_free_matvec_face_types(face_data: FaceTypeConductivity, vector: np.ndarray) -> np.ndarray:
    """Apply the operator from compact face-type data."""

    return matrix_free_matvec_faces(face_conductivity_arrays_from_types(face_data), vector)


def matrix_free_rhs(
    conductivity_s_m: np.ndarray,
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> np.ndarray:
    """Right-hand side for the imposed macroscopic field."""

    dtype = _complex_dtype(conductivity_s_m)
    sigma = np.asarray(conductivity_s_m, dtype=dtype)
    axis = direction_to_axis(direction)
    g_forward = face_conductivity_forward(sigma, axis)
    g_backward = np.roll(g_forward, 1, axis=axis)
    macro_divergence = field_strength_v_m * voxel_size_m * (g_forward - g_backward)
    rhs = -macro_divergence.ravel()
    rhs[0] = 0.0
    return rhs


def matrix_free_rhs_faces(
    face_conductivities: tuple[np.ndarray, np.ndarray, np.ndarray],
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> np.ndarray:
    """Right-hand side for precomputed positive-face conductivities."""

    axis = direction_to_axis(direction)
    g = face_conductivities[axis]
    macro_divergence = np.zeros(g.shape, dtype=g.dtype)
    drive = g * (field_strength_v_m * voxel_size_m)
    if axis == 0:
        macro_divergence[:-1, :, :] += drive[:-1, :, :]
        macro_divergence[1:, :, :] -= drive[:-1, :, :]
        macro_divergence[-1, :, :] += drive[-1, :, :]
        macro_divergence[0, :, :] -= drive[-1, :, :]
    elif axis == 1:
        macro_divergence[:, :-1, :] += drive[:, :-1, :]
        macro_divergence[:, 1:, :] -= drive[:, :-1, :]
        macro_divergence[:, -1, :] += drive[:, -1, :]
        macro_divergence[:, 0, :] -= drive[:, -1, :]
    else:
        macro_divergence[:, :, :-1] += drive[:, :, :-1]
        macro_divergence[:, :, 1:] -= drive[:, :, :-1]
        macro_divergence[:, :, -1] += drive[:, :, -1]
        macro_divergence[:, :, 0] -= drive[:, :, -1]
    rhs = -macro_divergence.ravel()
    rhs[0] = 0.0
    return rhs


def matrix_free_rhs_face_types(
    face_data: FaceTypeConductivity,
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> np.ndarray:
    """Right-hand side from compact face-type data."""

    return matrix_free_rhs_faces(face_conductivity_arrays_from_types(face_data), direction, field_strength_v_m, voxel_size_m)


def jacobi_inverse_diagonal(conductivity_s_m: np.ndarray) -> np.ndarray:
    """Jacobi preconditioner for the matrix-free periodic operator."""

    dtype = _complex_dtype(conductivity_s_m)
    sigma = np.asarray(conductivity_s_m, dtype=dtype)
    diagonal = np.zeros(sigma.shape, dtype=np.complex128)
    for axis in range(3):
        if sigma.shape[axis] <= 1:
            continue
        g_forward = face_conductivity_forward(sigma, axis)
        diagonal += g_forward + np.roll(g_forward, 1, axis=axis)
    diagonal = diagonal.ravel()
    diagonal[0] = 1.0
    inverse = np.zeros_like(diagonal)
    mask = np.abs(diagonal) > np.finfo(float).tiny
    inverse[mask] = 1.0 / diagonal[mask]
    return inverse


def jacobi_inverse_diagonal_faces(face_conductivities: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    """Jacobi preconditioner from positive-face conductivities."""

    shape = face_conductivities[0].shape
    dtype = _complex_dtype(*(g.dtype for g in face_conductivities))
    diagonal = np.zeros(shape, dtype=dtype)
    for axis, g_forward in enumerate(face_conductivities):
        g = np.asarray(g_forward, dtype=dtype)
        if axis == 0:
            diagonal[:-1, :, :] += g[:-1, :, :]
            diagonal[1:, :, :] += g[:-1, :, :]
            diagonal[-1, :, :] += g[-1, :, :]
            diagonal[0, :, :] += g[-1, :, :]
        elif axis == 1:
            diagonal[:, :-1, :] += g[:, :-1, :]
            diagonal[:, 1:, :] += g[:, :-1, :]
            diagonal[:, -1, :] += g[:, -1, :]
            diagonal[:, 0, :] += g[:, -1, :]
        else:
            diagonal[:, :, :-1] += g[:, :, :-1]
            diagonal[:, :, 1:] += g[:, :, :-1]
            diagonal[:, :, -1] += g[:, :, -1]
            diagonal[:, :, 0] += g[:, :, -1]
    diagonal = diagonal.ravel()
    diagonal[0] = 1.0
    inverse = np.zeros_like(diagonal)
    mask = np.abs(diagonal) > np.finfo(float).tiny
    inverse[mask] = 1.0 / diagonal[mask]
    return inverse


def jacobi_inverse_diagonal_face_types(face_data: FaceTypeConductivity) -> np.ndarray:
    """Jacobi preconditioner from compact face-type data."""

    return jacobi_inverse_diagonal_faces(face_conductivity_arrays_from_types(face_data))


def mean_current_density(
    conductivity_s_m: np.ndarray,
    potential: np.ndarray,
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> complex:
    """Compute volume-averaged current density for one direction."""

    axis = direction_to_axis(direction)
    dtype = _complex_dtype(conductivity_s_m, potential)
    sigma = np.asarray(conductivity_s_m, dtype=dtype)
    u = np.asarray(potential, dtype=dtype).reshape(sigma.shape)
    face_sigma = face_conductivity_forward(sigma, axis)
    face_field = field_strength_v_m - (np.roll(u, -1, axis=axis) - u) / voxel_size_m
    return complex(np.mean(face_sigma * face_field))


def mean_current_density_faces(
    face_conductivities: tuple[np.ndarray, np.ndarray, np.ndarray],
    potential: np.ndarray,
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> complex:
    """Compute volume-averaged current density from positive-face conductivities."""

    axis = direction_to_axis(direction)
    g = face_conductivities[axis]
    u = np.asarray(potential, dtype=g.dtype).reshape(g.shape)
    if axis == 0:
        current_sum = np.sum(g[:-1, :, :] * (field_strength_v_m - (u[1:, :, :] - u[:-1, :, :]) / voxel_size_m))
        current_sum += np.sum(g[-1, :, :] * (field_strength_v_m - (u[0, :, :] - u[-1, :, :]) / voxel_size_m))
    elif axis == 1:
        current_sum = np.sum(g[:, :-1, :] * (field_strength_v_m - (u[:, 1:, :] - u[:, :-1, :]) / voxel_size_m))
        current_sum += np.sum(g[:, -1, :] * (field_strength_v_m - (u[:, 0, :] - u[:, -1, :]) / voxel_size_m))
    else:
        current_sum = np.sum(g[:, :, :-1] * (field_strength_v_m - (u[:, :, 1:] - u[:, :, :-1]) / voxel_size_m))
        current_sum += np.sum(g[:, :, -1] * (field_strength_v_m - (u[:, :, 0] - u[:, :, -1]) / voxel_size_m))
    return complex(current_sum / np.prod(g.shape))


def solve_ac3d(
    conductivity_s_m: np.ndarray,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    solver: Literal["direct"] = "direct",
) -> AC3DResult:
    """Solve one complex effective-conductivity problem."""

    if solver != "direct":
        raise ValueError("only the direct sparse solver is implemented in this validated prototype")

    sigma = np.asarray(conductivity_s_m, dtype=np.complex128)
    axis = direction_to_axis(direction)
    matrix, rhs = build_periodic_system(sigma, axis, field_strength_v_m, voxel_size_m)
    potential_flat = spla.spsolve(matrix, rhs)
    residual = matrix @ potential_flat - rhs
    residual_norm = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).eps))
    potential = potential_flat.reshape(sigma.shape)

    mean_current = mean_current_density(sigma, potential, axis, field_strength_v_m, voxel_size_m)
    effective = mean_current / field_strength_v_m

    return AC3DResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=complex(effective),
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=residual_norm,
        potential=potential,
        solver=solver,
    )


def solve_ac3d_matrix_free(
    conductivity_s_m: np.ndarray,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    use_jacobi: bool = True,
    return_potential: bool = True,
    iteration_callback: Callable[[int], None] | None = None,
    x0: np.ndarray | None = None,
    residual_every: int = 0,
    residual_callback: Callable[[int, float], None] | None = None,
) -> AC3DIterativeResult:
    """Solve with a matrix-free BiCGSTAB Krylov iteration.

    This prototype keeps the same operator as ``solve_ac3d`` but avoids
    materializing the sparse matrix. It still stores several Krylov vectors in
    memory, so it is intended for medium grids unless paired with a lower-level
    streaming implementation.
    """

    dtype = _complex_dtype(conductivity_s_m)
    sigma = np.asarray(conductivity_s_m, dtype=dtype)
    axis = direction_to_axis(direction)
    n_cells = int(np.prod(sigma.shape))
    rhs = matrix_free_rhs(sigma, axis, field_strength_v_m, voxel_size_m)
    rhs_norm = max(np.linalg.norm(rhs), np.finfo(float).eps)

    operator = spla.LinearOperator(
        (n_cells, n_cells),
        matvec=lambda vector: matrix_free_matvec(sigma, vector),
        dtype=dtype,
    )
    preconditioner = None
    if use_jacobi:
        inverse_diagonal = jacobi_inverse_diagonal(sigma)
        preconditioner = spla.LinearOperator(
            (n_cells, n_cells),
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=dtype,
        )

    iterations = 0
    residual_history: list[tuple[int, float]] = []

    def callback(xk: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1
        if iteration_callback is not None:
            iteration_callback(iterations)
        if residual_every > 0 and iterations % residual_every == 0:
            residual_norm = float(np.linalg.norm(matrix_free_matvec(sigma, xk) - rhs) / rhs_norm)
            residual_history.append((iterations, residual_norm))
            if residual_callback is not None:
                residual_callback(iterations, residual_norm)

    initial_guess = None if x0 is None else np.asarray(x0, dtype=dtype).ravel()
    solution, info = _bicgstab(
        operator,
        rhs,
        x0=initial_guess,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        callback=callback,
    )
    residual = matrix_free_matvec(sigma, solution) - rhs
    residual_norm = float(np.linalg.norm(residual) / rhs_norm)
    if not residual_history or residual_history[-1][0] != iterations:
        residual_history.append((iterations, residual_norm))
        if residual_callback is not None:
            residual_callback(iterations, residual_norm)
    potential = solution.reshape(sigma.shape)
    mean_current = mean_current_density(sigma, potential, axis, field_strength_v_m, voxel_size_m)
    effective = mean_current / field_strength_v_m

    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=complex(effective),
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=residual_norm,
        potential=potential if return_potential else None,
        solver="matrix_free_bicgstab_jacobi" if use_jacobi else "matrix_free_bicgstab",
        iterations=iterations,
        info=int(info),
        residual_history=tuple(residual_history),
    )


def solve_ac3d_matrix_free_face_types(
    face_data: FaceTypeConductivity,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    use_jacobi: bool = True,
    return_potential: bool = True,
    iteration_callback: Callable[[int], None] | None = None,
    x0: np.ndarray | None = None,
    residual_every: int = 0,
    residual_callback: Callable[[int, float], None] | None = None,
) -> AC3DIterativeResult:
    """Solve with BiCGSTAB using precomputed two-phase face-type conductivities."""

    axis = direction_to_axis(direction)
    dtype = face_data.dtype
    face_conductivities = face_conductivity_arrays_from_types(face_data)
    n_cells = int(np.prod(face_data.shape))
    rhs = matrix_free_rhs_faces(face_conductivities, axis, field_strength_v_m, voxel_size_m)
    rhs_norm = max(np.linalg.norm(rhs), np.finfo(float).eps)

    operator = spla.LinearOperator(
        (n_cells, n_cells),
        matvec=lambda vector: matrix_free_matvec_faces(face_conductivities, vector),
        dtype=dtype,
    )
    preconditioner = None
    if use_jacobi:
        inverse_diagonal = jacobi_inverse_diagonal_faces(face_conductivities)
        preconditioner = spla.LinearOperator(
            (n_cells, n_cells),
            matvec=lambda vector: inverse_diagonal * vector,
            dtype=dtype,
        )

    iterations = 0
    residual_history: list[tuple[int, float]] = []

    def callback(xk: np.ndarray) -> None:
        nonlocal iterations
        iterations += 1
        if iteration_callback is not None:
            iteration_callback(iterations)
        if residual_every > 0 and iterations % residual_every == 0:
            residual_norm = float(np.linalg.norm(matrix_free_matvec_faces(face_conductivities, xk) - rhs) / rhs_norm)
            residual_history.append((iterations, residual_norm))
            if residual_callback is not None:
                residual_callback(iterations, residual_norm)

    initial_guess = None if x0 is None else np.asarray(x0, dtype=dtype).ravel()
    solution, info = _bicgstab(
        operator,
        rhs,
        x0=initial_guess,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner,
        callback=callback,
    )
    residual = matrix_free_matvec_faces(face_conductivities, solution) - rhs
    residual_norm = float(np.linalg.norm(residual) / rhs_norm)
    if not residual_history or residual_history[-1][0] != iterations:
        residual_history.append((iterations, residual_norm))
        if residual_callback is not None:
            residual_callback(iterations, residual_norm)
    potential = solution.reshape(face_data.shape)
    mean_current = mean_current_density_faces(face_conductivities, potential, axis, field_strength_v_m, voxel_size_m)
    effective = mean_current / field_strength_v_m

    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=complex(effective),
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=residual_norm,
        potential=potential if return_potential else None,
        solver="matrix_free_bicgstab_jacobi_face_types" if use_jacobi else "matrix_free_bicgstab_face_types",
        iterations=iterations,
        info=int(info),
        residual_history=tuple(residual_history),
    )


def effective_conductivity_tensor_diagonal(
    conductivity_s_m: np.ndarray,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> dict[str, AC3DResult]:
    """Solve x, y, and z imposed-field directions independently."""

    return {
        direction: solve_ac3d(conductivity_s_m, direction, field_strength_v_m, voxel_size_m)
        for direction in ("x", "y", "z")
    }

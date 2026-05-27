"""CuPy GPU prototype for matrix-free AC3D face-type solves."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from pore_scale_electrical.ac3d_solver import (
    AC3DIterativeResult,
    Direction,
    FaceTypeConductivity,
    axis_to_direction,
    direction_to_axis,
    face_type_conductivity,
)

try:  # pragma: no cover - exercised by GPU smoke tests when CuPy is present.
    import cupy as cp
    from cupyx.scipy.sparse.linalg import LinearOperator
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    LinearOperator = None  # type: ignore[assignment]


ComplexDType = Literal["complex64", "complex128"] | np.dtype | type
GPUPreconditioner = Literal["none", "jacobi", "fft"]
FFTReferenceConductance = Literal["mean-face", "mean-abs", "pore"]


@dataclass(frozen=True)
class GPUFaceTypeConductivity:
    """Face-type conductance data resident on the GPU."""

    face_types: tuple[object, object, object]
    conductance_values: object
    shape: tuple[int, int, int]
    dtype: np.dtype


def cupy_available() -> bool:
    """Return True when CuPy can allocate on at least one CUDA device."""

    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def require_cupy() -> None:
    if not cupy_available():
        raise RuntimeError("CuPy with a CUDA device is required for the GPU AC3D backend")


def parse_gpu_complex_dtype(dtype: ComplexDType) -> np.dtype:
    parsed = np.dtype(dtype)
    if parsed not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError("dtype must be complex64 or complex128")
    return parsed


def to_gpu_face_type_conductivity(face_data: FaceTypeConductivity, dtype: ComplexDType | None = None) -> GPUFaceTypeConductivity:
    """Copy CPU face-type data to GPU."""

    require_cupy()
    complex_dtype = face_data.dtype if dtype is None else parse_gpu_complex_dtype(dtype)
    return GPUFaceTypeConductivity(
        face_types=tuple(cp.asarray(codes, dtype=cp.uint8) for codes in face_data.face_types),  # type: ignore[union-attr]
        conductance_values=cp.asarray(face_data.conductance_values, dtype=complex_dtype),  # type: ignore[union-attr]
        shape=face_data.shape,
        dtype=np.dtype(complex_dtype),
    )


def face_type_conductivity_gpu(
    labels: np.ndarray,
    pore_label: int,
    solid_label: int,
    water_conductivity_s_m: complex,
    solid_conductivity_s_m: complex,
    dtype: ComplexDType = np.complex128,
) -> GPUFaceTypeConductivity:
    """Build CPU compact face types, then copy the compact representation to GPU."""

    face_data = face_type_conductivity(
        labels,
        pore_label,
        solid_label,
        water_conductivity_s_m,
        solid_conductivity_s_m,
        dtype=parse_gpu_complex_dtype(dtype),
    )
    return to_gpu_face_type_conductivity(face_data)


def face_conductivity_arrays_from_types_gpu(face_data: GPUFaceTypeConductivity) -> tuple[object, object, object]:
    """Expand compact GPU face-type codes into per-face conductance arrays."""

    require_cupy()
    values = cp.asarray(face_data.conductance_values, dtype=face_data.dtype)  # type: ignore[union-attr]
    return tuple(values[codes] for codes in face_data.face_types)  # type: ignore[return-value]


def _accumulate_axis_flux_gpu(y: object, x: object, g_forward: object, axis: int) -> None:
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


def matrix_free_matvec_faces_gpu(face_conductivities: tuple[object, object, object], vector: object) -> object:
    """Apply the periodic AC3D operator on GPU using positive-face conductivities."""

    require_cupy()
    shape = face_conductivities[0].shape
    dtype = cp.result_type(*(g.dtype for g in face_conductivities), vector)  # type: ignore[union-attr]
    x = cp.asarray(vector, dtype=dtype).reshape(shape)  # type: ignore[union-attr]
    y = cp.zeros_like(x)  # type: ignore[union-attr]
    for axis, g_forward in enumerate(face_conductivities):
        _accumulate_axis_flux_gpu(y, x, cp.asarray(g_forward, dtype=dtype), axis)  # type: ignore[union-attr]
    flat = y.ravel()
    flat[0] = cp.asarray(vector, dtype=dtype)[0]  # type: ignore[union-attr]
    return flat


def matrix_free_matvec_face_types_gpu(face_data: GPUFaceTypeConductivity, vector: object) -> object:
    """Apply the operator from compact GPU face-type data."""

    return matrix_free_matvec_faces_gpu(face_conductivity_arrays_from_types_gpu(face_data), vector)


def matrix_free_rhs_faces_gpu(
    face_conductivities: tuple[object, object, object],
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> object:
    """Right-hand side on GPU for precomputed positive-face conductivities."""

    require_cupy()
    axis = direction_to_axis(direction)
    g = face_conductivities[axis]
    macro_divergence = cp.zeros(g.shape, dtype=g.dtype)  # type: ignore[union-attr]
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


def jacobi_inverse_diagonal_faces_gpu(face_conductivities: tuple[object, object, object]) -> object:
    """Jacobi inverse diagonal on GPU from positive-face conductivities."""

    require_cupy()
    shape = face_conductivities[0].shape
    dtype = cp.result_type(*(g.dtype for g in face_conductivities))  # type: ignore[union-attr]
    diagonal = cp.zeros(shape, dtype=dtype)  # type: ignore[union-attr]
    for axis, g_forward in enumerate(face_conductivities):
        g = cp.asarray(g_forward, dtype=dtype)  # type: ignore[union-attr]
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
    flat = diagonal.ravel()
    flat[0] = 1.0
    inverse = cp.zeros_like(flat)  # type: ignore[union-attr]
    mask = cp.abs(flat) > np.finfo(float).tiny  # type: ignore[union-attr]
    inverse[mask] = 1.0 / flat[mask]
    return inverse


def poisson_reference_conductance_gpu(
    face_data: GPUFaceTypeConductivity,
    mode: FFTReferenceConductance = "mean-face",
) -> complex:
    """Choose a scalar conductance for the constant-coefficient FFT preconditioner."""

    require_cupy()
    values = cp.asarray(face_data.conductance_values, dtype=face_data.dtype)  # type: ignore[union-attr]
    if mode == "pore":
        reference = values[0]
    elif mode in ("mean-face", "mean-abs"):
        counts = cp.zeros(3, dtype=cp.float64)  # type: ignore[union-attr]
        for codes in face_data.face_types:
            counts += cp.bincount(codes.ravel(), minlength=3).astype(cp.float64)  # type: ignore[union-attr]
        if mode == "mean-face":
            reference = cp.sum(counts.astype(values.dtype) * values) / cp.sum(counts)  # type: ignore[union-attr]
        else:
            reference = cp.asarray(cp.sum(counts * cp.abs(values)) / cp.sum(counts), dtype=values.dtype)  # type: ignore[union-attr]
    else:
        raise ValueError("FFT reference mode must be mean-face, mean-abs, or pore")

    reference_complex = complex(reference.get())  # type: ignore[union-attr]
    if abs(reference_complex) <= np.finfo(float).tiny:
        fallback = complex(values[cp.argmax(cp.abs(values))].get())  # type: ignore[union-attr]
        if abs(fallback) <= np.finfo(float).tiny:
            raise ValueError("cannot build FFT preconditioner from near-zero conductances")
        return fallback
    return reference_complex


def periodic_laplacian_eigenvalues_gpu(shape: tuple[int, int, int], dtype: np.dtype) -> object:
    """Eigenvalues of the positive periodic 7-point Laplacian stencil."""

    require_cupy()
    real_dtype = cp.float32 if np.dtype(dtype) == np.dtype(np.complex64) else cp.float64  # type: ignore[union-attr]
    terms = []
    for axis, n in enumerate(shape):
        frequencies = cp.arange(n, dtype=real_dtype)  # type: ignore[union-attr]
        values = 2.0 - 2.0 * cp.cos(2.0 * np.pi * frequencies / n)  # type: ignore[union-attr]
        reshape = [1, 1, 1]
        reshape[axis] = n
        terms.append(values.reshape(tuple(reshape)))
    return terms[0] + terms[1] + terms[2]


def fft_poisson_inverse_denominator_gpu(
    shape: tuple[int, int, int],
    dtype: np.dtype,
    reference_conductance: complex,
) -> object:
    """Return reciprocal spectral denominator for a constant periodic Poisson solve."""

    require_cupy()
    denominator = periodic_laplacian_eigenvalues_gpu(shape, dtype) * cp.asarray(reference_conductance, dtype=dtype)  # type: ignore[union-attr]
    inverse = cp.zeros(shape, dtype=dtype)  # type: ignore[union-attr]
    mask = cp.abs(denominator) > np.finfo(float).tiny  # type: ignore[union-attr]
    inverse[mask] = 1.0 / denominator[mask]
    inverse[(0, 0, 0)] = 0.0
    return inverse


def make_fft_poisson_preconditioner_gpu(
    face_data: GPUFaceTypeConductivity,
    reference_mode: FFTReferenceConductance = "mean-face",
) -> tuple[Callable[[object], object], complex]:
    """Build an FFT inverse for a scalar periodic Poisson approximation."""

    require_cupy()
    reference = poisson_reference_conductance_gpu(face_data, reference_mode)
    inverse_denominator = fft_poisson_inverse_denominator_gpu(face_data.shape, face_data.dtype, reference)

    def apply(vector: object) -> object:
        rhs = cp.asarray(vector, dtype=face_data.dtype).reshape(face_data.shape)  # type: ignore[union-attr]
        gauge_value = rhs.ravel()[0]
        zero_mean_rhs = rhs - cp.mean(rhs)  # type: ignore[union-attr]
        solution = cp.fft.ifftn(cp.fft.fftn(zero_mean_rhs) * inverse_denominator)  # type: ignore[union-attr]
        flat = solution.ravel()
        flat[0] = gauge_value
        return flat

    return apply, reference


def mean_current_density_faces_gpu(
    face_conductivities: tuple[object, object, object],
    potential: object,
    direction: Direction,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> complex:
    """Compute volume-averaged current density on GPU."""

    require_cupy()
    axis = direction_to_axis(direction)
    g = face_conductivities[axis]
    u = cp.asarray(potential, dtype=g.dtype).reshape(g.shape)  # type: ignore[union-attr]
    if axis == 0:
        current_sum = cp.sum(g[:-1, :, :] * (field_strength_v_m - (u[1:, :, :] - u[:-1, :, :]) / voxel_size_m))  # type: ignore[union-attr]
        current_sum += cp.sum(g[-1, :, :] * (field_strength_v_m - (u[0, :, :] - u[-1, :, :]) / voxel_size_m))  # type: ignore[union-attr]
    elif axis == 1:
        current_sum = cp.sum(g[:, :-1, :] * (field_strength_v_m - (u[:, 1:, :] - u[:, :-1, :]) / voxel_size_m))  # type: ignore[union-attr]
        current_sum += cp.sum(g[:, -1, :] * (field_strength_v_m - (u[:, 0, :] - u[:, -1, :]) / voxel_size_m))  # type: ignore[union-attr]
    else:
        current_sum = cp.sum(g[:, :, :-1] * (field_strength_v_m - (u[:, :, 1:] - u[:, :, :-1]) / voxel_size_m))  # type: ignore[union-attr]
        current_sum += cp.sum(g[:, :, -1] * (field_strength_v_m - (u[:, :, 0] - u[:, :, -1]) / voxel_size_m))  # type: ignore[union-attr]
    return complex(cp.asnumpy(current_sum / np.prod(g.shape)))  # type: ignore[union-attr]


def _gpu_bicgstab(
    operator: object,
    rhs: object,
    *,
    x0: object | None,
    rtol: float,
    atol: float,
    maxiter: int | None,
    preconditioner: Callable[[object], object] | None,
    iteration_callback: Callable[[int], None] | None,
    residual_every: int,
    residual_callback: Callable[[int, float], None] | None,
) -> tuple[object, int, int, float, tuple[tuple[int, float], ...]]:
    """Small BiCGSTAB implementation for CuPy arrays.

    CuPy 13 exposes CG/CGS/GMRES but not BiCGSTAB, so the prototype keeps the
    same Krylov family as the CPU solver with this local implementation.
    """

    require_cupy()
    n = rhs.size
    maxiter = n if maxiter is None else maxiter
    dtype = rhs.dtype
    x = cp.zeros(n, dtype=dtype) if x0 is None else cp.asarray(x0, dtype=dtype).ravel().copy()  # type: ignore[union-attr]
    r = rhs - operator.matvec(x)
    r_hat = r.copy()
    v = cp.zeros_like(rhs)  # type: ignore[union-attr]
    p = cp.zeros_like(rhs)  # type: ignore[union-attr]
    rho_old = dtype.type(1.0 + 0.0j)
    alpha = dtype.type(1.0 + 0.0j)
    omega = dtype.type(1.0 + 0.0j)
    rhs_norm = float(cp.linalg.norm(rhs).get())  # type: ignore[union-attr]
    rhs_norm = max(rhs_norm, np.finfo(float).eps)
    tolerance = atol + rtol * rhs_norm
    residual_norm_abs = float(cp.linalg.norm(r).get())  # type: ignore[union-attr]
    relative_residual = residual_norm_abs / rhs_norm
    history: list[tuple[int, float]] = []

    def apply_preconditioner(vector: object) -> object:
        return vector if preconditioner is None else preconditioner(vector)

    if residual_norm_abs <= tolerance:
        return x, 0, 0, relative_residual, ((0, relative_residual),)

    info = int(maxiter)
    tiny = np.finfo(np.float32 if dtype == cp.complex64 else np.float64).tiny  # type: ignore[union-attr]
    iterations = 0
    for iterations in range(1, maxiter + 1):
        rho_new = cp.vdot(r_hat, r)  # type: ignore[union-attr]
        if float(cp.abs(rho_new).get()) <= tiny:  # type: ignore[union-attr]
            info = -10
            break
        beta = (rho_new / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        p_hat = apply_preconditioner(p)
        v = operator.matvec(p_hat)
        denominator = cp.vdot(r_hat, v)  # type: ignore[union-attr]
        if float(cp.abs(denominator).get()) <= tiny:  # type: ignore[union-attr]
            info = -11
            break
        alpha = rho_new / denominator
        s = r - alpha * v
        s_norm_abs = float(cp.linalg.norm(s).get())  # type: ignore[union-attr]
        if s_norm_abs <= tolerance:
            x = x + alpha * p_hat
            relative_residual = s_norm_abs / rhs_norm
            info = 0
            if residual_every > 0 and iterations % residual_every == 0:
                history.append((iterations, relative_residual))
                if residual_callback is not None:
                    residual_callback(iterations, relative_residual)
            if iteration_callback is not None:
                iteration_callback(iterations)
            break
        s_hat = apply_preconditioner(s)
        t = operator.matvec(s_hat)
        tt = cp.vdot(t, t)  # type: ignore[union-attr]
        if float(cp.abs(tt).get()) <= tiny:  # type: ignore[union-attr]
            info = -12
            break
        omega = cp.vdot(t, s) / tt  # type: ignore[union-attr]
        x = x + alpha * p_hat + omega * s_hat
        r = s - omega * t
        residual_norm_abs = float(cp.linalg.norm(r).get())  # type: ignore[union-attr]
        relative_residual = residual_norm_abs / rhs_norm
        if residual_every > 0 and iterations % residual_every == 0:
            history.append((iterations, relative_residual))
            if residual_callback is not None:
                residual_callback(iterations, relative_residual)
        if iteration_callback is not None:
            iteration_callback(iterations)
        if residual_norm_abs <= tolerance:
            info = 0
            break
        if float(cp.abs(omega).get()) <= tiny:  # type: ignore[union-attr]
            info = -13
            break
        rho_old = rho_new

    if not history or history[-1][0] != iterations:
        history.append((iterations, relative_residual))
        if residual_callback is not None:
            residual_callback(iterations, relative_residual)
    return x, int(info), int(iterations), float(relative_residual), tuple(history)


def solve_ac3d_matrix_free_gpu_face_types(
    face_data: GPUFaceTypeConductivity,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-8,
    atol: float = 0.0,
    maxiter: int | None = None,
    use_jacobi: bool = True,
    preconditioner: GPUPreconditioner | None = None,
    fft_reference: FFTReferenceConductance = "mean-face",
    return_potential: bool = True,
    iteration_callback: Callable[[int], None] | None = None,
    x0: np.ndarray | object | None = None,
    residual_every: int = 0,
    residual_callback: Callable[[int, float], None] | None = None,
) -> AC3DIterativeResult:
    """Solve with GPU BiCGSTAB using compact two-phase face types."""

    require_cupy()
    preconditioner_name: GPUPreconditioner = "jacobi" if preconditioner is None and use_jacobi else "none"
    if preconditioner is not None:
        preconditioner_name = preconditioner
    axis = direction_to_axis(direction)
    face_conductivities = face_conductivity_arrays_from_types_gpu(face_data)
    n_cells = int(np.prod(face_data.shape))
    rhs = matrix_free_rhs_faces_gpu(face_conductivities, axis, field_strength_v_m, voxel_size_m)
    operator = LinearOperator(  # type: ignore[operator]
        (n_cells, n_cells),
        matvec=lambda vector: matrix_free_matvec_faces_gpu(face_conductivities, vector),
        dtype=face_data.dtype,
    )
    preconditioner_apply: Callable[[object], object] | None = None
    fft_reference_value: complex | None = None
    if preconditioner_name == "jacobi":
        inverse_diagonal = jacobi_inverse_diagonal_faces_gpu(face_conductivities)
        preconditioner_apply = lambda vector: inverse_diagonal * vector
    elif preconditioner_name == "fft":
        preconditioner_apply, fft_reference_value = make_fft_poisson_preconditioner_gpu(face_data, fft_reference)
    elif preconditioner_name == "none":
        preconditioner_apply = None
    else:
        raise ValueError("preconditioner must be none, jacobi, or fft")
    initial_guess = None if x0 is None else cp.asarray(x0, dtype=face_data.dtype).ravel()  # type: ignore[union-attr]
    solution, info, iterations, relative_residual, residual_history = _gpu_bicgstab(
        operator,
        rhs,
        x0=initial_guess,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        preconditioner=preconditioner_apply,
        iteration_callback=iteration_callback,
        residual_every=residual_every,
        residual_callback=residual_callback,
    )
    mean_current = mean_current_density_faces_gpu(face_conductivities, solution, axis, field_strength_v_m, voxel_size_m)
    effective = mean_current / field_strength_v_m
    potential = cp.asnumpy(solution.reshape(face_data.shape)) if return_potential else None  # type: ignore[union-attr]
    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=complex(effective),
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=relative_residual,
        potential=potential,
        solver=(
            "gpu_bicgstab_fft_poisson_face_types"
            if preconditioner_name == "fft"
            else "gpu_bicgstab_jacobi_face_types"
            if preconditioner_name == "jacobi"
            else "gpu_bicgstab_face_types"
        ),
        iterations=iterations,
        info=info,
        residual_history=residual_history,
    )


def gpu_memory_info() -> dict[str, int]:
    """Return current CUDA free/total memory and CuPy memory-pool usage."""

    require_cupy()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()  # type: ignore[union-attr]
    pool = cp.get_default_memory_pool()  # type: ignore[union-attr]
    pinned_pool = cp.get_default_pinned_memory_pool()  # type: ignore[union-attr]
    return {
        "cuda_free_bytes": int(free_bytes),
        "cuda_total_bytes": int(total_bytes),
        "cupy_pool_used_bytes": int(pool.used_bytes()),
        "cupy_pool_total_bytes": int(pool.total_bytes()),
        "cupy_pinned_pool_free_blocks": int(pinned_pool.n_free_blocks()),
    }


def synchronize_gpu() -> None:
    """Synchronize the current CUDA device if CuPy is available."""

    require_cupy()
    cp.cuda.Stream.null.synchronize()  # type: ignore[union-attr]


def timed_gpu_call(func: Callable[[], object]) -> tuple[object, float]:
    """Run a callable and time it with CUDA synchronization around the call."""

    synchronize_gpu()
    start = time.perf_counter()
    result = func()
    synchronize_gpu()
    return result, time.perf_counter() - start

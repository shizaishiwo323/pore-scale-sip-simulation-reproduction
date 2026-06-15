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


_WEIGHTED_JACOBI_KERNEL = None
_MEAN_CURRENT_KERNEL = None
_RED_BLACK_SOR_KERNEL = None
_COMPACT_MATVEC_KERNEL = None
_COMPACT_RHS_KERNEL = None
_BILINEAR_DOT_KERNEL = None
_NORM2_KERNEL = None
_COCG_UPDATE_KERNEL = None
_COCG_P_UPDATE_KERNEL = None
_HERMITIAN_DOT_KERNEL = None
_RHS_HERMITIAN_DOT_KERNEL = None
_BICGSTAB_P_UPDATE_KERNEL = None
_BICGSTAB_ALPHA_KERNEL = None
_BICGSTAB_OMEGA_KERNEL = None


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


def _weighted_jacobi_kernel() -> object:
    global _WEIGHTED_JACOBI_KERNEL
    require_cupy()
    if _WEIGHTED_JACOBI_KERNEL is not None:
        return _WEIGHTED_JACOBI_KERNEL
    code = r'''
    extern "C" __global__
    void weighted_jacobi_step(
        const unsigned char* c0,
        const unsigned char* c1,
        const unsigned char* c2,
        const float* val_r,
        const float* val_i,
        const float2* x,
        float2* x_new,
        float* residual2,
        const long n0,
        const long n1,
        const long n2,
        const int drive_axis,
        const float field_times_voxel,
        const float omega
    ) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        const long n = n0 * n1 * n2;
        if (idx >= n) {
            return;
        }
        if (idx == 0) {
            const float2 xi = x[idx];
            residual2[idx] = xi.x * xi.x + xi.y * xi.y;
            x_new[idx] = make_float2(0.0f, 0.0f);
            return;
        }

        const long i0 = idx / (n1 * n2);
        const long rem = idx - i0 * n1 * n2;
        const long i1 = rem / n2;
        const long i2 = rem - i1 * n2;

        const long p0 = (((i0 + 1) % n0) * n1 + i1) * n2 + i2;
        const long m0 = (((i0 + n0 - 1) % n0) * n1 + i1) * n2 + i2;
        const long p1 = (i0 * n1 + ((i1 + 1) % n1)) * n2 + i2;
        const long m1 = (i0 * n1 + ((i1 + n1 - 1) % n1)) * n2 + i2;
        const long p2 = (i0 * n1 + i1) * n2 + ((i2 + 1) % n2);
        const long m2 = (i0 * n1 + i1) * n2 + ((i2 + n2 - 1) % n2);

        const float2 xi = x[idx];
        float ax_r = 0.0f;
        float ax_i = 0.0f;
        float diag_r = 0.0f;
        float diag_i = 0.0f;
        float rhs_r = 0.0f;
        float rhs_i = 0.0f;

        #define ADD_FACE(CODE, NB, SIGN_RHS) { \
            const int code = (int)(CODE); \
            const float gr = val_r[code]; \
            const float gi = val_i[code]; \
            const float2 xj = x[(NB)]; \
            const float dr = xi.x - xj.x; \
            const float di = xi.y - xj.y; \
            ax_r += gr * dr - gi * di; \
            ax_i += gr * di + gi * dr; \
            diag_r += gr; \
            diag_i += gi; \
            if ((SIGN_RHS) != 0) { \
                rhs_r += (float)(SIGN_RHS) * field_times_voxel * gr; \
                rhs_i += (float)(SIGN_RHS) * field_times_voxel * gi; \
            } \
        }

        ADD_FACE(c0[idx], p0, drive_axis == 0 ? -1 : 0);
        ADD_FACE(c0[m0], m0, drive_axis == 0 ? 1 : 0);
        ADD_FACE(c1[idx], p1, drive_axis == 1 ? -1 : 0);
        ADD_FACE(c1[m1], m1, drive_axis == 1 ? 1 : 0);
        ADD_FACE(c2[idx], p2, drive_axis == 2 ? -1 : 0);
        ADD_FACE(c2[m2], m2, drive_axis == 2 ? 1 : 0);
        #undef ADD_FACE

        const float rr = rhs_r - ax_r;
        const float ri = rhs_i - ax_i;
        residual2[idx] = rr * rr + ri * ri;
        const float denom = diag_r * diag_r + diag_i * diag_i;
        if (denom <= 1.17549435e-38f) {
            x_new[idx] = xi;
            return;
        }
        const float corr_r = (rr * diag_r + ri * diag_i) / denom;
        const float corr_i = (ri * diag_r - rr * diag_i) / denom;
        x_new[idx] = make_float2(xi.x + omega * corr_r, xi.y + omega * corr_i);
    }
    '''
    _WEIGHTED_JACOBI_KERNEL = cp.RawKernel(code, "weighted_jacobi_step")  # type: ignore[union-attr]
    return _WEIGHTED_JACOBI_KERNEL


def _mean_current_kernel() -> object:
    global _MEAN_CURRENT_KERNEL
    require_cupy()
    if _MEAN_CURRENT_KERNEL is not None:
        return _MEAN_CURRENT_KERNEL
    code = r'''
    extern "C" __global__
    void mean_current_faces(
        const unsigned char* c0,
        const unsigned char* c1,
        const unsigned char* c2,
        const float* val_r,
        const float* val_i,
        const float2* x,
        float* current_r,
        float* current_i,
        const long n0,
        const long n1,
        const long n2,
        const int drive_axis,
        const float field_strength,
        const float voxel_size
    ) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        const long n = n0 * n1 * n2;
        if (idx >= n) {
            return;
        }
        const long i0 = idx / (n1 * n2);
        const long rem = idx - i0 * n1 * n2;
        const long i1 = rem / n2;
        const long i2 = rem - i1 * n2;
        long nb;
        unsigned char code;
        if (drive_axis == 0) {
            nb = (((i0 + 1) % n0) * n1 + i1) * n2 + i2;
            code = c0[idx];
        } else if (drive_axis == 1) {
            nb = (i0 * n1 + ((i1 + 1) % n1)) * n2 + i2;
            code = c1[idx];
        } else {
            nb = (i0 * n1 + i1) * n2 + ((i2 + 1) % n2);
            code = c2[idx];
        }
        const int c = (int)code;
        const float gr = val_r[c];
        const float gi = val_i[c];
        const float2 xi = x[idx];
        const float2 xj = x[nb];
        const float er = field_strength - (xj.x - xi.x) / voxel_size;
        const float ei = - (xj.y - xi.y) / voxel_size;
        current_r[idx] = gr * er - gi * ei;
        current_i[idx] = gr * ei + gi * er;
    }
    '''
    _MEAN_CURRENT_KERNEL = cp.RawKernel(code, "mean_current_faces")  # type: ignore[union-attr]
    return _MEAN_CURRENT_KERNEL


def _red_black_sor_kernel() -> object:
    global _RED_BLACK_SOR_KERNEL
    require_cupy()
    if _RED_BLACK_SOR_KERNEL is not None:
        return _RED_BLACK_SOR_KERNEL
    code = r'''
    extern "C" __global__
    void red_black_sor_step(
        const unsigned char* c0,
        const unsigned char* c1,
        const unsigned char* c2,
        const float* val_r,
        const float* val_i,
        float2* x,
        float* residual2,
        const long n0,
        const long n1,
        const long n2,
        const int drive_axis,
        const float field_times_voxel,
        const float omega,
        const int parity
    ) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        const long n = n0 * n1 * n2;
        if (idx >= n) {
            return;
        }

        const long i0 = idx / (n1 * n2);
        const long rem = idx - i0 * n1 * n2;
        const long i1 = rem / n2;
        const long i2 = rem - i1 * n2;
        const int cell_parity = (int)((i0 + i1 + i2) & 1L);
        if (parity >= 0 && cell_parity != parity) {
            return;
        }
        if (idx == 0) {
            const float2 xi0 = x[idx];
            if (parity < 0) {
                residual2[idx] = xi0.x * xi0.x + xi0.y * xi0.y;
            } else {
                x[idx] = make_float2(0.0f, 0.0f);
            }
            return;
        }

        const long p0 = (((i0 + 1) % n0) * n1 + i1) * n2 + i2;
        const long m0 = (((i0 + n0 - 1) % n0) * n1 + i1) * n2 + i2;
        const long p1 = (i0 * n1 + ((i1 + 1) % n1)) * n2 + i2;
        const long m1 = (i0 * n1 + ((i1 + n1 - 1) % n1)) * n2 + i2;
        const long p2 = (i0 * n1 + i1) * n2 + ((i2 + 1) % n2);
        const long m2 = (i0 * n1 + i1) * n2 + ((i2 + n2 - 1) % n2);

        const float2 xi = x[idx];
        float ax_r = 0.0f;
        float ax_i = 0.0f;
        float diag_r = 0.0f;
        float diag_i = 0.0f;
        float rhs_r = 0.0f;
        float rhs_i = 0.0f;

        #define ADD_FACE(CODE, NB, SIGN_RHS) { \
            const int code = (int)(CODE); \
            const float gr = val_r[code]; \
            const float gi = val_i[code]; \
            const float2 xj = x[(NB)]; \
            const float dr = xi.x - xj.x; \
            const float di = xi.y - xj.y; \
            ax_r += gr * dr - gi * di; \
            ax_i += gr * di + gi * dr; \
            diag_r += gr; \
            diag_i += gi; \
            if ((SIGN_RHS) != 0) { \
                rhs_r += (float)(SIGN_RHS) * field_times_voxel * gr; \
                rhs_i += (float)(SIGN_RHS) * field_times_voxel * gi; \
            } \
        }

        ADD_FACE(c0[idx], p0, drive_axis == 0 ? -1 : 0);
        ADD_FACE(c0[m0], m0, drive_axis == 0 ? 1 : 0);
        ADD_FACE(c1[idx], p1, drive_axis == 1 ? -1 : 0);
        ADD_FACE(c1[m1], m1, drive_axis == 1 ? 1 : 0);
        ADD_FACE(c2[idx], p2, drive_axis == 2 ? -1 : 0);
        ADD_FACE(c2[m2], m2, drive_axis == 2 ? 1 : 0);
        #undef ADD_FACE

        const float rr = rhs_r - ax_r;
        const float ri = rhs_i - ax_i;
        if (parity < 0) {
            residual2[idx] = rr * rr + ri * ri;
            return;
        }
        const float denom = diag_r * diag_r + diag_i * diag_i;
        if (denom <= 1.17549435e-38f) {
            return;
        }
        const float corr_r = (rr * diag_r + ri * diag_i) / denom;
        const float corr_i = (ri * diag_r - rr * diag_i) / denom;
        x[idx] = make_float2(xi.x + omega * corr_r, xi.y + omega * corr_i);
    }
    '''
    _RED_BLACK_SOR_KERNEL = cp.RawKernel(code, "red_black_sor_step")  # type: ignore[union-attr]
    return _RED_BLACK_SOR_KERNEL


def _compact_matvec_kernel() -> object:
    global _COMPACT_MATVEC_KERNEL
    require_cupy()
    if _COMPACT_MATVEC_KERNEL is not None:
        return _COMPACT_MATVEC_KERNEL
    code = r'''
    extern "C" __global__
    void compact_matvec(
        const unsigned char* c0,
        const unsigned char* c1,
        const unsigned char* c2,
        const float* val_r,
        const float* val_i,
        const float2* x,
        float2* y,
        const long n0,
        const long n1,
        const long n2
    ) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        const long n = n0 * n1 * n2;
        if (idx >= n) {
            return;
        }
        if (idx == 0) {
            y[idx] = x[idx];
            return;
        }
        const long i0 = idx / (n1 * n2);
        const long rem = idx - i0 * n1 * n2;
        const long i1 = rem / n2;
        const long i2 = rem - i1 * n2;
        const long p0 = (((i0 + 1) % n0) * n1 + i1) * n2 + i2;
        const long m0 = (((i0 + n0 - 1) % n0) * n1 + i1) * n2 + i2;
        const long p1 = (i0 * n1 + ((i1 + 1) % n1)) * n2 + i2;
        const long m1 = (i0 * n1 + ((i1 + n1 - 1) % n1)) * n2 + i2;
        const long p2 = (i0 * n1 + i1) * n2 + ((i2 + 1) % n2);
        const long m2 = (i0 * n1 + i1) * n2 + ((i2 + n2 - 1) % n2);
        const float2 xi = x[idx];
        float yr = 0.0f;
        float yi = 0.0f;
        #define ADD_FACE(CODE, NB) { \
            const int code = (int)(CODE); \
            const float gr = val_r[code]; \
            const float gi = val_i[code]; \
            const float2 xj = x[(NB)]; \
            const float dr = xi.x - xj.x; \
            const float di = xi.y - xj.y; \
            yr += gr * dr - gi * di; \
            yi += gr * di + gi * dr; \
        }
        ADD_FACE(c0[idx], p0);
        ADD_FACE(c0[m0], m0);
        ADD_FACE(c1[idx], p1);
        ADD_FACE(c1[m1], m1);
        ADD_FACE(c2[idx], p2);
        ADD_FACE(c2[m2], m2);
        #undef ADD_FACE
        y[idx] = make_float2(yr, yi);
    }
    '''
    _COMPACT_MATVEC_KERNEL = cp.RawKernel(code, "compact_matvec")  # type: ignore[union-attr]
    return _COMPACT_MATVEC_KERNEL


def _compact_rhs_kernel() -> object:
    global _COMPACT_RHS_KERNEL
    require_cupy()
    if _COMPACT_RHS_KERNEL is not None:
        return _COMPACT_RHS_KERNEL
    code = r'''
    extern "C" __global__
    void compact_rhs(
        const unsigned char* c0,
        const unsigned char* c1,
        const unsigned char* c2,
        const float* val_r,
        const float* val_i,
        float2* rhs,
        const long n0,
        const long n1,
        const long n2,
        const int drive_axis,
        const float field_times_voxel
    ) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        const long n = n0 * n1 * n2;
        if (idx >= n) {
            return;
        }
        if (idx == 0) {
            rhs[idx] = make_float2(0.0f, 0.0f);
            return;
        }
        const long i0 = idx / (n1 * n2);
        const long rem = idx - i0 * n1 * n2;
        const long i1 = rem / n2;
        const long i2 = rem - i1 * n2;
        const long m0 = (((i0 + n0 - 1) % n0) * n1 + i1) * n2 + i2;
        const long m1 = (i0 * n1 + ((i1 + n1 - 1) % n1)) * n2 + i2;
        const long m2 = (i0 * n1 + i1) * n2 + ((i2 + n2 - 1) % n2);
        float rr = 0.0f;
        float ri = 0.0f;
        #define ADD_RHS(CODE, SIGN_RHS) { \
            const int code = (int)(CODE); \
            const float gr = val_r[code]; \
            const float gi = val_i[code]; \
            rr += (float)(SIGN_RHS) * field_times_voxel * gr; \
            ri += (float)(SIGN_RHS) * field_times_voxel * gi; \
        }
        if (drive_axis == 0) {
            ADD_RHS(c0[idx], -1);
            ADD_RHS(c0[m0], 1);
        } else if (drive_axis == 1) {
            ADD_RHS(c1[idx], -1);
            ADD_RHS(c1[m1], 1);
        } else {
            ADD_RHS(c2[idx], -1);
            ADD_RHS(c2[m2], 1);
        }
        #undef ADD_RHS
        rhs[idx] = make_float2(rr, ri);
    }
    '''
    _COMPACT_RHS_KERNEL = cp.RawKernel(code, "compact_rhs")  # type: ignore[union-attr]
    return _COMPACT_RHS_KERNEL


def _bilinear_dot_kernel() -> object:
    global _BILINEAR_DOT_KERNEL
    require_cupy()
    if _BILINEAR_DOT_KERNEL is not None:
        return _BILINEAR_DOT_KERNEL
    code = r'''
    extern "C" __global__
    void bilinear_dot_blocks(const float2* a, const float2* b, float2* out, const long n) {
        __shared__ float sr[256];
        __shared__ float si[256];
        const int tid = threadIdx.x;
        long idx = (long)blockDim.x * blockIdx.x + tid;
        float rr = 0.0f;
        float ri = 0.0f;
        if (idx < n) {
            const float2 av = a[idx];
            const float2 bv = b[idx];
            rr = av.x * bv.x - av.y * bv.y;
            ri = av.x * bv.y + av.y * bv.x;
        }
        sr[tid] = rr;
        si[tid] = ri;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sr[tid] += sr[tid + stride];
                si[tid] += si[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[blockIdx.x] = make_float2(sr[0], si[0]);
        }
    }
    '''
    _BILINEAR_DOT_KERNEL = cp.RawKernel(code, "bilinear_dot_blocks")  # type: ignore[union-attr]
    return _BILINEAR_DOT_KERNEL


def _norm2_kernel() -> object:
    global _NORM2_KERNEL
    require_cupy()
    if _NORM2_KERNEL is not None:
        return _NORM2_KERNEL
    code = r'''
    extern "C" __global__
    void norm2_blocks(const float2* a, float* out, const long n) {
        __shared__ float sr[256];
        const int tid = threadIdx.x;
        long idx = (long)blockDim.x * blockIdx.x + tid;
        float rr = 0.0f;
        if (idx < n) {
            const float2 av = a[idx];
            rr = av.x * av.x + av.y * av.y;
        }
        sr[tid] = rr;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sr[tid] += sr[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[blockIdx.x] = sr[0];
        }
    }
    '''
    _NORM2_KERNEL = cp.RawKernel(code, "norm2_blocks")  # type: ignore[union-attr]
    return _NORM2_KERNEL


def _cocg_update_kernel() -> object:
    global _COCG_UPDATE_KERNEL
    require_cupy()
    if _COCG_UPDATE_KERNEL is not None:
        return _COCG_UPDATE_KERNEL
    code = r'''
    extern "C" __global__
    void cocg_update(float2* x, float2* r, const float2* p, const float2* q, const float ar, const float ai, const long n) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float2 pv = p[idx];
        const float2 qv = q[idx];
        const float axp_r = ar * pv.x - ai * pv.y;
        const float axp_i = ar * pv.y + ai * pv.x;
        const float axq_r = ar * qv.x - ai * qv.y;
        const float axq_i = ar * qv.y + ai * qv.x;
        float2 xv = x[idx];
        float2 rv = r[idx];
        xv.x += axp_r;
        xv.y += axp_i;
        rv.x -= axq_r;
        rv.y -= axq_i;
        if (idx == 0) {
            xv = make_float2(0.0f, 0.0f);
            rv = make_float2(0.0f, 0.0f);
        }
        x[idx] = xv;
        r[idx] = rv;
    }
    '''
    _COCG_UPDATE_KERNEL = cp.RawKernel(code, "cocg_update")  # type: ignore[union-attr]
    return _COCG_UPDATE_KERNEL


def _cocg_p_update_kernel() -> object:
    global _COCG_P_UPDATE_KERNEL
    require_cupy()
    if _COCG_P_UPDATE_KERNEL is not None:
        return _COCG_P_UPDATE_KERNEL
    code = r'''
    extern "C" __global__
    void cocg_p_update(float2* p, const float2* r, const float br, const float bi, const long n) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float2 pv = p[idx];
        const float2 rv = r[idx];
        const float bp_r = br * pv.x - bi * pv.y;
        const float bp_i = br * pv.y + bi * pv.x;
        p[idx] = make_float2(rv.x + bp_r, rv.y + bp_i);
    }
    '''
    _COCG_P_UPDATE_KERNEL = cp.RawKernel(code, "cocg_p_update")  # type: ignore[union-attr]
    return _COCG_P_UPDATE_KERNEL


def _hermitian_dot_kernel() -> object:
    global _HERMITIAN_DOT_KERNEL
    require_cupy()
    if _HERMITIAN_DOT_KERNEL is not None:
        return _HERMITIAN_DOT_KERNEL
    code = r'''
    extern "C" __global__
    void hermitian_dot_blocks(const float2* a, const float2* b, float2* out, const long n) {
        __shared__ float sr[256];
        __shared__ float si[256];
        const int tid = threadIdx.x;
        long idx = (long)blockDim.x * blockIdx.x + tid;
        float rr = 0.0f;
        float ri = 0.0f;
        if (idx < n) {
            const float2 av = a[idx];
            const float2 bv = b[idx];
            rr = av.x * bv.x + av.y * bv.y;
            ri = av.x * bv.y - av.y * bv.x;
        }
        sr[tid] = rr;
        si[tid] = ri;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sr[tid] += sr[tid + stride];
                si[tid] += si[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[blockIdx.x] = make_float2(sr[0], si[0]);
        }
    }
    '''
    _HERMITIAN_DOT_KERNEL = cp.RawKernel(code, "hermitian_dot_blocks")  # type: ignore[union-attr]
    return _HERMITIAN_DOT_KERNEL


def _rhs_hermitian_dot_kernel() -> object:
    global _RHS_HERMITIAN_DOT_KERNEL
    require_cupy()
    if _RHS_HERMITIAN_DOT_KERNEL is not None:
        return _RHS_HERMITIAN_DOT_KERNEL
    code = r'''
    extern "C" __global__
    void rhs_hermitian_dot_blocks(
        const unsigned char* c0,
        const unsigned char* c1,
        const unsigned char* c2,
        const float* val_r,
        const float* val_i,
        const float2* b,
        float2* out,
        const long n0,
        const long n1,
        const long n2,
        const int drive_axis,
        const float field_times_voxel
    ) {
        __shared__ float sr[256];
        __shared__ float si[256];
        const int tid = threadIdx.x;
        const long idx = (long)blockDim.x * blockIdx.x + tid;
        const long n = n0 * n1 * n2;
        float rr = 0.0f;
        float ri = 0.0f;
        if (idx < n && idx != 0) {
            const long i0 = idx / (n1 * n2);
            const long rem = idx - i0 * n1 * n2;
            const long i1 = rem / n2;
            const long i2 = rem - i1 * n2;
            const long m0 = (((i0 + n0 - 1) % n0) * n1 + i1) * n2 + i2;
            const long m1 = (i0 * n1 + ((i1 + n1 - 1) % n1)) * n2 + i2;
            const long m2 = (i0 * n1 + i1) * n2 + ((i2 + n2 - 1) % n2);
            float ar = 0.0f;
            float ai = 0.0f;
            #define ADD_RHS(CODE, SIGN_RHS) { \
                const int code = (int)(CODE); \
                const float gr = val_r[code]; \
                const float gi = val_i[code]; \
                ar += (float)(SIGN_RHS) * field_times_voxel * gr; \
                ai += (float)(SIGN_RHS) * field_times_voxel * gi; \
            }
            if (drive_axis == 0) {
                ADD_RHS(c0[idx], -1);
                ADD_RHS(c0[m0], 1);
            } else if (drive_axis == 1) {
                ADD_RHS(c1[idx], -1);
                ADD_RHS(c1[m1], 1);
            } else {
                ADD_RHS(c2[idx], -1);
                ADD_RHS(c2[m2], 1);
            }
            #undef ADD_RHS
            const float2 bv = b[idx];
            rr = ar * bv.x + ai * bv.y;
            ri = ar * bv.y - ai * bv.x;
        }
        sr[tid] = rr;
        si[tid] = ri;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                sr[tid] += sr[tid + stride];
                si[tid] += si[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out[blockIdx.x] = make_float2(sr[0], si[0]);
        }
    }
    '''
    _RHS_HERMITIAN_DOT_KERNEL = cp.RawKernel(code, "rhs_hermitian_dot_blocks")  # type: ignore[union-attr]
    return _RHS_HERMITIAN_DOT_KERNEL


def _bicgstab_p_update_kernel() -> object:
    global _BICGSTAB_P_UPDATE_KERNEL
    require_cupy()
    if _BICGSTAB_P_UPDATE_KERNEL is not None:
        return _BICGSTAB_P_UPDATE_KERNEL
    code = r'''
    extern "C" __global__
    void bicgstab_p_update(float2* p, const float2* r, const float2* v, const float br, const float bi, const float wr, const float wi, const long n) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float2 pv = p[idx];
        const float2 vv = v[idx];
        const float2 rv = r[idx];
        const float wp_r = wr * vv.x - wi * vv.y;
        const float wp_i = wr * vv.y + wi * vv.x;
        const float ur = pv.x - wp_r;
        const float ui = pv.y - wp_i;
        const float bu_r = br * ur - bi * ui;
        const float bu_i = br * ui + bi * ur;
        p[idx] = make_float2(rv.x + bu_r, rv.y + bu_i);
    }
    '''
    _BICGSTAB_P_UPDATE_KERNEL = cp.RawKernel(code, "bicgstab_p_update")  # type: ignore[union-attr]
    return _BICGSTAB_P_UPDATE_KERNEL


def _bicgstab_alpha_kernel() -> object:
    global _BICGSTAB_ALPHA_KERNEL
    require_cupy()
    if _BICGSTAB_ALPHA_KERNEL is not None:
        return _BICGSTAB_ALPHA_KERNEL
    code = r'''
    extern "C" __global__
    void bicgstab_alpha_update(float2* x, float2* r, const float2* p, const float2* v, const float ar, const float ai, const long n) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float2 pv = p[idx];
        const float2 vv = v[idx];
        float2 xv = x[idx];
        float2 rv = r[idx];
        const float ap_r = ar * pv.x - ai * pv.y;
        const float ap_i = ar * pv.y + ai * pv.x;
        const float av_r = ar * vv.x - ai * vv.y;
        const float av_i = ar * vv.y + ai * vv.x;
        xv.x += ap_r;
        xv.y += ap_i;
        rv.x -= av_r;
        rv.y -= av_i;
        if (idx == 0) {
            xv = make_float2(0.0f, 0.0f);
            rv = make_float2(0.0f, 0.0f);
        }
        x[idx] = xv;
        r[idx] = rv;
    }
    '''
    _BICGSTAB_ALPHA_KERNEL = cp.RawKernel(code, "bicgstab_alpha_update")  # type: ignore[union-attr]
    return _BICGSTAB_ALPHA_KERNEL


def _bicgstab_omega_kernel() -> object:
    global _BICGSTAB_OMEGA_KERNEL
    require_cupy()
    if _BICGSTAB_OMEGA_KERNEL is not None:
        return _BICGSTAB_OMEGA_KERNEL
    code = r'''
    extern "C" __global__
    void bicgstab_omega_update(float2* x, float2* r, const float2* t, const float wr, const float wi, const long n) {
        const long idx = (long)blockDim.x * blockIdx.x + threadIdx.x;
        if (idx >= n) {
            return;
        }
        const float2 rv0 = r[idx];
        const float2 tv = t[idx];
        float2 xv = x[idx];
        float2 rv = rv0;
        const float ws_r = wr * rv0.x - wi * rv0.y;
        const float ws_i = wr * rv0.y + wi * rv0.x;
        const float wt_r = wr * tv.x - wi * tv.y;
        const float wt_i = wr * tv.y + wi * tv.x;
        xv.x += ws_r;
        xv.y += ws_i;
        rv.x -= wt_r;
        rv.y -= wt_i;
        if (idx == 0) {
            xv = make_float2(0.0f, 0.0f);
            rv = make_float2(0.0f, 0.0f);
        }
        x[idx] = xv;
        r[idx] = rv;
    }
    '''
    _BICGSTAB_OMEGA_KERNEL = cp.RawKernel(code, "bicgstab_omega_update")  # type: ignore[union-attr]
    return _BICGSTAB_OMEGA_KERNEL


def _launch_compact_matvec(face_data: GPUFaceTypeConductivity, val_r: object, val_i: object, x: object, y: object, block_size: int, grid_size: int) -> None:
    shape_args = tuple(int(n) for n in face_data.shape)
    _compact_matvec_kernel()(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            x,
            y,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
        ),
    )


def _block_bilinear_dot(a: object, b: object, block_sums: object, block_size: int, grid_size: int) -> complex:
    n = int(a.size)
    _bilinear_dot_kernel()((grid_size,), (block_size,), (a, b, block_sums, np.int64(n)))
    return complex(cp.sum(block_sums, dtype=cp.complex128).get())  # type: ignore[union-attr]


def _block_hermitian_dot(a: object, b: object, block_sums: object, block_size: int, grid_size: int) -> complex:
    n = int(a.size)
    _hermitian_dot_kernel()((grid_size,), (block_size,), (a, b, block_sums, np.int64(n)))
    return complex(cp.sum(block_sums, dtype=cp.complex128).get())  # type: ignore[union-attr]


def _block_rhs_hermitian_dot(
    face_data: GPUFaceTypeConductivity,
    val_r: object,
    val_i: object,
    b: object,
    block_sums: object,
    block_size: int,
    grid_size: int,
    axis: int,
    field_strength_v_m: float,
    voxel_size_m: float,
) -> complex:
    shape_args = tuple(int(n) for n in face_data.shape)
    _rhs_hermitian_dot_kernel()(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            b,
            block_sums,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m * voxel_size_m),
        ),
    )
    return complex(cp.sum(block_sums, dtype=cp.complex128).get())  # type: ignore[union-attr]


def _block_norm(a: object, block_sums: object, block_size: int, grid_size: int) -> float:
    n = int(a.size)
    _norm2_kernel()((grid_size,), (block_size,), (a, block_sums, np.int64(n)))
    return float(cp.sqrt(cp.sum(block_sums, dtype=cp.float64)).get())  # type: ignore[union-attr]


def solve_ac3d_bicgstab_compact_gpu_face_types(
    face_data: GPUFaceTypeConductivity,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    residual_every: int = 25,
    iteration_callback: Callable[[int], None] | None = None,
    residual_callback: Callable[[int, float], None] | None = None,
    return_potential: bool = False,
) -> AC3DIterativeResult:
    """Low-memory compact BiCGSTAB solve without storing ``r_hat`` explicitly."""

    require_cupy()
    if np.dtype(face_data.dtype) != np.dtype(np.complex64):
        raise ValueError("compact BiCGSTAB GPU solver currently requires complex64")
    axis = direction_to_axis(direction)
    n_cells = int(np.prod(face_data.shape))
    block_size = 256
    grid_size = (n_cells + block_size - 1) // block_size
    values = cp.asarray(face_data.conductance_values, dtype=cp.complex64)  # type: ignore[union-attr]
    val_r = cp.ascontiguousarray(values.real.astype(cp.float32))  # type: ignore[union-attr]
    val_i = cp.ascontiguousarray(values.imag.astype(cp.float32))  # type: ignore[union-attr]
    x = cp.zeros(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    r = cp.empty(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    p = cp.zeros(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    v = cp.zeros(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    t = cp.empty(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    block_dot = cp.empty(grid_size, dtype=cp.complex64)  # type: ignore[union-attr]
    block_norm = cp.empty(grid_size, dtype=cp.float32)  # type: ignore[union-attr]
    shape_args = tuple(int(n) for n in face_data.shape)
    _compact_rhs_kernel()(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            r,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m * voxel_size_m),
        ),
    )
    rhs_norm = max(_block_norm(r, block_norm, block_size, grid_size), np.finfo(float).eps)
    tolerance = atol + rtol * rhs_norm
    residual_norm_abs = rhs_norm
    relative_residual = 1.0
    rho_old = complex(1.0, 0.0)
    alpha = complex(1.0, 0.0)
    omega = complex(1.0, 0.0)
    history: list[tuple[int, float]] = [(0, relative_residual)]
    info = int(maxiter)
    iterations = 0
    tiny = np.finfo(np.float32).tiny
    p_update_kernel = _bicgstab_p_update_kernel()
    alpha_kernel = _bicgstab_alpha_kernel()
    omega_kernel = _bicgstab_omega_kernel()

    for iterations in range(1, maxiter + 1):
        rho_new = _block_rhs_hermitian_dot(
            face_data,
            val_r,
            val_i,
            r,
            block_dot,
            block_size,
            grid_size,
            axis,
            field_strength_v_m,
            voxel_size_m,
        )
        if abs(rho_new) <= tiny:
            info = -10
            break
        beta = (rho_new / rho_old) * (alpha / omega)
        p_update_kernel(
            (grid_size,),
            (block_size,),
            (
                p,
                r,
                v,
                np.float32(beta.real),
                np.float32(beta.imag),
                np.float32(omega.real),
                np.float32(omega.imag),
                np.int64(n_cells),
            ),
        )
        _launch_compact_matvec(face_data, val_r, val_i, p, v, block_size, grid_size)
        denominator = _block_rhs_hermitian_dot(
            face_data,
            val_r,
            val_i,
            v,
            block_dot,
            block_size,
            grid_size,
            axis,
            field_strength_v_m,
            voxel_size_m,
        )
        if abs(denominator) <= tiny:
            info = -11
            break
        alpha = rho_new / denominator
        alpha_kernel(
            (grid_size,),
            (block_size,),
            (x, r, p, v, np.float32(alpha.real), np.float32(alpha.imag), np.int64(n_cells)),
        )
        residual_norm_abs = _block_norm(r, block_norm, block_size, grid_size)
        if residual_norm_abs <= tolerance:
            relative_residual = residual_norm_abs / rhs_norm
            info = 0
            history.append((iterations, relative_residual))
            if residual_callback is not None:
                residual_callback(iterations, relative_residual)
            if iteration_callback is not None:
                iteration_callback(iterations)
            break
        _launch_compact_matvec(face_data, val_r, val_i, r, t, block_size, grid_size)
        tt = _block_hermitian_dot(t, t, block_dot, block_size, grid_size)
        if abs(tt) <= tiny:
            info = -12
            break
        omega = _block_hermitian_dot(t, r, block_dot, block_size, grid_size) / tt
        omega_kernel(
            (grid_size,),
            (block_size,),
            (x, r, t, np.float32(omega.real), np.float32(omega.imag), np.int64(n_cells)),
        )
        if residual_every > 0 and iterations % residual_every == 0:
            residual_norm_abs = _block_norm(r, block_norm, block_size, grid_size)
            relative_residual = residual_norm_abs / rhs_norm
            history.append((iterations, relative_residual))
            if residual_callback is not None:
                residual_callback(iterations, relative_residual)
            if residual_norm_abs <= tolerance:
                info = 0
                if iteration_callback is not None:
                    iteration_callback(iterations)
                break
        if abs(omega) <= tiny:
            info = -13
            break
        rho_old = rho_new
        if iteration_callback is not None:
            iteration_callback(iterations)

    if history[-1][0] != iterations:
        residual_norm_abs = _block_norm(r, block_norm, block_size, grid_size)
        relative_residual = residual_norm_abs / rhs_norm
        history.append((iterations, relative_residual))
        if residual_callback is not None:
            residual_callback(iterations, relative_residual)
        if residual_norm_abs <= tolerance and info > 0:
            info = 0

    del r, p, v, t, block_dot
    cp.get_default_memory_pool().free_all_blocks()  # type: ignore[union-attr]
    current_r = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    current_i = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    _mean_current_kernel()(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            x,
            current_r,
            current_i,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m),
            np.float32(voxel_size_m),
        ),
    )
    mean_current = complex(
        float(cp.mean(current_r, dtype=cp.float64).get()),  # type: ignore[union-attr]
        float(cp.mean(current_i, dtype=cp.float64).get()),  # type: ignore[union-attr]
    )
    potential = cp.asnumpy(x.reshape(face_data.shape)) if return_potential else None  # type: ignore[union-attr]
    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=mean_current / field_strength_v_m,
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=relative_residual,
        potential=potential,
        solver="gpu_bicgstab_compact_face_types",
        iterations=int(iterations),
        info=int(info),
        residual_history=tuple(history),
    )


def solve_ac3d_cocg_gpu_face_types(
    face_data: GPUFaceTypeConductivity,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    residual_every: int = 25,
    iteration_callback: Callable[[int], None] | None = None,
    residual_callback: Callable[[int, float], None] | None = None,
    return_potential: bool = False,
) -> AC3DIterativeResult:
    """Low-memory conjugate-orthogonal CG solve for complex-symmetric AC3D."""

    require_cupy()
    if np.dtype(face_data.dtype) != np.dtype(np.complex64):
        raise ValueError("COCG GPU solver currently requires complex64")
    axis = direction_to_axis(direction)
    n_cells = int(np.prod(face_data.shape))
    block_size = 256
    grid_size = (n_cells + block_size - 1) // block_size
    values = cp.asarray(face_data.conductance_values, dtype=cp.complex64)  # type: ignore[union-attr]
    val_r = cp.ascontiguousarray(values.real.astype(cp.float32))  # type: ignore[union-attr]
    val_i = cp.ascontiguousarray(values.imag.astype(cp.float32))  # type: ignore[union-attr]
    x = cp.zeros(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    r = cp.empty(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    p = cp.empty(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    q = cp.empty(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    block_dot = cp.empty(grid_size, dtype=cp.complex64)  # type: ignore[union-attr]
    block_norm = cp.empty(grid_size, dtype=cp.float32)  # type: ignore[union-attr]
    shape_args = tuple(int(n) for n in face_data.shape)
    _compact_rhs_kernel()(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            r,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m * voxel_size_m),
        ),
    )
    p[...] = r
    rhs_norm = max(_block_norm(r, block_norm, block_size, grid_size), np.finfo(float).eps)
    tolerance = atol + rtol * rhs_norm
    residual_norm_abs = rhs_norm
    relative_residual = 1.0
    rho = _block_bilinear_dot(r, r, block_dot, block_size, grid_size)
    history: list[tuple[int, float]] = [(0, relative_residual)]
    info = int(maxiter)
    iterations = 0
    tiny = np.finfo(np.float32).tiny

    update_kernel = _cocg_update_kernel()
    p_update_kernel = _cocg_p_update_kernel()
    for iterations in range(1, maxiter + 1):
        _launch_compact_matvec(face_data, val_r, val_i, p, q, block_size, grid_size)
        denominator = _block_bilinear_dot(p, q, block_dot, block_size, grid_size)
        if abs(denominator) <= tiny:
            info = -11
            break
        alpha = rho / denominator
        update_kernel(
            (grid_size,),
            (block_size,),
            (x, r, p, q, np.float32(alpha.real), np.float32(alpha.imag), np.int64(n_cells)),
        )
        if residual_every > 0 and iterations % residual_every == 0:
            residual_norm_abs = _block_norm(r, block_norm, block_size, grid_size)
            relative_residual = residual_norm_abs / rhs_norm
            history.append((iterations, relative_residual))
            if residual_callback is not None:
                residual_callback(iterations, relative_residual)
            if residual_norm_abs <= tolerance:
                info = 0
                if iteration_callback is not None:
                    iteration_callback(iterations)
                break
        rho_new = _block_bilinear_dot(r, r, block_dot, block_size, grid_size)
        if abs(rho) <= tiny:
            info = -10
            break
        beta = rho_new / rho
        p_update_kernel(
            (grid_size,),
            (block_size,),
            (p, r, np.float32(beta.real), np.float32(beta.imag), np.int64(n_cells)),
        )
        rho = rho_new
        if iteration_callback is not None:
            iteration_callback(iterations)

    if history[-1][0] != iterations:
        residual_norm_abs = _block_norm(r, block_norm, block_size, grid_size)
        relative_residual = residual_norm_abs / rhs_norm
        history.append((iterations, relative_residual))
        if residual_callback is not None:
            residual_callback(iterations, relative_residual)
        if residual_norm_abs <= tolerance and info > 0:
            info = 0

    del r, p, q, block_dot
    cp.get_default_memory_pool().free_all_blocks()  # type: ignore[union-attr]
    current_r = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    current_i = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    _mean_current_kernel()(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            x,
            current_r,
            current_i,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m),
            np.float32(voxel_size_m),
        ),
    )
    mean_current = complex(
        float(cp.mean(current_r, dtype=cp.float64).get()),  # type: ignore[union-attr]
        float(cp.mean(current_i, dtype=cp.float64).get()),  # type: ignore[union-attr]
    )
    potential = cp.asnumpy(x.reshape(face_data.shape)) if return_potential else None  # type: ignore[union-attr]
    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=mean_current / field_strength_v_m,
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=relative_residual,
        potential=potential,
        solver="gpu_cocg_face_types",
        iterations=int(iterations),
        info=int(info),
        residual_history=tuple(history),
    )


def solve_ac3d_red_black_sor_gpu_face_types(
    face_data: GPUFaceTypeConductivity,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    omega: float = 1.35,
    residual_every: int = 25,
    iteration_callback: Callable[[int], None] | None = None,
    residual_callback: Callable[[int, float], None] | None = None,
    return_potential: bool = False,
) -> AC3DIterativeResult:
    """Low-memory red-black SOR solve for compact face types.

    The coloring is exact for periodic grids with even dimensions, which is the
    case for the sample-16 full-resolution volume. Odd periodic dimensions can
    connect equal colors across the wrap face and should use another backend.
    """

    require_cupy()
    if np.dtype(face_data.dtype) != np.dtype(np.complex64):
        raise ValueError("red-black SOR GPU solver currently requires complex64")
    if any(int(n) % 2 for n in face_data.shape):
        raise ValueError("red-black SOR requires even grid dimensions for periodic coloring")
    axis = direction_to_axis(direction)
    n_cells = int(np.prod(face_data.shape))
    block_size = 256
    grid_size = (n_cells + block_size - 1) // block_size
    kernel = _red_black_sor_kernel()
    current_kernel = _mean_current_kernel()
    values = cp.asarray(face_data.conductance_values, dtype=cp.complex64)  # type: ignore[union-attr]
    val_r = cp.ascontiguousarray(values.real.astype(cp.float32))  # type: ignore[union-attr]
    val_i = cp.ascontiguousarray(values.imag.astype(cp.float32))  # type: ignore[union-attr]
    x = cp.zeros(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    residual2 = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    shape_args = tuple(int(n) for n in face_data.shape)

    def launch(parity: int) -> None:
        kernel(
            (grid_size,),
            (block_size,),
            (
                face_data.face_types[0],
                face_data.face_types[1],
                face_data.face_types[2],
                val_r,
                val_i,
                x,
                residual2,
                np.int64(shape_args[0]),
                np.int64(shape_args[1]),
                np.int64(shape_args[2]),
                np.int32(axis),
                np.float32(field_strength_v_m * voxel_size_m),
                np.float32(omega),
                np.int32(parity),
            ),
        )

    launch(-1)
    rhs_norm = float(cp.sqrt(cp.sum(residual2, dtype=cp.float64)).get())  # type: ignore[union-attr]
    rhs_norm = max(rhs_norm, np.finfo(float).eps)
    tolerance = atol + rtol * rhs_norm
    history: list[tuple[int, float]] = [(0, 1.0)]
    relative_residual = 1.0
    info = int(maxiter)
    iterations = 0

    for iterations in range(1, maxiter + 1):
        launch(0)
        launch(1)
        if residual_every > 0 and iterations % residual_every == 0:
            launch(-1)
            residual_norm_abs = float(cp.sqrt(cp.sum(residual2, dtype=cp.float64)).get())  # type: ignore[union-attr]
            relative_residual = residual_norm_abs / rhs_norm
            history.append((iterations, relative_residual))
            if residual_callback is not None:
                residual_callback(iterations, relative_residual)
            if residual_norm_abs <= tolerance:
                info = 0
                if iteration_callback is not None:
                    iteration_callback(iterations)
                break
        if iteration_callback is not None:
            iteration_callback(iterations)

    if history[-1][0] != iterations:
        launch(-1)
        residual_norm_abs = float(cp.sqrt(cp.sum(residual2, dtype=cp.float64)).get())  # type: ignore[union-attr]
        relative_residual = residual_norm_abs / rhs_norm
        history.append((iterations, relative_residual))
        if residual_callback is not None:
            residual_callback(iterations, relative_residual)

    current_r = residual2
    current_i = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    current_kernel(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            x,
            current_r,
            current_i,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m),
            np.float32(voxel_size_m),
        ),
    )
    mean_current = complex(
        float(cp.mean(current_r, dtype=cp.float64).get()),  # type: ignore[union-attr]
        float(cp.mean(current_i, dtype=cp.float64).get()),  # type: ignore[union-attr]
    )
    potential = cp.asnumpy(x.reshape(face_data.shape)) if return_potential else None  # type: ignore[union-attr]
    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=mean_current / field_strength_v_m,
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=relative_residual,
        potential=potential,
        solver="gpu_red_black_sor_face_types",
        iterations=int(iterations),
        info=int(info),
        residual_history=tuple(history),
    )


def solve_ac3d_weighted_jacobi_gpu_face_types(
    face_data: GPUFaceTypeConductivity,
    direction: Direction = "x",
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    rtol: float = 1.0e-5,
    atol: float = 0.0,
    maxiter: int = 1000,
    omega: float = 0.65,
    residual_every: int = 25,
    iteration_callback: Callable[[int], None] | None = None,
    residual_callback: Callable[[int, float], None] | None = None,
    return_potential: bool = False,
) -> AC3DIterativeResult:
    """Low-memory full-grid weighted-Jacobi AC3D solve for compact face types.

    This path avoids expanding the three face conductance arrays and keeps only
    two full complex potential arrays plus a real residual workspace on device.
    It is slower than Krylov methods, but allows very large full-resolution
    grids to run when BiCGSTAB cannot fit in GPU memory.
    """

    require_cupy()
    if np.dtype(face_data.dtype) != np.dtype(np.complex64):
        raise ValueError("weighted Jacobi GPU solver currently requires complex64")
    axis = direction_to_axis(direction)
    n_cells = int(np.prod(face_data.shape))
    block_size = 256
    grid_size = (n_cells + block_size - 1) // block_size
    kernel = _weighted_jacobi_kernel()
    current_kernel = _mean_current_kernel()
    values = cp.asarray(face_data.conductance_values, dtype=cp.complex64)  # type: ignore[union-attr]
    val_r = cp.ascontiguousarray(values.real.astype(cp.float32))  # type: ignore[union-attr]
    val_i = cp.ascontiguousarray(values.imag.astype(cp.float32))  # type: ignore[union-attr]
    x = cp.zeros(n_cells, dtype=cp.complex64)  # type: ignore[union-attr]
    x_new = cp.zeros_like(x)  # type: ignore[union-attr]
    residual2 = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    shape_args = tuple(int(n) for n in face_data.shape)
    kernel_args = (
        face_data.face_types[0],
        face_data.face_types[1],
        face_data.face_types[2],
        val_r,
        val_i,
        x,
        x_new,
        residual2,
        np.int64(shape_args[0]),
        np.int64(shape_args[1]),
        np.int64(shape_args[2]),
        np.int32(axis),
        np.float32(field_strength_v_m * voxel_size_m),
        np.float32(0.0),
    )
    kernel((grid_size,), (block_size,), kernel_args)
    rhs_norm = float(cp.sqrt(cp.sum(residual2, dtype=cp.float64)).get())  # type: ignore[union-attr]
    rhs_norm = max(rhs_norm, np.finfo(float).eps)
    tolerance = atol + rtol * rhs_norm
    history: list[tuple[int, float]] = [(0, 1.0)]
    relative_residual = 1.0
    info = int(maxiter)
    iterations = 0

    for iterations in range(1, maxiter + 1):
        kernel_args = (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            x,
            x_new,
            residual2,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m * voxel_size_m),
            np.float32(omega),
        )
        kernel((grid_size,), (block_size,), kernel_args)
        residual_norm_abs = float(cp.sqrt(cp.sum(residual2, dtype=cp.float64)).get())  # type: ignore[union-attr]
        relative_residual = residual_norm_abs / rhs_norm
        x, x_new = x_new, x
        if residual_every > 0 and (iterations % residual_every == 0 or residual_norm_abs <= tolerance):
            history.append((iterations, relative_residual))
            if residual_callback is not None:
                residual_callback(iterations, relative_residual)
        if iteration_callback is not None:
            iteration_callback(iterations)
        if residual_norm_abs <= tolerance:
            info = 0
            break

    if history[-1][0] != iterations:
        history.append((iterations, relative_residual))
        if residual_callback is not None:
            residual_callback(iterations, relative_residual)

    current_r = residual2
    current_i = cp.empty(n_cells, dtype=cp.float32)  # type: ignore[union-attr]
    current_kernel(
        (grid_size,),
        (block_size,),
        (
            face_data.face_types[0],
            face_data.face_types[1],
            face_data.face_types[2],
            val_r,
            val_i,
            x,
            current_r,
            current_i,
            np.int64(shape_args[0]),
            np.int64(shape_args[1]),
            np.int64(shape_args[2]),
            np.int32(axis),
            np.float32(field_strength_v_m),
            np.float32(voxel_size_m),
        ),
    )
    mean_current = complex(
        float(cp.mean(current_r, dtype=cp.float64).get()),  # type: ignore[union-attr]
        float(cp.mean(current_i, dtype=cp.float64).get()),  # type: ignore[union-attr]
    )
    potential = cp.asnumpy(x.reshape(face_data.shape)) if return_potential else None  # type: ignore[union-attr]
    return AC3DIterativeResult(
        direction=axis_to_direction(axis),
        effective_conductivity_s_m=mean_current / field_strength_v_m,
        mean_current_density_a_m2=mean_current,
        field_strength_v_m=field_strength_v_m,
        residual_norm=relative_residual,
        potential=potential,
        solver="gpu_weighted_jacobi_face_types",
        iterations=int(iterations),
        info=int(info),
        residual_history=tuple(history),
    )


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

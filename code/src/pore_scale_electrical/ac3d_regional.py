"""Regional (region-resolved) SIP upscaling on a 3-D CT volume with a pnextract partition.

3-D counterpart of the ``pore_mode="regional"`` route in ``pore_scale_electrical_2d``:

* the pore space is split into pnextract pore bodies (``_VElems``) and throat
  segments (pore voxels within ``L_t / 2`` of a throat interface, inside the two
  pores the throat connects; ``L_t`` = pnextract link2 throat length);
* every region ``k`` carries its own Stern relaxation radius ``r_k`` (pnextract
  inscribed radius) and its own field-weighted dynamic pore size ``Lambda_k``
  computed from the dc pore-only field, so the Eq. 12 increment
  ``2 C_p*(r_k) / Lambda_k`` is added only inside region ``k``;
* throat segments additionally carry Titov membrane polarization as the
  intensive conductivity ``sigma_w Z_dc / Z*`` (same as the 2-D route).

The polarization kernels are imported from ``pore_scale_electrical_2d.polarization``
so the 2-D and 3-D regional routes use identical equations. The field solve
reuses the GPU BiCGSTAB of ``ac3d_gpu`` with per-face (harmonic mean)
conductances built from a per-voxel conductivity map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from pore_scale_electrical.ac3d_active_domain import periodic_active_component_anchor_indices
from pore_scale_electrical.ac3d_gpu import _gpu_bicgstab, periodic_laplacian_eigenvalues_gpu, require_cupy

try:  # pragma: no cover - exercised only with a CUDA device
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]


SOLID_REGION = 0

# pnextract _VElems codes (verified on the Berea run: -2 equals the solid voxel count).
VELEMS_SOLID = -2
VELEMS_UNASSIGNED_PORE = -1


# --------------------------------------------------------------------------- pnextract inputs


def read_velems(prefix: Path, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Read ``<prefix>_VElems.raw`` (int32, padded by one voxel) and strip the padding.

    The value of a pore voxel is ``pore_id + 1`` (pore ids 1..N in node1/node2),
    ``-1`` marks pore voxels pnextract did not assign, ``-2`` is solid.
    """

    nz, ny, nx = shape_zyx
    raw = np.fromfile(Path(f"{prefix}_VElems.raw"), dtype="<i4")
    padded = raw.reshape(nz + 2, ny + 2, nx + 2)
    return np.ascontiguousarray(padded[1:-1, 1:-1, 1:-1])


def read_throat_voxels(prefix: Path, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    """Read ``<prefix>_throats.raw.gz`` (``write_throats``) as throat ids on the image grid.

    pnextract fills a padded field at ``(i+1, j+1, k+1)`` but writes index range
    ``0..n-1`` of the padded field, i.e. the written array is shifted by one
    voxel. The shift is undone here (written ``[k+1]`` -> image ``[k]``); the last
    image plane of each axis is therefore not covered. Value = throat_id (1-based),
    0 = not a throat-interface voxel. The file is plain int32 despite the ``.gz`` suffix.
    """

    nz, ny, nx = shape_zyx
    written = np.fromfile(Path(f"{prefix}_throats.raw.gz"), dtype="<i4").reshape(nz, ny, nx)
    out = np.zeros(shape_zyx, dtype=np.int32)
    out[:-1, :-1, :-1] = written[1:, 1:, 1:]
    return out


@dataclass(frozen=True)
class RegionMap:
    """Region labels: 0 solid, 1..n_pores pore bodies, n_pores + t throat segment t."""

    labels: np.ndarray
    n_pores: int
    n_throats: int
    stats: dict

    @property
    def n_regions(self) -> int:
        return self.n_pores + self.n_throats

    def throat_region(self, throat_id: np.ndarray | int) -> np.ndarray | int:
        return self.n_pores + np.asarray(throat_id)


def build_region_map(
    pore_mask: np.ndarray,
    velems: np.ndarray,
    throat_voxels: np.ndarray,
    throat_pore1: np.ndarray,
    throat_pore2: np.ndarray,
    throat_length_m: np.ndarray,
    voxel_size_m: float,
    n_pores: int,
) -> RegionMap:
    """Partition the pore space into pnextract pore bodies and throat segments.

    ``throat_*`` arrays are indexed by ``throat_id - 1``. A pore voxel joins
    throat ``t`` when its nearest throat-interface voxel belongs to ``t``, its
    Euclidean distance to that voxel is ``<= L_t / 2`` and it lies in one of the
    two pores ``t`` connects (interface voxels themselves always join ``t``).
    pnextract-unassigned pore voxels first take the label of the nearest pore body.
    """

    pore = np.asarray(pore_mask, dtype=bool)
    pore_id = np.where(velems >= 2, velems - 1, 0).astype(np.int32)
    if np.any(pore_id[~pore] != 0) or np.any((velems == VELEMS_SOLID) & pore):
        raise ValueError("VElems solid/pore codes do not match the segmentation")
    unassigned = pore & (pore_id == 0)
    n_unassigned = int(np.count_nonzero(unassigned))
    if n_unassigned:
        _, nearest = ndimage.distance_transform_edt(pore_id == 0, return_indices=True)
        filled = pore_id[tuple(nearest)]
        pore_id[unassigned] = filled[unassigned]
        del nearest, filled

    n_throats = int(len(throat_length_m))
    is_interface = (throat_voxels > 0) & pore
    distance_vox, nearest = ndimage.distance_transform_edt(~is_interface, return_indices=True)
    nearest_tid = throat_voxels[tuple(nearest)]
    del nearest
    half_length_vox = np.zeros(n_throats + 1)
    half_length_vox[1:] = 0.5 * np.asarray(throat_length_m, dtype=float) / voxel_size_m
    p1 = np.zeros(n_throats + 1, dtype=np.int64)
    p2 = np.zeros(n_throats + 1, dtype=np.int64)
    p1[1:] = throat_pore1
    p2[1:] = throat_pore2
    within = pore & (nearest_tid > 0) & (distance_vox <= half_length_vox[nearest_tid])
    adjacent = (pore_id == p1[nearest_tid]) | (pore_id == p2[nearest_tid])
    in_throat = is_interface | (within & adjacent)
    labels = pore_id.copy()
    labels[in_throat] = n_pores + nearest_tid[in_throat]
    labels[~pore] = SOLID_REGION

    counts = np.bincount(labels.ravel(), minlength=n_pores + n_throats + 1)
    throat_voxel_count = int(np.count_nonzero(in_throat))
    stats = {
        "pore_voxels": int(np.count_nonzero(pore)),
        "pnextract_unassigned_pore_voxels_filled_by_nearest_pore": n_unassigned,
        "throat_interface_voxels": int(np.count_nonzero(is_interface)),
        "throat_segment_voxels": throat_voxel_count,
        "throat_segment_volume_fraction_of_pore_space": throat_voxel_count / max(int(np.count_nonzero(pore)), 1),
        "pore_bodies_with_no_voxels_left": int(np.count_nonzero(counts[1 : n_pores + 1] == 0)),
        "throats_with_no_voxels": int(np.count_nonzero(counts[n_pores + 1 :] == 0)),
    }
    return RegionMap(labels=labels, n_pores=int(n_pores), n_throats=n_throats, stats=stats)


# --------------------------------------------------------------------------- dynamic pore size per region


@dataclass(frozen=True)
class RegionalDynamicPoreSize:
    volume_integral_v2_m: np.ndarray  # int_Vk |E|^2 dV, index = region label
    surface_integral_v2: np.ndarray  # int_Sk |E_t|^2 dS over pore-solid faces of region k
    voxel_count: np.ndarray
    wall_face_count: np.ndarray
    voxel_size_m: float

    @property
    def dynamic_pore_size_m(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * self.volume_integral_v2_m / self.surface_integral_v2

    @property
    def geometric_size_m(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 2.0 * self.voxel_count * self.voxel_size_m / self.wall_face_count

    @property
    def global_dynamic_pore_size_m(self) -> float:
        return float(2.0 * self.volume_integral_v2_m[1:].sum() / self.surface_integral_v2[1:].sum())


def regional_dynamic_pore_sizes(
    labels: np.ndarray,
    field_components: np.ndarray,
    voxel_size_m: float,
    n_regions: int,
) -> RegionalDynamicPoreSize:
    """Region-resolved version of ``compute_dynamic_pore_size_from_fields``.

    Same discretisation: pore-centre ``|E|^2`` for the volume integral and the
    pore-side tangential ``|E|^2 - |E_axis|^2`` on every pore-solid face. Each
    face is attributed to the region of its pore-side voxel; faces between two
    pore regions are internal and are not walls. Summing over regions recovers
    the global Niu ``Lambda`` exactly.
    """

    lab = np.asarray(labels)
    pore = lab != SOLID_REGION
    energy = np.sum(np.abs(field_components) ** 2, axis=0)
    size = n_regions + 1
    volume = np.bincount(lab[pore], weights=energy[pore], minlength=size) * voxel_size_m**3
    voxel_count = np.bincount(lab[pore], minlength=size).astype(float)
    surface = np.zeros(size)
    faces = np.zeros(size)
    for axis in range(3):
        neighbor_pore = np.roll(pore, -1, axis=axis)
        pore_to_solid = pore & ~neighbor_pore
        tangential = energy[pore_to_solid] - np.abs(field_components[axis][pore_to_solid]) ** 2
        surface += np.bincount(lab[pore_to_solid], weights=tangential, minlength=size)
        faces += np.bincount(lab[pore_to_solid], minlength=size)
        solid_to_pore = ~pore & neighbor_pore
        neighbor_energy = np.roll(energy, -1, axis=axis)
        neighbor_axis = np.roll(np.abs(field_components[axis]) ** 2, -1, axis=axis)
        neighbor_label = np.roll(lab, -1, axis=axis)
        tangential = neighbor_energy[solid_to_pore] - neighbor_axis[solid_to_pore]
        surface += np.bincount(neighbor_label[solid_to_pore], weights=tangential, minlength=size)
        faces += np.bincount(neighbor_label[solid_to_pore], minlength=size)
        del neighbor_energy, neighbor_axis, neighbor_label
    surface *= voxel_size_m**2
    volume[SOLID_REGION] = surface[SOLID_REGION] = voxel_count[SOLID_REGION] = faces[SOLID_REGION] = 0.0
    return RegionalDynamicPoreSize(volume, surface, voxel_count, faces, float(voxel_size_m))


def local_thickness_gpu(mask: np.ndarray) -> np.ndarray:
    """Local thickness of a binary phase in voxels (Hildebrand & Rueegsegger 1997).

    Diameter of the largest inscribed ball that contains the voxel, evaluated on integer radii:
    a voxel gets 2 r at the largest r for which it lies within r of a voxel whose distance to
    the other phase is >= r. Distances are measured to the nearest voxel of the other phase
    (non-periodic), so every voxel of the phase has thickness >= 2.
    """

    from cupyx.scipy import ndimage as cnd

    require_cupy()
    phase = cp.asarray(mask, dtype=bool)  # type: ignore[union-attr]
    distance = cnd.distance_transform_edt(phase)
    thickness = cp.zeros(phase.shape, dtype=cp.float32)  # type: ignore[union-attr]
    for radius in range(int(cp.floor(distance.max()).get()), 0, -1):  # type: ignore[union-attr]
        centers = distance >= radius
        reach = cnd.distance_transform_edt(~centers) <= radius
        thickness = cp.where(reach & phase & (thickness == 0), 2.0 * radius, thickness)  # type: ignore[union-attr]
        del centers, reach
    return cp.asnumpy(thickness)  # type: ignore[union-attr]


# --------------------------------------------------------------------------- linear response


def voxel_sensitivities(
    sigma_voxels: np.ndarray,
    potential: np.ndarray,
    *,
    drive_axis: int,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
) -> np.ndarray:
    """``N E0^2 d sigma_eff / d sigma_v`` for every voxel of a solved (real, dc) problem.

    sigma_eff = (1/(N E0^2)) sum_f g_f E_f^2 is stationary in the potential, so its first-order
    change is (1/(N E0^2)) sum_f (d g_f / d sigma_v) d sigma_v E_f^2 with the unperturbed face
    fields E_f. For the half-cell harmonic mean g = 2 a b / (a + b): dg/da = 2 b^2 / (a + b)^2.
    Summing the returned map over a region and multiplying by that region's (complex) increment
    gives the region's linear-response contribution to sigma_eff.
    """

    s = np.asarray(sigma_voxels, dtype=float)
    u = np.asarray(potential, dtype=float)
    out = np.zeros(s.shape)
    for axis in range(3):
        neighbor = np.roll(s, -1, axis=axis)
        e = -(np.roll(u, -1, axis=axis) - u) / voxel_size_m
        if axis == drive_axis:
            e += field_strength_v_m
        e *= e
        denominator = (s + neighbor) ** 2
        safe = np.where(denominator > 0, denominator, 1.0)
        left = np.where(denominator > 0, 2.0 * neighbor**2 / safe, 0.0) * e
        right = np.where(denominator > 0, 2.0 * s**2 / safe, 0.0) * e
        out += left + np.roll(right, 1, axis=axis)
        del neighbor, e, denominator, safe, left, right
    return out / field_strength_v_m**2


# --------------------------------------------------------------------------- GPU field solve on a per-voxel map
#
# Memory-lean variant of ac3d_gpu.solve_ac3d_matrix_free_gpu_face_types for an arbitrary
# per-voxel conductivity: the positive-face conductances are the same half-cell harmonic
# means 2 s_i s_j / (s_i + s_j) (0 when both are 0), but they are formed on the fly inside a
# fused CUDA matvec instead of being stored as three face arrays. Operator, right-hand side,
# gauge rows (inactive cells + one anchor per periodic active component, or cell 0 when all
# cells are active), FFT-Poisson preconditioner and BiCGSTAB are unchanged.

_MATVEC_SOURCE = r"""
__device__ __forceinline__ double2 harmonic_face(const double2 a, const double2 b) {
    const double sr = a.x + b.x, si = a.y + b.y;
    if (sr == 0.0 && si == 0.0) return make_double2(0.0, 0.0);
    const double pr = a.x * b.x - a.y * b.y, pi = a.x * b.y + a.y * b.x;
    const double d = sr * sr + si * si;
    return make_double2(2.0 * (pr * sr + pi * si) / d, 2.0 * (pi * sr - pr * si) / d);
}

extern "C" __global__ void voxel_matvec(
    const double2* sigma, const double2* x, double2* y, const unsigned char* fixed,
    const long long n0, const long long n1, const long long n2
) {
    const long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
    const long long n = n0 * n1 * n2;
    if (idx >= n) return;
    const double2 xi = x[idx];
    if (fixed[idx]) { y[idx] = xi; return; }
    const long long i0 = idx / (n1 * n2);
    const long long rem = idx - i0 * n1 * n2;
    const long long i1 = rem / n2;
    const long long i2 = rem - i1 * n2;
    const long long nb[6] = {
        (((i0 + 1) % n0) * n1 + i1) * n2 + i2,
        (((i0 + n0 - 1) % n0) * n1 + i1) * n2 + i2,
        (i0 * n1 + ((i1 + 1) % n1)) * n2 + i2,
        (i0 * n1 + ((i1 + n1 - 1) % n1)) * n2 + i2,
        (i0 * n1 + i1) * n2 + ((i2 + 1) % n2),
        (i0 * n1 + i1) * n2 + ((i2 + n2 - 1) % n2),
    };
    const double2 si = sigma[idx];
    double yr = 0.0, yi = 0.0;
    for (int k = 0; k < 6; ++k) {
        const double2 g = harmonic_face(si, sigma[nb[k]]);
        const double2 xj = x[nb[k]];
        const double dr = xi.x - xj.x, di = xi.y - xj.y;
        yr += g.x * dr - g.y * di;
        yi += g.x * di + g.y * dr;
    }
    y[idx] = make_double2(yr, yi);
}
"""
_MATVEC_KERNEL = None
_HARMONIC_KERNEL = None


def _matvec_kernel() -> object:
    global _MATVEC_KERNEL
    if _MATVEC_KERNEL is None:
        _MATVEC_KERNEL = cp.RawKernel(_MATVEC_SOURCE, "voxel_matvec")  # type: ignore[union-attr]
    return _MATVEC_KERNEL


def face_conductivity_axis_gpu(sigma: object, axis: int) -> object:
    """Positive-face harmonic means along one axis (periodic), 0 where both voxels are 0."""

    global _HARMONIC_KERNEL
    if _HARMONIC_KERNEL is None:
        _HARMONIC_KERNEL = cp.ElementwiseKernel(  # type: ignore[union-attr]
            "complex128 a, complex128 b",
            "complex128 g",
            "g = (a + b == complex<double>(0.0, 0.0)) ? complex<double>(0.0, 0.0) : 2.0 * a * b / (a + b)",
            "regional_harmonic_face",
        )
    return _HARMONIC_KERNEL(sigma, cp.roll(sigma, -1, axis=axis))  # type: ignore[union-attr]


def _face_field(u: object, axis: int, drive_axis: int, field_strength_v_m: float, voxel_size_m: float) -> object:
    """Face field ``E0 [axis == drive] - (u_{i+1} - u_i) / h`` on positive faces."""

    e = cp.roll(u, -1, axis=axis)  # type: ignore[union-attr]
    e -= u
    e *= -1.0 / voxel_size_m
    if axis == drive_axis:
        e += field_strength_v_m
    return e


def mean_current_and_energy_gpu(
    sigma: object, potential: object, drive_axis: int, field_strength_v_m: float, voxel_size_m: float
) -> tuple[complex, complex]:
    """Mean-current ``<J>/E0`` and the bilinear energy form ``sum_f g_f E_f^2 / (N E0^2)``.

    Both equal sigma_eff at the exact solution; the energy form is stationary (its error is
    second order in the potential error), so the difference checks convergence for small
    polarization increments.
    """

    n = float(np.prod(sigma.shape))
    current = 0.0 + 0.0j
    energy = 0.0 + 0.0j
    for axis in range(3):
        g = face_conductivity_axis_gpu(sigma, axis)
        e = _face_field(potential, axis, drive_axis, field_strength_v_m, voxel_size_m)
        g *= e
        if axis == drive_axis:
            current = complex(cp.sum(g).get()) / n  # type: ignore[union-attr]
        g *= e
        energy += complex(cp.sum(g).get())  # type: ignore[union-attr]
        del g, e
    return current / field_strength_v_m, energy / (n * field_strength_v_m**2)


@dataclass(frozen=True)
class VoxelSolveResult:
    effective_conductivity_s_m: complex
    energy_form_conductivity_s_m: complex
    iterations: int
    info: int
    true_residual_norm: float
    true_residual_passed: bool
    residual_history: tuple[tuple[int, float], ...]
    potential: np.ndarray | None
    elapsed_s: float
    fft_reference_s_m: complex


class _VoxelOperator:
    """``matvec`` of the periodic finite-volume operator with identity rows at ``fixed`` cells."""

    def __init__(self, sigma: object, fixed: object) -> None:
        self.sigma, self.fixed = sigma, fixed
        self.shape = tuple(int(v) for v in sigma.shape)
        self.grid = ((int(np.prod(self.shape)) + 255) // 256,)

    def matvec(self, vector: object) -> object:
        x = cp.ascontiguousarray(vector, dtype=cp.complex128).ravel()  # type: ignore[union-attr]
        y = cp.empty_like(x)  # type: ignore[union-attr]
        n0, n1, n2 = (np.int64(v) for v in self.shape)
        _matvec_kernel()(self.grid, (256,), (self.sigma, x, y, self.fixed, n0, n1, n2))
        return y


def _cocg(operator, rhs, *, x0, rtol: float, maxiter: int, preconditioner, residual_every: int):
    """Preconditioned COCG (bilinear, unconjugated dot products) for complex-symmetric systems.

    Used on the pure periodic operator of a fully conducting medium: singular with the constant
    null space, a zero-mean right-hand side and a preconditioner that removes the constant mode.
    """

    x = cp.zeros_like(rhs) if x0 is None else cp.asarray(x0, dtype=cp.complex128).ravel().copy()  # type: ignore[union-attr]
    r = rhs - operator.matvec(x) if x0 is not None else rhs.copy()
    rhs_norm = max(float(cp.linalg.norm(rhs).get()), np.finfo(float).tiny)  # type: ignore[union-attr]
    relative = float(cp.linalg.norm(r).get()) / rhs_norm  # type: ignore[union-attr]
    history: list[tuple[int, float]] = [(0, relative)]
    iterations = 0
    if relative > rtol:
        z = preconditioner(r)
        p = z.copy()
        rho = cp.sum(r * z)  # type: ignore[union-attr]
        for iterations in range(1, maxiter + 1):
            q = operator.matvec(p)
            alpha = rho / cp.sum(p * q)  # type: ignore[union-attr]
            x += alpha * p
            r -= alpha * q
            del q
            relative = float(cp.linalg.norm(r).get()) / rhs_norm  # type: ignore[union-attr]
            if residual_every > 0 and iterations % residual_every == 0:
                history.append((iterations, relative))
            if relative <= rtol:
                break
            z = preconditioner(r)
            rho_new = cp.sum(r * z)  # type: ignore[union-attr]
            p *= rho_new / rho
            p += z
            rho = rho_new
            del z
    true_relative = float(cp.linalg.norm(rhs - operator.matvec(x)).get()) / rhs_norm  # type: ignore[union-attr]
    if not history or history[-1][0] != iterations:
        history.append((iterations, relative))
    passed = true_relative <= rtol * 1.0001
    return x, (0 if passed else max(iterations, 1)), iterations, true_relative, passed, tuple(history)


def solve_voxel_conductivity_gpu(
    sigma_voxels: np.ndarray,
    *,
    drive_axis: int = 0,
    field_strength_v_m: float = 1.0,
    voxel_size_m: float = 1.0,
    fft_reference_s_m: complex,
    rtol: float = 1.0e-8,
    maxiter: int = 3000,
    x0: np.ndarray | None = None,
    residual_every: int = 50,
    return_potential: bool = True,
) -> VoxelSolveResult:
    """Periodic AC3D solve for an arbitrary complex per-voxel conductivity map.

    Fully conducting media (every cell has a conducting face) use COCG on the pure periodic
    operator (constant null space, no gauge row); media with insulating cells keep the
    BiCGSTAB path with identity rows for inactive cells and one anchor per active component.
    """

    import time

    require_cupy()
    start = time.perf_counter()
    pool = cp.get_default_memory_pool()  # type: ignore[union-attr]
    pool.free_all_blocks()
    sigma = cp.ascontiguousarray(cp.asarray(sigma_voxels, dtype=cp.complex128))  # type: ignore[union-attr]
    shape = tuple(int(n) for n in sigma.shape)

    # Active cell: at least one face with non-zero harmonic conductance (same as |diagonal| > 0).
    nonzero = sigma != 0
    has_partner = cp.zeros(shape, dtype=bool)  # type: ignore[union-attr]
    for axis in range(3):
        has_partner |= cp.roll(nonzero, -1, axis=axis)  # type: ignore[union-attr]
        has_partner |= cp.roll(nonzero, 1, axis=axis)  # type: ignore[union-attr]
    active = nonzero & has_partner
    del nonzero, has_partner
    fixed = cp.zeros(shape, dtype=cp.uint8)  # type: ignore[union-attr]
    fully_active = bool(cp.all(active).get())  # type: ignore[union-attr]
    if not fully_active:
        fixed[~active] = 1
        anchors = periodic_active_component_anchor_indices(cp.asnumpy(active))  # type: ignore[union-attr]
        fixed.ravel()[cp.asarray(anchors, dtype=cp.int64)] = 1  # type: ignore[union-attr]
    del active
    fixed_flat = fixed.ravel().astype(bool)

    g_drive = face_conductivity_axis_gpu(sigma, drive_axis)
    rhs = (cp.roll(g_drive, 1, axis=drive_axis) - g_drive) * (field_strength_v_m * voxel_size_m)  # type: ignore[union-attr]
    del g_drive
    rhs = rhs.ravel()
    rhs[fixed_flat] = 0.0

    operator = _VoxelOperator(sigma, fixed.ravel())
    eigen = periodic_laplacian_eigenvalues_gpu(shape, np.dtype(np.complex128))
    inverse_eigen = cp.zeros(shape, dtype=cp.float64)  # type: ignore[union-attr]
    positive = eigen > np.finfo(float).tiny
    inverse_eigen[positive] = 1.0 / eigen[positive]
    inverse_eigen[0, 0, 0] = 0.0
    del eigen, positive
    inverse_reference = 1.0 / complex(fft_reference_s_m)

    def preconditioner(vector: object) -> object:
        r = vector.reshape(shape)
        spectrum = cp.fft.fftn(r - cp.mean(r))  # type: ignore[union-attr]
        spectrum *= inverse_eigen
        spectrum *= inverse_reference
        out = cp.fft.ifftn(spectrum).ravel()  # type: ignore[union-attr]
        del spectrum
        if not fully_active:
            out[fixed_flat] = vector[fixed_flat]
        return out

    initial = None if x0 is None else cp.asarray(x0, dtype=cp.complex128).ravel()  # type: ignore[union-attr]
    if fully_active:
        solution, info, iterations, true_residual, passed, history = _cocg(
            operator, rhs, x0=initial, rtol=rtol, maxiter=maxiter, preconditioner=preconditioner, residual_every=residual_every
        )
    else:
        solution, info, iterations, _recursive, true_residual, passed, history = _gpu_bicgstab(
            operator,
            rhs,
            x0=initial,
            rtol=rtol,
            atol=0.0,
            maxiter=maxiter,
            preconditioner=preconditioner,
            iteration_callback=None,
            residual_every=residual_every,
            residual_callback=None,
        )
    del rhs, inverse_eigen, initial
    pool.free_all_blocks()
    u = solution.reshape(shape)
    mean_current, energy = mean_current_and_energy_gpu(sigma, u, drive_axis, field_strength_v_m, voxel_size_m)
    potential = cp.asnumpy(u) if return_potential else None  # type: ignore[union-attr]
    del sigma, fixed, fixed_flat, solution, u, operator
    pool.free_all_blocks()
    return VoxelSolveResult(
        effective_conductivity_s_m=complex(mean_current),
        energy_form_conductivity_s_m=complex(energy),
        iterations=int(iterations),
        info=int(info),
        true_residual_norm=float(true_residual),
        true_residual_passed=bool(passed),
        residual_history=tuple(history),
        potential=potential,
        elapsed_s=float(time.perf_counter() - start),
        fft_reference_s_m=complex(fft_reference_s_m),
    )

"""Periodic 2-D complex finite-volume solver (AC2D-periodic).

This is the two-dimensional counterpart of the AC3D periodic solver in
``pore_scale_electrical.ac3d_solver`` (Niu et al., 2020). For a prescribed
macroscopic field ``E0`` along one axis, the periodic correction potential
``u`` satisfies

    div( sigma*(x, omega) * (E0 - grad u) ) = 0,

and the effective conductivity is ``<J> / E0``. Unknowns are cell centred;
face conductances use the half-cell series (harmonic) average, generalized to
rectilinear cells so that thin throats can be resolved by grading the grid.

2-D convention: all quantities are per unit out-of-plane depth. Conductivity
(S/m) and Lambda (m) are intensive, so the 2-D model is a translationally
invariant extrusion and no thickness enters ``sigma_eff``.

Array layout: ``(ny, nx)`` with row index increasing with ``y`` and column
index increasing with ``x``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components


Direction2D = Literal["x", "y"]


@dataclass(frozen=True)
class Grid2D:
    """Rectilinear cell sizes in metres (uniform grids are a special case)."""

    dx_m: np.ndarray
    dy_m: np.ndarray

    def __post_init__(self) -> None:
        dx = np.asarray(self.dx_m, dtype=float)
        dy = np.asarray(self.dy_m, dtype=float)
        if dx.ndim != 1 or dy.ndim != 1 or dx.size < 2 or dy.size < 1:
            raise ValueError("dx_m and dy_m must be 1-D with at least 2 x-cells")
        if np.any(dx <= 0) or np.any(dy <= 0):
            raise ValueError("cell sizes must be positive")
        object.__setattr__(self, "dx_m", dx)
        object.__setattr__(self, "dy_m", dy)

    @classmethod
    def uniform(cls, nx: int, ny: int, dx_m: float, dy_m: float | None = None) -> "Grid2D":
        return cls(np.full(nx, float(dx_m)), np.full(ny, float(dx_m if dy_m is None else dy_m)))

    @property
    def nx(self) -> int:
        return int(self.dx_m.size)

    @property
    def ny(self) -> int:
        return int(self.dy_m.size)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    @property
    def length_x_m(self) -> float:
        return float(self.dx_m.sum())

    @property
    def length_y_m(self) -> float:
        return float(self.dy_m.sum())

    @property
    def x_edges_m(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(self.dx_m)]

    @property
    def y_edges_m(self) -> np.ndarray:
        return np.r_[0.0, np.cumsum(self.dy_m)]

    @property
    def x_centers_m(self) -> np.ndarray:
        edges = self.x_edges_m
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def y_centers_m(self) -> np.ndarray:
        edges = self.y_edges_m
        return 0.5 * (edges[:-1] + edges[1:])

    @property
    def cell_area_m2(self) -> np.ndarray:
        return self.dy_m[:, None] * self.dx_m[None, :]


@dataclass(frozen=True)
class AC2DPeriodicResult:
    """Result of one imposed-field periodic solve."""

    direction: str
    effective_conductivity_s_m: complex
    transverse_conductivity_s_m: complex
    field_strength_v_m: float
    residual_norm: float
    potential: np.ndarray
    active_mask: np.ndarray
    n_unknowns: int
    n_components: int


def series_face_conductance(sigma_a, sigma_b, length_a, length_b, face_length):
    """Conductance per unit depth of two half cells in series.

    ``G = face_length / (length_a / (2 sigma_a) + length_b / (2 sigma_b))``,
    written as ``2 sigma_a sigma_b face_length / (length_a sigma_b + length_b sigma_a)``
    so that a zero-conductivity side gives an exactly insulating face. On a
    uniform grid this equals the AC3D harmonic face conductivity times
    ``face_length / length``.
    """

    sa = np.asarray(sigma_a, dtype=np.complex128)
    sb = np.asarray(sigma_b, dtype=np.complex128)
    denominator = np.asarray(length_a) * sb + np.asarray(length_b) * sa
    numerator = 2.0 * sa * sb * np.asarray(face_length)
    insulating = np.abs(denominator) <= np.finfo(float).tiny
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(insulating, 0.0j, numerator / np.where(insulating, 1.0, denominator))


def periodic_face_conductances(sigma: np.ndarray, grid: Grid2D) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(Gx, dcx, Gy, dcy)`` for positive-direction periodic faces.

    ``Gx[j, i]`` couples cell ``(j, i)`` and ``(j, (i+1) % nx)``; ``dcx[i]`` is
    their centre distance. ``Gy`` / ``dcy`` are the analogous y quantities
    (``Gy`` is all zero when ``ny == 1``).
    """

    s = np.asarray(sigma, dtype=np.complex128)
    if s.shape != grid.shape:
        raise ValueError(f"sigma shape {s.shape} does not match grid {grid.shape}")
    dx, dy = grid.dx_m, grid.dy_m
    dx_next = np.roll(dx, -1)
    gx = series_face_conductance(s, np.roll(s, -1, axis=1), dx[None, :], dx_next[None, :], dy[:, None])
    dcx = 0.5 * (dx + dx_next)
    if grid.ny > 1:
        dy_next = np.roll(dy, -1)
        gy = series_face_conductance(s, np.roll(s, -1, axis=0), dy[:, None], dy_next[:, None], dx[None, :])
        dcy = 0.5 * (dy + dy_next)
    else:
        gy = np.zeros_like(gx)
        dcy = np.ones(1)
    return gx, dcx, gy, dcy


def solve_ac2d_periodic(
    conductivity_s_m: np.ndarray,
    grid: Grid2D,
    direction: Direction2D = "x",
    field_strength_v_m: float = 1.0,
) -> AC2DPeriodicResult:
    """Solve the periodic correction-potential problem with a direct sparse LU.

    Cells with ``sigma == 0`` (e.g. the insulating solid in the Niu pore and
    membrane mechanisms) are removed from the unknowns. One potential per
    connected component of conducting cells is anchored, which removes the
    constant null space of each periodic component, as in the AC3D
    active-domain solver.
    """

    if direction not in ("x", "y"):
        raise ValueError("direction must be 'x' or 'y'")
    if field_strength_v_m == 0:
        raise ValueError("field_strength_v_m must be non-zero")
    sigma = np.asarray(conductivity_s_m, dtype=np.complex128)
    if direction == "y":
        result = solve_ac2d_periodic(sigma.T, Grid2D(grid.dy_m, grid.dx_m), "x", field_strength_v_m)
        return AC2DPeriodicResult(
            direction="y",
            effective_conductivity_s_m=result.effective_conductivity_s_m,
            transverse_conductivity_s_m=result.transverse_conductivity_s_m,
            field_strength_v_m=field_strength_v_m,
            residual_norm=result.residual_norm,
            potential=result.potential.T,
            active_mask=result.active_mask.T,
            n_unknowns=result.n_unknowns,
            n_components=result.n_components,
        )

    ny, nx = sigma.shape
    gx, dcx, gy, dcy = periodic_face_conductances(sigma, grid)
    active = np.abs(sigma) > 0
    n_active = int(np.count_nonzero(active))
    if n_active == 0:
        raise ValueError("conductivity map has no conducting cells")
    index = np.full(sigma.shape, -1, dtype=np.int64)
    index[active] = np.arange(n_active)

    a_x = index
    b_x = np.roll(index, -1, axis=1)
    a_y = index
    b_y = np.roll(index, -1, axis=0)
    face_sets = [(a_x, b_x, gx, np.broadcast_to(dcx[None, :], gx.shape), True)]
    if ny > 1:
        face_sets.append((a_y, b_y, gy, np.broadcast_to(dcy[:, None], gy.shape), False))

    rows, cols, vals = [], [], []
    rhs = np.zeros(n_active, dtype=np.complex128)
    adjacency_rows, adjacency_cols = [], []
    for a, b, g, dist, drives in face_sets:
        use = (a >= 0) & (b >= 0) & (np.abs(g) > 0)
        ia, ib, gv = a[use], b[use], g[use]
        rows.extend([ia, ib, ia, ib])
        cols.extend([ia, ib, ib, ia])
        vals.extend([gv, gv, -gv, -gv])
        adjacency_rows.append(ia)
        adjacency_cols.append(ib)
        if drives:
            drive = gv * field_strength_v_m * dist[use]
            np.add.at(rhs, ia, -drive)
            np.add.at(rhs, ib, drive)

    matrix = sp.coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_active, n_active)
    ).tocsr()
    adjacency = sp.coo_matrix(
        (np.ones(sum(len(r) for r in adjacency_rows)), (np.concatenate(adjacency_rows), np.concatenate(adjacency_cols))),
        shape=(n_active, n_active),
    )
    n_components, component = connected_components(adjacency, directed=False)
    anchors = np.unique(component, return_index=True)[1]
    system = matrix.tolil()
    system_rhs = rhs.copy()
    for anchor in anchors:
        system.rows[anchor] = [int(anchor)]
        system.data[anchor] = [1.0 + 0.0j]
        system_rhs[anchor] = 0.0
    system = system.tocsc()
    # Minimum-degree ordering on A^T + A: ~2.6x faster than COLAMD on graded 2-D grids.
    solution = spla.spsolve(system, system_rhs, permc_spec="MMD_AT_PLUS_A")
    residual = system @ solution - system_rhs
    residual_norm = float(np.linalg.norm(residual) / max(np.linalg.norm(system_rhs), np.finfo(float).tiny))

    potential = np.full(sigma.shape, np.nan + 0.0j, dtype=np.complex128)
    potential[active] = solution
    u = np.where(active, potential, 0.0)
    flux_x = gx * (field_strength_v_m * dcx[None, :] + u - np.roll(u, -1, axis=1))
    area = grid.length_x_m * grid.length_y_m
    mean_jx = complex(np.sum(flux_x * dcx[None, :]) / area)
    if ny > 1:
        flux_y = gy * (u - np.roll(u, -1, axis=0))
        mean_jy = complex(np.sum(flux_y * dcy[:, None]) / area)
    else:
        mean_jy = 0.0j
    return AC2DPeriodicResult(
        direction="x",
        effective_conductivity_s_m=mean_jx / field_strength_v_m,
        transverse_conductivity_s_m=mean_jy / field_strength_v_m,
        field_strength_v_m=float(field_strength_v_m),
        residual_norm=residual_norm,
        potential=potential,
        active_mask=active,
        n_unknowns=n_active,
        n_components=int(n_components),
    )


def cell_fields(result: AC2DPeriodicResult, conductivity_s_m: np.ndarray, grid: Grid2D) -> np.ndarray:
    """Cell-centred total field ``E = E0 - grad u`` on conducting cells.

    Face fields are taken only across conducting faces; a cell that touches an
    insulating neighbour uses the one-sided face value, mirroring the pore-side
    finite differences of ``pore_scale_electrical.dynamic_pore_size``.
    Returns an array of shape ``(2, ny, nx)`` (``Ex``, ``Ey``); zero elsewhere.
    """

    if result.direction != "x":
        raise ValueError("cell_fields expects an x-direction solve")
    sigma = np.asarray(conductivity_s_m, dtype=np.complex128)
    gx, dcx, gy, dcy = periodic_face_conductances(sigma, grid)
    u = np.where(result.active_mask, result.potential, 0.0)
    fields = np.zeros((2, *sigma.shape), dtype=np.complex128)
    fields[0] = _centre_component(u, gx, dcx[None, :], result.field_strength_v_m, axis=1)
    if grid.ny > 1:
        fields[1] = _centre_component(u, gy, dcy[:, None], 0.0, axis=0)
    fields[:, ~result.active_mask] = 0.0
    return fields


def _centre_component(u, g, distance, drive, *, axis):
    """Average the conducting forward/backward face fields onto cell centres."""

    conducting = np.abs(g) > 0
    face_field = np.where(conducting, drive + (u - np.roll(u, -1, axis=axis)) / distance, 0.0)
    backward = np.roll(conducting, 1, axis=axis)
    total = face_field + np.roll(face_field, 1, axis=axis)
    count = conducting.astype(float) + backward.astype(float)
    return np.where(count > 0, total / np.maximum(count, 1.0), 0.0)

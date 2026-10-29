"""2-D geometry helpers: dynamic pore size, graded grids and ideal geometries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .solver import AC2DPeriodicResult, Grid2D, cell_fields


@dataclass(frozen=True)
class DynamicPoreSize2D:
    """Niu's field-weighted dynamic pore size, 2 / Lambda = int_S |E_t|^2 / int_V |E|^2."""

    area_integral_v2: float
    wall_integral_v2_per_m: float
    dynamic_pore_size_m: float
    wall_length_m: float
    pore_area_m2: float
    geometric_size_m: float


def wall_faces(pore_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Boolean masks of pore cells whose +x, -x, +y, -y neighbour is solid (periodic)."""

    pore = np.asarray(pore_mask, dtype=bool)
    return (
        pore & ~np.roll(pore, -1, axis=1),
        pore & ~np.roll(pore, 1, axis=1),
        pore & ~np.roll(pore, -1, axis=0),
        pore & ~np.roll(pore, 1, axis=0),
    )


def dynamic_pore_size_2d(
    pore_mask: np.ndarray,
    fields: np.ndarray,
    grid: Grid2D,
    region_mask: np.ndarray | None = None,
) -> DynamicPoreSize2D:
    """2-D port of ``pore_scale_electrical.dynamic_pore_size``.

    Area integral of ``|E|^2`` over pore cells and line integral of the
    pore-side tangential field over pore-solid faces (vertical faces carry
    ``Ey``, horizontal faces carry ``Ex``). ``region_mask`` restricts both
    integrals to one region; faces between two pore regions are internal and
    are not walls.
    """

    pore = np.asarray(pore_mask, dtype=bool)
    region = pore if region_mask is None else (pore & np.asarray(region_mask, dtype=bool))
    energy = np.abs(fields[0]) ** 2 + np.abs(fields[1]) ** 2
    area = grid.cell_area_m2
    area_integral = float(np.sum(energy[region] * area[region]))
    xp, xm, yp, ym = wall_faces(pore)
    dy = np.broadcast_to(grid.dy_m[:, None], pore.shape)
    dx = np.broadcast_to(grid.dx_m[None, :], pore.shape)
    vertical = (xp.astype(float) + xm.astype(float)) * region
    horizontal = (yp.astype(float) + ym.astype(float)) * region
    wall_integral = float(np.sum(vertical * np.abs(fields[1]) ** 2 * dy) + np.sum(horizontal * np.abs(fields[0]) ** 2 * dx))
    wall_length = float(np.sum(vertical * dy) + np.sum(horizontal * dx))
    pore_area = float(np.sum(area[region]))
    if wall_integral <= np.finfo(float).tiny:
        raise ValueError("no tangential field on pore walls")
    return DynamicPoreSize2D(
        area_integral_v2=area_integral,
        wall_integral_v2_per_m=wall_integral,
        dynamic_pore_size_m=2.0 * area_integral / wall_integral,
        wall_length_m=wall_length,
        pore_area_m2=pore_area,
        geometric_size_m=2.0 * pore_area / wall_length,
    )


def dynamic_pore_size_2d_curve(
    result: AC2DPeriodicResult,
    conductivity_s_m: np.ndarray,
    grid: Grid2D,
    wall_points_m: np.ndarray,
    wall_normals: np.ndarray,
    wall_lengths_m: np.ndarray,
    wall_regions: np.ndarray | None = None,
    region_map: np.ndarray | None = None,
    offsets_in_cells: tuple[float, float] = (2.0, 4.0),
) -> dict[int, DynamicPoreSize2D]:
    """Niu's dynamic pore size with the wall integral taken on the true (smooth) wall.

    ``dynamic_pore_size_2d`` samples the tangential field on the staircase
    faces of a voxelized wall; for oblique walls that sample is corrupted
    within about one cell of the wall and the bias does not vanish with
    refinement (+34 % Lambda for a square array of disks, see
    ``code/scripts/sip_simulation/benchmark_ac2d_rayleigh_stern.py``).
    Here ``|E_t|^2`` is evaluated at two distances ``s1 < s2`` inside the pore
    along the wall normal (``s = offset * h_n``, ``h_n`` the cell size along the
    normal), from bilinear interpolation of the correction potential, and
    extrapolated linearly to ``s = 0``.

    ``wall_points_m`` (k, 2) are (x, y) points on the wall, ``wall_normals``
    (k, 2) unit normals pointing into the pore, ``wall_lengths_m`` (k,) the arc
    length each point represents. ``wall_regions`` (k,) and ``region_map``
    (region labels, 0 = solid) give regional values; key 0 is the whole pore
    space. Only fields along x (``result.direction == "x"``) are supported.
    """

    from scipy.interpolate import RegularGridInterpolator

    if result.direction != "x":
        raise ValueError("expects an x-direction solve")
    pts = np.asarray(wall_points_m, dtype=float)
    nrm = np.asarray(wall_normals, dtype=float)
    dl = np.asarray(wall_lengths_m, dtype=float)
    tan = np.c_[-nrm[:, 1], nrm[:, 0]]
    lx, ly = grid.length_x_m, grid.length_y_m
    xc, yc = grid.x_centers_m, grid.y_centers_m
    u = np.where(result.active_mask, result.potential, np.nan + 0.0j)
    u_pad = np.pad(u, 1, mode="wrap")
    interp = RegularGridInterpolator(
        (np.r_[yc[-1] - ly, yc, yc[0] + ly], np.r_[xc[-1] - lx, xc, xc[0] + lx]),
        u_pad, method="linear", bounds_error=False, fill_value=np.nan,
    )

    def local_steps(points):
        i = np.clip(np.searchsorted(grid.x_edges_m, np.mod(points[:, 0], lx)) - 1, 0, grid.nx - 1)
        j = np.clip(np.searchsorted(grid.y_edges_m, np.mod(points[:, 1], ly)) - 1, 0, grid.ny - 1)
        return grid.dx_m[i], grid.dy_m[j]

    def potential(points):
        return interp(np.c_[np.mod(points[:, 1], ly), np.mod(points[:, 0], lx)])

    dx_w, dy_w = local_steps(pts)
    h_normal = np.abs(nrm[:, 0]) * dx_w + np.abs(nrm[:, 1]) * dy_w
    h_tangent = np.abs(tan[:, 0]) * dx_w + np.abs(tan[:, 1]) * dy_w
    e0 = result.field_strength_v_m
    samples = []
    for offset in offsets_in_cells:
        p = pts + (offset * h_normal)[:, None] * nrm
        du = potential(p + h_tangent[:, None] * tan) - potential(p - h_tangent[:, None] * tan)
        et = e0 * tan[:, 0] - du / (2.0 * h_tangent)
        samples.append(np.abs(et) ** 2)
    if np.any(~np.isfinite(samples[0])) or np.any(~np.isfinite(samples[1])):
        bad = int(np.sum(~np.isfinite(samples[0]) | ~np.isfinite(samples[1])))
        raise ValueError(f"{bad} wall samples fall outside the pore space; check normals or offsets")
    s1, s2 = offsets_in_cells
    et2_wall = samples[0] + (samples[0] - samples[1]) * s1 / (s2 - s1)

    fields = cell_fields(result, conductivity_s_m, grid)
    energy = (np.abs(fields[0]) ** 2 + np.abs(fields[1]) ** 2) * grid.cell_area_m2
    pore = result.active_mask
    regions = np.zeros(len(pts), dtype=int) if wall_regions is None else np.asarray(wall_regions)
    labels = [0] + ([] if region_map is None else sorted(int(v) for v in np.unique(regions) if v > 0))
    out: dict[int, DynamicPoreSize2D] = {}
    for label in labels:
        region = pore if label == 0 else pore & (np.asarray(region_map) == label)
        on_wall = np.ones(len(pts), dtype=bool) if label == 0 else regions == label
        area_integral = float(energy[region].sum())
        wall_integral = float(np.sum(et2_wall[on_wall] * dl[on_wall]))
        wall_length = float(dl[on_wall].sum())
        pore_area = float(grid.cell_area_m2[region].sum())
        out[label] = DynamicPoreSize2D(
            area_integral_v2=area_integral,
            wall_integral_v2_per_m=wall_integral,
            dynamic_pore_size_m=2.0 * area_integral / wall_integral,
            wall_length_m=wall_length,
            pore_area_m2=pore_area,
            geometric_size_m=2.0 * pore_area / wall_length,
        )
    return out


def graded_axis(
    length_m: float,
    band_low_m: float,
    band_high_m: float,
    fine_step_m: float,
    coarse_step_m: float,
    growth: float = 1.2,
) -> np.ndarray:
    """Cell sizes on ``[0, length]`` with a uniform fine band ``[low, high]``.

    Band edges are exact cell faces. Cells grow geometrically away from the
    band up to ``coarse_step``; the outermost cell on each side is adjusted so
    that the sizes sum to ``length`` exactly.
    """

    if not 0.0 < band_low_m < band_high_m < length_m:
        raise ValueError("band must lie strictly inside the axis")
    n_fine = max(1, int(round((band_high_m - band_low_m) / fine_step_m)))
    fine = np.full(n_fine, (band_high_m - band_low_m) / n_fine)

    def side(extent: float) -> np.ndarray:
        cells: list[float] = []
        step = fine[0]
        remaining = extent
        while remaining > 1e-15:
            step = min(step * growth, coarse_step_m)
            if remaining < 1.5 * step:
                cells.append(remaining)
                break
            cells.append(step)
            remaining -= step
        return np.asarray(cells)

    below = side(band_low_m)[::-1]
    above = side(length_m - band_high_m)
    return np.r_[below, fine, above]


@dataclass(frozen=True)
class TwoPoreChain2D:
    """Periodic chain ... B - throat - S - throat - B ... along x.

    Region labels: 0 solid, 1 big pore body, 2 small pore body, 3 throat B->S,
    4 throat S->B' (split across the periodic x boundary).
    """

    big_radius_m: float
    small_radius_m: float
    throat_width_m: float
    throat_length_m: float
    solid_margin_m: float

    BIG: ClassVar[int] = 1
    SMALL: ClassVar[int] = 2
    THROAT_BS: ClassVar[int] = 3
    THROAT_SB: ClassVar[int] = 4

    @property
    def period_m(self) -> float:
        return 2.0 * (self.big_radius_m + self.small_radius_m + self.throat_length_m)

    @property
    def height_m(self) -> float:
        return 2.0 * (self.big_radius_m + self.solid_margin_m)

    @property
    def big_center_m(self) -> tuple[float, float]:
        return (0.5 * self.throat_length_m + self.big_radius_m, 0.5 * self.height_m)

    @property
    def small_center_m(self) -> tuple[float, float]:
        return (1.5 * self.throat_length_m + 2.0 * self.big_radius_m + self.small_radius_m, 0.5 * self.height_m)

    def grid(self, coarse_step_m: float, throat_cells: int = 8, growth: float = 1.2) -> Grid2D:
        """Uniform x cells; y graded so the throat walls are exact faces."""

        nx = int(round(self.period_m / coarse_step_m))
        half = 0.5 * self.throat_width_m
        yc = 0.5 * self.height_m
        band = 2.0 * half
        dy = graded_axis(
            self.height_m,
            yc - half - band,
            yc + half + band,
            self.throat_width_m / throat_cells,
            coarse_step_m,
            growth,
        )
        return Grid2D(np.full(nx, self.period_m / nx), dy)

    def region_map(self, grid: Grid2D) -> np.ndarray:
        x = grid.x_centers_m[None, :]
        y = grid.y_centers_m[:, None]
        (xb, yb), (xs, ys) = self.big_center_m, self.small_center_m
        in_big = (x - xb) ** 2 + (y - yb) ** 2 <= self.big_radius_m**2
        in_small = (x - xs) ** 2 + (y - ys) ** 2 <= self.small_radius_m**2
        in_band = np.abs(y - yb) <= 0.5 * self.throat_width_m
        labels = np.zeros(grid.shape, dtype=np.uint8)
        between = in_band & (x > xb) & (x < xs) & ~in_big & ~in_small
        outside = in_band & ((x < xb) | (x > xs)) & ~in_big & ~in_small
        labels[between] = self.THROAT_BS
        labels[outside] = self.THROAT_SB
        labels[in_big] = self.BIG
        labels[in_small] = self.SMALL
        return labels

    def wall_curves(self, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Points on the true pore walls for ``dynamic_pore_size_2d_curve``.

        Returns ``(points, inward_normals, lengths, region_labels)``: circle arcs
        of both pore bodies without the throat mouths, and the two straight walls
        of each throat (throat S->B' wraps across the periodic x boundary).
        """

        h = 0.5 * self.throat_width_m
        yc = 0.5 * self.height_m
        points, normals, lengths, regions = [], [], [], []
        for label, (xo, yo), radius in ((self.BIG, self.big_center_m, self.big_radius_m),
                                        (self.SMALL, self.small_center_m, self.small_radius_m)):
            mouth = np.arcsin(h / radius)
            for lo, hi in ((mouth, np.pi - mouth), (np.pi + mouth, 2.0 * np.pi - mouth)):
                n = max(8, int(np.ceil((hi - lo) * radius / spacing_m)))
                theta = lo + (np.arange(n) + 0.5) * (hi - lo) / n
                points.append(np.c_[xo + radius * np.cos(theta), yo + radius * np.sin(theta)])
                normals.append(-np.c_[np.cos(theta), np.sin(theta)])
                lengths.append(np.full(n, (hi - lo) * radius / n))
                regions.append(np.full(n, label))
        xb_end = self.big_center_m[0] + np.sqrt(self.big_radius_m**2 - h**2)
        xs_start = self.small_center_m[0] - np.sqrt(self.small_radius_m**2 - h**2)
        xs_end = self.small_center_m[0] + np.sqrt(self.small_radius_m**2 - h**2)
        xb_start = self.big_center_m[0] - np.sqrt(self.big_radius_m**2 - h**2) + self.period_m
        for label, x0, x1 in ((self.THROAT_BS, xb_end, xs_start), (self.THROAT_SB, xs_end, xb_start)):
            n = max(8, int(np.ceil((x1 - x0) / spacing_m)))
            x = np.mod(x0 + (np.arange(n) + 0.5) * (x1 - x0) / n, self.period_m)
            for side in (1.0, -1.0):
                points.append(np.c_[x, np.full(n, yc + side * h)])
                normals.append(np.tile([0.0, -side], (n, 1)))
                lengths.append(np.full(n, (x1 - x0) / n))
                regions.append(np.full(n, label))
        return np.vstack(points), np.vstack(normals), np.concatenate(lengths), np.concatenate(regions)

    def analytic_properties(self) -> dict[int, dict[str, float]]:
        """Exact areas, wall lengths, 2A/P and inscribed (local-thickness) radii."""

        h = 0.5 * self.throat_width_m
        out: dict[int, dict[str, float]] = {}
        for label, radius in ((self.BIG, self.big_radius_m), (self.SMALL, self.small_radius_m)):
            half_angle = np.arcsin(h / radius)
            area = np.pi * radius**2
            wall = 2.0 * np.pi * radius - 2.0 * (2.0 * half_angle * radius)
            out[label] = {"area_m2": area, "wall_length_m": wall, "geometric_size_m": 2 * area / wall, "inscribed_radius_m": radius}
        for label in (self.THROAT_BS, self.THROAT_SB):
            length = self.throat_length_m + sum(
                r - np.sqrt(r**2 - h**2) for r in (self.big_radius_m, self.small_radius_m)
            )
            area = self.throat_width_m * length
            out[label] = {"area_m2": area, "wall_length_m": 2.0 * length, "geometric_size_m": self.throat_width_m, "inscribed_radius_m": h}
        return out

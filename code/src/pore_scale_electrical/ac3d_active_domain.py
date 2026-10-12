"""Active-domain helpers for zero-solid AC3D mechanism solves."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def nonwinding_initial_potential(active_mask: np.ndarray, axis: int, step: float, *, allow_partial: bool = False) -> np.ndarray | None:
    """Unwrap isolated periodic components into a gauge-fixed linear potential.

    Return None if any component winds, unless allow_partial keeps those components
    at zero and initializes only the nonwinding components. This is only an
    initial guess: the field solver must still verify its assembled residual.
    """
    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 3 or axis not in (0, 1, 2) or not np.isfinite(step):
        raise ValueError('expected a 3-D mask, axis 0/1/2 and finite step')
    labels, count = ndimage.label(active)
    parent = np.arange(count + 1)
    offset = np.zeros(count + 1, dtype=np.int64)
    winding = np.zeros(count + 1, dtype=bool)

    def find(i):
        chain = []
        root = i
        while parent[root] != root:
            chain.append(root)
            root = int(parent[root])
        distance = 0
        for node in reversed(chain):
            distance += int(offset[node])
            parent[node], offset[node] = root, distance
        return root

    for boundary in range(3):
        lower, upper = np.take(labels, 0, boundary), np.take(labels, -1, boundary)
        mask = (lower > 0) & (upper > 0)
        pairs = np.unique(np.column_stack((lower[mask], upper[mask])), axis=0)
        for a, b in pairs:
            a, b = int(a), int(b)
            ra, rb = find(a), find(b)
            shift = int(offset[a]) + int(boundary == axis) - int(offset[b])
            if ra == rb:
                if shift != 0:
                    if not allow_partial:
                        return None
                    winding[ra] = True
            else:
                parent[rb], offset[rb] = ra, shift
                winding[ra] |= winding[rb]
    roots = np.array([find(i) for i in range(count + 1)])
    components, first = np.unique(labels, return_index=True)
    anchors = {}
    for component, index in zip(components, first):
        if component:
            root = roots[component]
            anchors[root] = min(int(index), anchors.get(root, int(index)))
    shift = -active.shape[axis] * offset.astype(float)
    anchor_values = np.zeros(count + 1)
    for root, index in anchors.items():
        anchor_label = labels.ravel()[index]
        anchor_values[root] = np.unravel_index(index, active.shape)[axis] + shift[anchor_label]
    shift -= anchor_values[roots]
    coordinate_shape = [1, 1, 1]
    coordinate_shape[axis] = active.shape[axis]
    potential = (np.arange(active.shape[axis]).reshape(coordinate_shape) + shift[labels]) * step
    potential[~active] = 0
    potential[winding[roots[labels]]] = 0
    return potential


def active_component_anchor_indices(active_mask: np.ndarray) -> tuple[int, ...]:
    """Return one flattened C-order gauge anchor for each active 6-connected component."""

    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 3:
        raise ValueError("active_mask must be a 3-D boolean array")
    if not np.any(active):
        return ()
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    structure[1, 0, 1] = True
    structure[1, 2, 1] = True
    structure[1, 1, 0] = True
    structure[1, 1, 2] = True
    labels, n_components = ndimage.label(active, structure=structure)
    flat = labels.ravel(order="C")
    components, first_indices = np.unique(flat, return_index=True)
    return tuple(int(first) for component, first in zip(components, first_indices) if component > 0)


class _UnionFind:
    def __init__(self, labels: range) -> None:
        self.parent = {int(label): int(label) for label in labels}

    def find(self, label: int) -> int:
        root = int(label)
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[int(label)] != int(label):
            next_label = self.parent[int(label)]
            self.parent[int(label)] = root
            label = next_label
        return root

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def periodic_active_component_anchor_indices(active_mask: np.ndarray) -> tuple[int, ...]:
    """Return gauge anchors after merging 6-connected components across periodic faces."""

    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 3:
        raise ValueError("active_mask must be a 3-D boolean array")
    if not np.any(active):
        return ()

    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, 1] = True
    structure[0, 1, 1] = True
    structure[2, 1, 1] = True
    structure[1, 0, 1] = True
    structure[1, 2, 1] = True
    structure[1, 1, 0] = True
    structure[1, 1, 2] = True
    labels, n_components = ndimage.label(active, structure=structure)
    uf = _UnionFind(range(1, int(n_components) + 1))

    for axis in range(3):
        if active.shape[axis] <= 1:
            continue
        lower = np.take(labels, 0, axis=axis)
        upper = np.take(labels, -1, axis=axis)
        mask = (lower > 0) & (upper > 0)
        for left, right in zip(lower[mask].ravel(), upper[mask].ravel(), strict=False):
            uf.union(int(left), int(right))

    flat = labels.ravel(order="C")
    component_first_index: dict[int, int] = {}
    components, first_indices = np.unique(flat, return_index=True)
    for component, first_index in zip(components, first_indices):
        if component > 0:
            root = uf.find(int(component))
            first = int(first_index)
            component_first_index[root] = min(first, component_first_index.get(root, first))
    return tuple(sorted(component_first_index.values()))

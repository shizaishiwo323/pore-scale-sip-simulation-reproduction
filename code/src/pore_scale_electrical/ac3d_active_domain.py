"""Active-domain helpers for zero-solid AC3D mechanism solves."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


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
    anchors: list[int] = []
    for component in range(1, int(n_components) + 1):
        indices = np.flatnonzero(flat == component)
        if indices.size:
            anchors.append(int(indices[0]))
    return tuple(anchors)


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
    for component in range(1, int(n_components) + 1):
        indices = np.flatnonzero(flat == component)
        if indices.size:
            root = uf.find(component)
            first = int(indices[0])
            component_first_index[root] = min(first, component_first_index.get(root, first))
    return tuple(sorted(component_first_index.values()))

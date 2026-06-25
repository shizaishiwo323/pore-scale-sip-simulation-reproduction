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

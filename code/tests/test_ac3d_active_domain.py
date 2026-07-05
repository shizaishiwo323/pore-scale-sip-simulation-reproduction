import numpy as np

from pore_scale_electrical.ac3d_active_domain import active_component_anchor_indices, periodic_active_component_anchor_indices


def test_active_component_anchor_indices_returns_one_anchor_per_component():
    active = np.zeros((3, 3, 1), dtype=bool)
    active[0, 0, 0] = True
    active[0, 1, 0] = True
    active[2, 2, 0] = True

    anchors = active_component_anchor_indices(active)

    assert anchors == (0, 8)


def test_active_component_anchor_indices_handles_empty_active_domain():
    active = np.zeros((2, 2, 2), dtype=bool)

    assert active_component_anchor_indices(active) == ()


def test_periodic_active_component_anchor_indices_merges_boundary_component():
    active = np.zeros((3, 3, 1), dtype=bool)
    active[0, 1, 0] = True
    active[2, 1, 0] = True

    anchors = periodic_active_component_anchor_indices(active)

    assert anchors == (1,)

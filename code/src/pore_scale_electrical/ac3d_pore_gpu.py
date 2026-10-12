"""GPU CSR solve for one uniform conducting pore phase and strictly insulating solid."""
import numpy as np
import scipy.sparse as sp
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
from cupyx.scipy.sparse.linalg import cg, LinearOperator
from .ac3d_active_domain import periodic_active_component_anchor_indices, nonwinding_initial_potential
from .ac3d_solver import AC3DIterativeResult, direction_to_axis, axis_to_direction


def build_pore_system(labels, pore_label, direction):
    active = np.asarray(labels == pore_label)
    axis = direction_to_axis(direction)
    indices = np.flatnonzero(active)
    if not len(indices):
        raise ValueError('No pore cells')
    mapping = np.full(active.shape, -1, dtype=np.int32)
    mapping.ravel()[indices] = np.arange(len(indices), dtype=np.int32)
    anchors = mapping.ravel()[list(periodic_active_component_anchor_indices(active))]
    fixed = np.zeros(len(indices), dtype=bool)
    fixed[anchors] = True
    diagonal = np.zeros(len(indices))
    rhs = np.zeros(len(indices))
    rows, cols = [], []
    for d in range(3):
        neighbor = np.roll(mapping, -1, axis=d)
        mask = (mapping >= 0) & (neighbor >= 0)
        i, j = mapping[mask], neighbor[mask]
        diagonal += np.bincount(i, minlength=len(indices)) + np.bincount(j, minlength=len(indices))
        if d == axis:
            rhs = np.bincount(j, minlength=len(indices)).astype(float)-np.bincount(i, minlength=len(indices))
            forward = (cp.asarray(i), cp.asarray(j))
        free_edge = ~(fixed[i] | fixed[j])
        rows.extend([i[free_edge], j[free_edge]])
        cols.extend([j[free_edge], i[free_edge]])
    rhs[fixed] = 0
    diagonal[fixed] = 1
    row, col = np.concatenate(rows), np.concatenate(cols)
    matrix = sp.csr_matrix((-np.ones(len(row)), (row,col)), shape=(len(indices),len(indices)))
    matrix += sp.diags(diagonal, format='csr')
    initial = nonwinding_initial_potential(active, axis, 1.0, allow_partial=True).ravel()[indices]
    return {'matrix':csr_matrix(matrix), 'rhs':cp.asarray(rhs), 'diagonal':cp.asarray(diagonal),
            'initial':cp.asarray(initial), 'forward':forward, 'indices':indices, 'shape':active.shape,
            'axis':axis, 'fixed':cp.asarray(fixed)}


def solve_pore_system(system, water_sigma, voxel_size_m, rtol, maxiter, x0=None,
                      residual_every=100, residual_callback=None, iteration_callback=None):
    matrix, rhs = system['matrix'], system['rhs']
    initial = system['initial'] if x0 is None else cp.asarray(np.asarray(x0).real.ravel()[system['indices']]/voxel_size_m)
    initial = initial.copy()
    initial[system['fixed']] = 0
    rhsnorm = float(cp.linalg.norm(rhs).get())
    history = []
    iterations = 0
    def residual(x):
        return float(cp.linalg.norm(rhs-matrix@x).get())/(rhsnorm or 1.0)
    def callback(x):
        nonlocal iterations
        iterations += 1
        if iteration_callback: iteration_callback(iterations)
        if residual_every and iterations % residual_every == 0:
            value = residual(x)
            history.append((iterations,value))
            if residual_callback: residual_callback(iterations,value)
    preconditioner = LinearOperator(matrix.shape, matvec=lambda x:x/system['diagonal'],dtype=np.float64)
    solution, info = cg(matrix,rhs,x0=initial,tol=rtol,maxiter=maxiter,M=preconditioner,callback=callback)
    value = residual(solution)
    history.append((iterations,value))
    if residual_callback: residual_callback(iterations,value)
    i,j = system['forward']
    effective = complex(water_sigma)*float(cp.sum(1-(solution[j]-solution[i])).get())/np.prod(system['shape'])
    potential = np.zeros(system['shape'],dtype=np.float64)
    potential.ravel()[system['indices']] = cp.asnumpy(solution)*voxel_size_m
    passed = value <= rtol
    return AC3DIterativeResult(direction=axis_to_direction(system['axis']),effective_conductivity_s_m=effective,
        mean_current_density_a_m2=effective,field_strength_v_m=1.,residual_norm=value,potential=potential,
        solver='gpu_active_pore_csr_cg_normalized',iterations=iterations,info=0 if passed else (info or -20),
        residual_history=tuple(history),true_residual_norm=value,true_residual_passed=passed)

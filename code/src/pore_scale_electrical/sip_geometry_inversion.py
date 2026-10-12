"""Constrained effective-geometry inversion, using the existing Niu kernels.

This is a quasistatic bulk surrogate, not a reconstruction of a 3-D network.
Pore mass is normalized; throat coefficients are area times normalized mass.
Area is conditional on fixed D, eta, Lambda, formation factor and kernel choice.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .polarization import (
    PolarizationParameters, elementary_pore_conductance,
    elementary_membrane_conductance_perturbation, throat_zdc_from_length_area,
)


@dataclass
class InversionSettings:
    parameters: PolarizationParameters = field(default_factory=PolarizationParameters)
    formation_factor: float = 16.3
    bins: int = 36
    radius_range_m: tuple = (1e-7, 1e-3)
    length_range_m: tuple = (1e-8, 1e-3)
    area_range_m2: tuple = (1e-13, 1e-8)
    real_relative_scale: float = 0.02
    imag_relative_scale: float = 0.05
    real_floor_s_m: float = 1e-6
    imag_floor_s_m: float = 1e-7
    epsilon_range: tuple = (1., 100.)
    dc_range_s_m: tuple = (0., 0.01)
    regularization_area_scale_m2: float = 3e-11


def validate_data(f, real, imag):
    f, real, imag = (np.asarray(a, dtype=float) for a in (f, real, imag))
    if any(a.ndim != 1 for a in (f, real, imag)) or not (len(f) == len(real) == len(imag)):
        raise ValueError('Frequency, real and imaginary arrays must be matching 1-D arrays.')
    if len(f) < 6 or not all(np.isfinite(a).all() for a in (f, real, imag)):
        raise ValueError('At least six finite complex observations are required.')
    if np.any(f <= 0) or np.any(real <= 0) or np.any(imag <= 0) or len(np.unique(f)) != len(f):
        raise ValueError('Unique positive frequencies and positive conductivities required; audit excluded data explicitly.')
    return f, real, imag


class GeometryKernel:
    area_unit = 1e-11
    dc_unit = 0.0025
    epsilon_unit = 40.

    def __init__(self, frequency_hz, settings=None):
        self.settings = settings or InversionSettings()
        c = self.settings
        if c.bins < 5 or c.formation_factor <= 0 or min(c.radius_range_m+c.length_range_m+c.area_range_m2) <= 0:
            raise ValueError('Positive geometry ranges/formation factor and at least five bins required.')
        if any(lo >= hi for lo, hi in (c.radius_range_m, c.length_range_m, c.area_range_m2, c.epsilon_range,c.dc_range_s_m)):
            raise ValueError('Bounds must be increasing.')
        if min(c.real_relative_scale,c.imag_relative_scale,c.real_floor_s_m,c.imag_floor_s_m) <= 0:
            raise ValueError('Residual scales must be positive.')
        p = c.parameters
        if not 0 < p.membrane_polarizability < 1 or min(p.dynamic_pore_size_m,p.diffusion_coefficient_m2_s,p.water_conductivity_s_m,p.surface_conductance_s,c.regularization_area_scale_m2) <= 0:
            raise ValueError('Positive material/scaling parameters and membrane polarizability in (0,1) required.')
        if c.dc_range_s_m[0] < 0 or c.epsilon_range[0] <= 0:
            raise ValueError('Nonnegative DC and positive dielectric bounds required.')
        self.f = np.asarray(frequency_hz, dtype=float)
        if self.f.ndim != 1 or not np.isfinite(self.f).all() or np.any(self.f <= 0):
            raise ValueError('Positive finite frequency vector required.')
        self.radius = np.geomspace(*c.radius_range_m, c.bins)
        self.length = np.geomspace(*c.length_range_m, c.bins)
        self.n = c.bins
        p = c.parameters
        factor = 2/(p.dynamic_pore_size_m*c.formation_factor)
        kp = elementary_pore_conductance(self.f, self.radius, p)*factor
        zdc = throat_zdc_from_length_area(self.length, np.full(c.bins,self.area_unit),p.water_conductivity_s_m)
        km = elementary_membrane_conductance_perturbation(self.f,self.length,zdc,p)*factor
        self.design = np.column_stack([kp, km, np.full(len(self.f),self.dc_unit),
                                      1j*2*np.pi*self.f*p.epsilon_0_f_m*self.epsilon_unit])
        # Smooth bin masses on equal log grids. The q scale is a fixed prior,
        # not a fitted area; its effect is explored by the regularization scan.
        d = np.diff(np.eye(self.n),n=2,axis=0)*self.n
        self.reg = np.zeros((2*(self.n-2),2*self.n+2))
        self.reg[:self.n-2,:self.n] = d
        self.reg[self.n-2:,self.n:2*self.n] = d/(c.regularization_area_scale_m2/self.area_unit)

    def residual_system(self, real, imag, indices=None):
        indices = np.arange(len(self.f)) if indices is None else np.asarray(indices)
        c = self.settings
        sr = np.maximum(np.abs(real)*c.real_relative_scale,c.real_floor_s_m)
        si = np.maximum(np.abs(imag)*c.imag_relative_scale,c.imag_floor_s_m)
        matrix = np.vstack([self.design.real[indices]/sr[indices,None], self.design.imag[indices]/si[indices,None]])
        target = np.r_[real[indices]/sr[indices],imag[indices]/si[indices]]
        return matrix, target

    def fit(self, real, imag, alpha, indices=None, fixed_area_m2=None, initial=None):
        _, real, imag = validate_data(self.f,real,imag)
        if alpha < 0: raise ValueError('Regularization must be nonnegative.')
        m,y = self.residual_system(real,imag,indices)
        normalizer = len(y)**0.5
        m = np.vstack([m/normalizer,np.sqrt(alpha)*self.reg])
        y = np.r_[y/normalizer,np.zeros(len(self.reg))]
        h = m.T@m
        g = m.T@y
        n = self.n
        constraint = np.zeros((2,2*n+2))
        constraint[0,:n] = 1
        constraint[1,n:2*n] = 1
        amin,amax = np.array(self.settings.area_range_m2)/self.area_unit
        if fixed_area_m2 is not None:
            if not self.settings.area_range_m2[0] <= fixed_area_m2 <= self.settings.area_range_m2[1]:
                raise ValueError('Profile area outside configured bounds.')
            amin = amax = fixed_area_m2/self.area_unit
        lower = np.zeros(2*n+2)
        upper = np.r_[np.ones(n),np.full(n,amax),self.settings.dc_range_s_m[1]/self.dc_unit,self.settings.epsilon_range[1]/self.epsilon_unit]
        lower[-2] = self.settings.dc_range_s_m[0]/self.dc_unit
        lower[-1] = self.settings.epsilon_range[0]/self.epsilon_unit
        x = np.r_[np.full(n,1/n),np.full(n,np.clip(3.,amin,amax)/n),np.median(real)/self.dc_unit,1.]
        if initial is not None:
            x = np.array(initial,copy=True)
            x[n:2*n] *= np.clip(x[n:2*n].sum(),amin,amax)/x[n:2*n].sum()
        x = np.clip(x,lower,upper)
        fit = minimize(lambda z: .5*z@h@z-g@z,x,jac=lambda z:h@z-g,method='SLSQP',
                       bounds=Bounds(lower,upper),constraints=[LinearConstraint(constraint[:1],1.,1.),
                                                              LinearConstraint(constraint[1:],amin,amax)],
                       options={'ftol':1e-10,'maxiter':2000})
        if not fit.success:
            raise RuntimeError(f'Constrained inversion failed: {fit.message}')
        x = fit.x
        if abs(x[:n].sum()-1)>1e-6 or np.min(x)<-1e-9:
            raise RuntimeError('Inversion violates nonnegative normalized distribution constraints.')
        pred = self.design@x
        matrix,target = self.residual_system(real,imag)
        area = x[n:2*n].sum()*self.area_unit
        return {'x':x, 'prediction':pred,'area_m2':area,'pore_mass':x[:n],
                'throat_mass':x[n:2*n]*self.area_unit/area,
                'sigma_dc_s_m':x[-2]*self.dc_unit,'epsilon_effective':x[-1]*self.epsilon_unit,
                'weighted_rmse':float(np.sqrt(np.mean((matrix@x-target)**2))),
                'real_relative_rmse':float(np.sqrt(np.mean(((pred.real-real)/real)**2))),
                'imag_relative_rmse':float(np.sqrt(np.mean(((pred.imag-imag)/imag)**2))),
                'objective':float(np.mean((m@x-y)**2)),'iterations':fit.nit}

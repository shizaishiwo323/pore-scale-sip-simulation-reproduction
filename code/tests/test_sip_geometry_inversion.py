from dataclasses import replace
import numpy as np
import pytest

from pore_scale_electrical.sip_geometry_inversion import GeometryKernel, InversionSettings, validate_data
from pore_scale_electrical.polarization import (
    pore_polarization_conductance, membrane_polarization_conductance,
    throat_zdc_from_length_area,
)


def test_joint_dictionary_matches_independent_forward_and_fits_synthetic_spectrum():
    f = np.geomspace(.002,45000,36)
    c = InversionSettings(bins=12)
    k = GeometryKernel(f,c)
    p = np.exp(-.5*((np.log10(k.radius)+4.5)/.4)**2); p /= p.sum()
    w = np.exp(-.5*((np.log10(k.length)+5)/.5)**2); w /= w.sum()
    area = 2.8e-11
    x = np.r_[p,w*area/k.area_unit,1.,1.]
    z = throat_zdc_from_length_area(k.length,np.full(k.n,area),c.parameters.water_conductivity_s_m)
    forward = .0025 + 2/(c.parameters.dynamic_pore_size_m*c.formation_factor)*(
        pore_polarization_conductance(f,k.radius,p,c.parameters)+
        membrane_polarization_conductance(f,k.length,w,z,c.parameters)) + 1j*2*np.pi*f*c.parameters.epsilon_0_f_m*40
    np.testing.assert_allclose(k.design@x,forward,rtol=1e-12)
    fit = k.fit(forward.real,forward.imag,0.,fixed_area_m2=area)
    assert fit['weighted_rmse'] < .001
    assert np.isclose(fit['area_m2'],area,rtol=1e-8)
    assert np.isclose(fit['pore_mass'].sum(),1) and np.isclose(fit['throat_mass'].sum(),1)


def test_diffusion_geometry_scaling_is_nonunique_and_invalid_data_rejected():
    c = InversionSettings(bins=8)
    f = np.geomspace(.002,45000,20)
    k = GeometryKernel(f,c)
    c2 = replace(c,parameters=replace(c.parameters,diffusion_coefficient_m2_s=4*c.parameters.diffusion_coefficient_m2_s),
                 radius_range_m=tuple(2*np.array(c.radius_range_m)),length_range_m=tuple(2*np.array(c.length_range_m)))
    k2 = GeometryKernel(f,c2)
    x = np.r_[np.full(8,1/8),np.full(8,3/8),1.,1.]
    x2 = x.copy(); x2[8:16] *= 2
    np.testing.assert_allclose(k.design@x,k2.design@x2,rtol=1e-12)
    with pytest.raises(ValueError): validate_data([1,1,2,3,4,5],np.ones(6),np.ones(6))
    with pytest.raises(ValueError): validate_data(f,np.ones(20),np.full(20,np.nan))

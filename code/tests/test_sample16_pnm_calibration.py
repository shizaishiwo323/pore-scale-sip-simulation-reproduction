from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "sip_simulation"
sys.path.insert(0, str(SCRIPT_DIR))


def test_fit_linear_complex_surrogate_recovers_synthetic_coefficients():
    from calibrate_sample16_fullres_pnm_parameters import fit_linear_complex_surrogate

    frequency_hz = np.array([0.1, 0.7, 3.0, 20.0, 130.0, 1000.0], dtype=float)
    omega = 2.0 * np.pi * frequency_hz
    pore = np.array(
        [
            1.0e-4 + 2.0e-5j,
            2.4e-4 + 1.2e-5j,
            1.5e-4 + 7.0e-5j,
            3.3e-4 + 3.1e-5j,
            2.8e-4 + 9.0e-5j,
            4.2e-4 + 5.5e-5j,
        ]
    )
    membrane = np.array(
        [
            2.0e-4 + 1.0e-5j,
            0.8e-4 + 5.0e-5j,
            3.0e-4 + 2.0e-5j,
            1.2e-4 + 8.0e-5j,
            3.8e-4 + 4.0e-5j,
            2.7e-4 + 1.1e-4j,
        ]
    )

    real_coeff = np.array([5.0e-3, 1.7, -0.4, 2.0e-4])
    imag_coeff = np.array([1.0e-6, 2.4, 0.6, 3.0e-8])
    logf = np.log10(frequency_hz) - np.log10(frequency_hz).mean()

    target_real = real_coeff[0] + real_coeff[1] * pore.real + real_coeff[2] * membrane.real + real_coeff[3] * logf
    target_imag = imag_coeff[0] + imag_coeff[1] * pore.imag + imag_coeff[2] * membrane.imag + imag_coeff[3] * omega

    fit = fit_linear_complex_surrogate(frequency_hz, target_real, target_imag, pore, membrane)

    assert np.allclose(fit["model_real_s_m"], target_real, rtol=1e-10, atol=1e-10)
    assert np.allclose(fit["model_imag_s_m"], target_imag, rtol=1e-10, atol=1e-10)
    assert np.allclose(fit["real_coefficients"], real_coeff, rtol=1e-8, atol=1e-8)
    assert np.allclose(fit["imag_coefficients"], imag_coeff, rtol=1e-8, atol=1e-8)


def test_fit_debye_dictionary_surrogate_recovers_synthetic_curve():
    from calibrate_sample16_fullres_pnm_parameters import debye_response, fit_debye_dictionary_surrogate

    frequency_hz = np.logspace(-1, 5, 40)
    centers_hz = np.array([10.0, 3000.0])
    coeff = np.array([2.0e-4, 7.0e-4])
    response = debye_response(frequency_hz, centers_hz) @ coeff
    logf = np.log10(frequency_hz) - np.log10(frequency_hz).mean()
    target_real = 5.0e-3 + 2.0e-5 * logf + response.real
    target_imag = 1.0e-6 + 4.0e-12 * (2.0 * np.pi * frequency_hz) + response.imag

    fit = fit_debye_dictionary_surrogate(frequency_hz, target_real, target_imag, centers_hz)

    assert np.allclose(fit["model_real_s_m"], target_real, rtol=1e-10, atol=1e-10)
    assert np.allclose(fit["model_imag_s_m"], target_imag, rtol=1e-10, atol=1e-10)


def test_fit_positive_debye_dictionary_surrogate_keeps_amplitudes_nonnegative():
    from calibrate_sample16_fullres_pnm_parameters import fit_positive_debye_dictionary_surrogate

    frequency_hz = np.logspace(-1, 4, 30)
    centers_hz = np.array([3.0, 200.0, 5000.0])
    target_real = np.full_like(frequency_hz, 5.0e-3)
    target_imag = 2.0e-5 * np.exp(-((np.log10(frequency_hz) - 2.3) ** 2) / 0.5)

    fit = fit_positive_debye_dictionary_surrogate(
        frequency_hz,
        target_real,
        target_imag,
        centers_hz,
        max_amplitude_s_m=1.0e-3,
    )

    assert np.all(np.asarray(fit["debye_amplitudes_s_m"]) >= -1.0e-12)

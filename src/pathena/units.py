"""Shared TIGRESS unit constants and conversion helpers."""

import numpy as np


DEFAULT_MUH = 1.4271
PC_CGS = 3.0856775814913674e18
MP_CGS = 1.67262192369e-24
KB_CGS = 1.380649e-16
KM_CGS = 1.0e5
MSUN_CGS = 1.98847e33
MYR_CGS = 365.25 * 24.0 * 3600.0 * 1.0e6


def tigress_units(muH=DEFAULT_MUH):
    """Return conversion factors for the standard TIGRESS code units."""
    energy_density = muH * MP_CGS * KM_CGS**2
    return {
        "energy_density": energy_density,
        "pressure_over_kB": energy_density / KB_CGS,
        "temperature_per_p_over_rho": energy_density / KB_CGS,
        "magnetic_field_microgauss": np.sqrt(4.0 * np.pi * energy_density) * 1.0e6,
    }


def star_particle_units(muH=DEFAULT_MUH):
    """Return code-unit factors needed for star-particle plotting."""
    dunit = muH * MP_CGS
    return {
        "mass_msun": dunit * PC_CGS**3 / MSUN_CGS,
        "time_myr": PC_CGS / KM_CGS / MYR_CGS,
    }

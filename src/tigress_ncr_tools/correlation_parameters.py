"""Shared environmental predictors for suite correlation products."""

import numpy as np


ENVIRONMENT_PARAMETER_SPECS = (
    (
        "stellar_surface_density",
        r"$\Sigma_*$",
        r"$\Sigma_*\ [M_\odot\,{\rm pc}^{-2}]$",
        "log",
    ),
    (
        "stellar_scale_height",
        r"$H_*$",
        r"$H_*\ [{\rm pc}]$",
        "log",
    ),
    (
        "omega",
        r"$\Omega$",
        r"$\Omega\ [{\rm Myr}^{-1}]$",
        "log",
    ),
    ("qshear", r"$q$", r"$q$", "linear"),
    (
        "kappa",
        r"$\kappa$",
        r"$\kappa=\sqrt{2(2-q)}\,\Omega\ [{\rm Myr}^{-1}]$",
        "log",
    ),
    (
        "stellar_midplane_density",
        r"$\rho_*$",
        r"$\rho_*=\Sigma_*/(2H_*)\ [M_\odot\,{\rm pc}^{-3}]$",
        "log",
    ),
    (
        "mean_sfr10",
        r"$\langle\Sigma_{\rm SFR,10}\rangle$",
        r"$\langle\Sigma_{\rm SFR,10}\rangle_{200-600}$ "
        r"$[M_\odot\,{\rm kpc}^{-2}\,{\rm yr}^{-1}]$",
        "log",
    ),
)


def parameter_axis_limits(values, scale, margin=0.05):
    """Return data-only limits with symmetric padding in plotting space."""
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if scale == "log":
        valid &= values > 0.0
    values = values[valid]
    if values.size == 0:
        raise ValueError("parameter axis requires at least one valid value")
    if scale == "log":
        values = np.log10(values)
    lower = float(np.min(values))
    upper = float(np.max(values))
    span = upper - lower
    padding = margin * span if span > 0.0 else max(abs(lower) * margin, margin)
    limits = (lower - padding, upper + padding)
    if scale == "log":
        return tuple(10.0**limit for limit in limits)
    return limits

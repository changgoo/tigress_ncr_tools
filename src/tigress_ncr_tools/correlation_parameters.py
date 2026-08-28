"""Shared environmental predictors for suite correlation products."""


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

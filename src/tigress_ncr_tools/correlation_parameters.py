"""Shared environmental predictors for suite correlation products."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def spearman_coefficient(x, y):
    """Return the finite-pair Spearman coefficient and sample count."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    count = int(np.count_nonzero(valid))
    if count < 3:
        return np.nan, count
    x_rank = pd.Series(x[valid]).rank(method="average")
    y_rank = pd.Series(y[valid]).rank(method="average")
    return float(x_rank.corr(y_rank)), count


def plot_annotated_correlation_matrix(
    matrix,
    row_labels,
    column_labels,
    output,
    *,
    title,
    dpi=180,
    figsize=None,
):
    """Plot one annotated Spearman matrix with a fixed diverging scale."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (len(row_labels), len(column_labels)):
        raise ValueError("correlation matrix shape does not match its labels")
    if figsize is None:
        figsize = (max(7.0, 1.25 * len(column_labels)), 2.2 + 0.6 * len(row_labels))
    figure, axis = plt.subplots(figsize=figsize)
    image = axis.imshow(
        matrix,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(column_labels)), column_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    for row, column in np.ndindex(matrix.shape):
        value = matrix[row, column]
        axis.text(
            column,
            row,
            "--" if not np.isfinite(value) else f"{value:+.2f}",
            ha="center",
            va="center",
            fontsize=8.5,
            color=(
                "0.45"
                if not np.isfinite(value)
                else "white" if abs(value) > 0.55 else "black"
            ),
        )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, fraction=0.045)
    colorbar.set_label(r"Spearman $\rho_s$ across models")
    figure.suptitle(title, fontsize=13, y=0.985)
    figure.subplots_adjust(left=0.16, right=0.91, bottom=0.08, top=0.80)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)
    print(f"Wrote {output}", flush=True)

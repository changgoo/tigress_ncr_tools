"""Derived physical fields for TIGRESS-NCR slice output."""

import warnings

import numpy as np

from .units import DEFAULT_MUH, tigress_units


LEGACY_SPECIFIC_SCALAR_ALIASES = {
    # Current R8/TIGRESS-NCR config: NSCALARS=5, HI/H2/EL species enabled.
    "xHI": "specific_scalar_2",
    "xH2": "specific_scalar_3",
    "xe": "specific_scalar_4",
}


def require_slice_field(fields, name, plane="unknown"):
    """Return a raw slice field, accepting known legacy scalar aliases."""
    if name in fields:
        return fields[name]
    legacy = LEGACY_SPECIFIC_SCALAR_ALIASES.get(name)
    if legacy in fields:
        return fields[legacy]
    available = ", ".join(sorted(fields))
    raise KeyError(
        f"plane {plane} is missing field {name!r}; available: {available}"
    )


def optional_slice_field(fields, name, template, plane="unknown"):
    """Return an optional field or a matching all-NaN array with a warning."""
    if name in fields:
        return fields[name]
    warnings.warn(
        f"plane {plane} is missing field {name!r}; using an all-NaN array",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.full_like(template, np.nan, dtype=float)


def slice_vector_component(fields, vector_name, component, plane="unknown"):
    """Return one component from a raw three-component slice vector."""
    vector = require_slice_field(fields, vector_name, plane=plane)
    if vector.ndim != 3 or vector.shape[-1] <= component:
        raise ValueError(
            f"plane {plane} field {vector_name!r} is not a 3-component vector"
        )
    return vector[..., component]


def derive_slice_fields(fields, plane="unknown", muH=DEFAULT_MUH):
    """Derive the physical scalar fields used by slice visualizations.

    Parameters
    ----------
    fields : mapping
        Raw fields from one entry of ``read_slicevtk(...)["planes"]``.
    plane : str
        Plane name used in diagnostic errors and warnings.
    muH : float
        Hydrogen mass per H nucleus in proton masses.

    Returns
    -------
    dict
        Physical arrays including ``nH``, ``nH2``, ``nHI``, ``nHII``, ``T``,
        ``P``, velocity components, radiation energy densities, and magnetic
        components/magnitude when available.
    """
    units = tigress_units(muH)
    nH = require_slice_field(fields, "density", plane=plane)
    xH2 = require_slice_field(fields, "xH2", plane=plane)
    xHI = require_slice_field(fields, "xHI", plane=plane)
    xe = fields.get(
        "xe",
        fields.get(LEGACY_SPECIFIC_SCALAR_ALIASES["xe"], np.zeros_like(nH)),
    )
    pressure = require_slice_field(fields, "pressure", plane=plane)
    denominator = nH * (1.1 + xe - xH2)

    velocity = require_slice_field(fields, "velocity", plane=plane)
    if velocity.ndim != 3 or velocity.shape[-1] < 3:
        raise ValueError(
            f"plane {plane} field 'velocity' is not a 3-component vector"
        )

    out = {
        "nH": nH,
        "nH2": nH * xH2,
        "nHI": nH * xHI,
        "P": pressure * units["pressure_over_kB"],
        "v1": velocity[..., 0],
        "v2": velocity[..., 1],
        "v3": velocity[..., 2],
        # Retain the established summary-plot name.
        "vz": velocity[..., 2],
        "Erad_PE": optional_slice_field(
            fields, "rad_energy_density_PE", nH, plane=plane
        ) * units["energy_density"],
        "Erad_PH": optional_slice_field(
            fields, "rad_energy_density_PH", nH, plane=plane
        ) * units["energy_density"],
    }
    out["nHII"] = out["nH"] - out["nHI"] - 2.0 * out["nH2"]
    out["T"] = np.divide(
        pressure * units["temperature_per_p_over_rho"],
        denominator,
        out=np.full_like(pressure, np.nan, dtype=float),
        where=denominator != 0.0,
    )

    if "cell_centered_B" in fields:
        magnetic = fields["cell_centered_B"]
        if magnetic.ndim != 3 or magnetic.shape[-1] < 3:
            raise ValueError(
                f"plane {plane} field 'cell_centered_B' is not a "
                "3-component vector"
            )
        magnetic = magnetic[..., :3] * units["magnetic_field_microgauss"]
        out.update({
            "B1": magnetic[..., 0],
            "B2": magnetic[..., 1],
            "B3": magnetic[..., 2],
            "Bmag": np.sqrt(np.sum(magnetic**2, axis=-1)),
        })
    return out


def derive_plane_fields(slc, plane, muH=DEFAULT_MUH):
    """Derive physical fields for a named plane in a slicevtk frame."""
    return derive_slice_fields(slc["planes"][plane]["fields"], plane, muH=muH)

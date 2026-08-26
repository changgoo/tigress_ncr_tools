"""Definitions for projected quantities shared by PDF and spectrum tools."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectedQuantity:
    """Metadata needed to analyze one positive face-on projected map."""

    key: str
    field: str
    slug: str
    label: str
    symbol: str
    physical_unit: str
    delta_range: tuple
    s_range: tuple


PROJECTED_QUANTITIES = {
    "gas": ProjectedQuantity(
        key="gas",
        field="nH",
        slug="density",
        label="gas column",
        symbol=r"\Sigma",
        physical_unit="H nuclei cm^-2 after multiplying by pc in cm",
        delta_range=(-1.0, 30.0),
        s_range=(-6.0, 4.0),
    ),
    "hi": ProjectedQuantity(
        key="hi",
        field="nHI",
        slug="hi",
        label="H I column",
        symbol=r"\Sigma_{\rm HI}",
        physical_unit="H nuclei cm^-2 after multiplying by pc in cm",
        delta_range=(-1.0, 30.0),
        s_range=(-24.0, 5.0),
    ),
    "em": ProjectedQuantity(
        key="em",
        field="ne_sq",
        slug="em",
        label="emission measure",
        symbol=r"{\rm EM}",
        physical_unit="pc cm^-6",
        delta_range=(-1.0, 300.0),
        s_range=(-12.0, 8.0),
    ),
}

FIELD_TO_QUANTITY = {item.field: item for item in PROJECTED_QUANTITIES.values()}


def projected_quantity(quantity="gas"):
    """Return validated metadata for a projected quantity key."""
    try:
        return PROJECTED_QUANTITIES[str(quantity).lower()]
    except KeyError as error:
        choices = ", ".join(PROJECTED_QUANTITIES)
        raise ValueError(
            f"unknown projected quantity {quantity!r}; choose {choices}"
        ) from error


def projected_quantity_from_field(field):
    """Return metadata for a stored proj2d field name."""
    try:
        return FIELD_TO_QUANTITY[str(field)]
    except KeyError as error:
        choices = ", ".join(sorted(FIELD_TO_QUANTITY))
        raise ValueError(
            f"unsupported proj2d field {field!r}; choose {choices}"
        ) from error


def archive_scalar(data, name, default=None):
    """Return a scalar string from an archive-like mapping."""
    if name not in data:
        return default
    value = data[name]
    try:
        return str(value.item())
    except AttributeError:
        return str(value)


def archive_projected_quantity(data):
    """Resolve quantity metadata from new or backward-compatible archives."""
    key = archive_scalar(data, "quantity")
    if key is not None:
        return projected_quantity(key)
    field = archive_scalar(data, "field", "nH")
    return projected_quantity_from_field(field)

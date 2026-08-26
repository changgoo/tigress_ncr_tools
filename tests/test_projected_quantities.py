import numpy as np
import pytest

from tigress_ncr_tools.projected_quantities import (
    archive_projected_quantity,
    projected_quantity,
)


def test_projected_quantity_fields_and_pdf_ranges():
    assert projected_quantity("gas").field == "nH"
    assert projected_quantity("hi").field == "nHI"
    assert projected_quantity("em").field == "ne_sq"
    assert projected_quantity("hi").s_range[0] < projected_quantity("gas").s_range[0]
    assert (
        projected_quantity("em").delta_range[1]
        > projected_quantity("gas").delta_range[1]
    )


def test_archive_quantity_supports_new_and_legacy_metadata():
    assert archive_projected_quantity({"quantity": np.asarray("em")}).key == "em"
    assert archive_projected_quantity({"field": np.asarray("nHI")}).key == "hi"
    assert archive_projected_quantity({}).key == "gas"
    with pytest.raises(ValueError):
        projected_quantity("mystery")

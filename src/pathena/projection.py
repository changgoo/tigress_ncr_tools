"""NumPy helpers for shear-periodic line-of-sight projections.

The main entry point, :func:`project_shearing_periodic`, assumes raw data cubes
are already loaded into memory. Arrays are interpreted as cell-centered values
on a uniform Cartesian mesh.
"""

import numpy as np


def basis_vectors(theta, phi, degrees=True):
    """Return LOS and image-plane basis vectors for TIGRESS projections."""
    if degrees:
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)

    n_los = np.array([st * cp, st * sp, ct])
    e_imgx = np.array([-sp, cp, 0.0])
    e_imgy = np.array([-ct * cp, -ct * sp, st])
    return n_los, e_imgx, e_imgy

def _as_zyx(arr, axis_order):
    arr = np.asarray(arr)
    if axis_order == "zyx":
        return arr
    if axis_order == "xyz":
        return np.transpose(arr, (2, 1, 0))
    raise ValueError("axis_order must be 'zyx' or 'xyz'")


def _wrap_periodic(x, xlo, length):
    return xlo + np.mod(x - xlo, length)


def _expand_fields(fields, vectors):
    expanded = []
    for name in fields:
        name = name.strip()
        matched = False
        if "*" not in name:
            for base in vectors:
                if name == base + "vec" or name == base + "_vec":
                    expanded.extend([base + "los", base + "imgx", base + "imgy"])
                    matched = True
                    break
        if not matched:
            expanded.append(name)
    return expanded


def _parse_vector_alias(name, vectors):
    for base in vectors:
        for suffix in ("los", "los2", "imgx", "imgy"):
            if name == base + suffix or name == base + "_" + suffix:
                return base, suffix
    return None


def _parse_field(name, scalars, vectors):
    if name in scalars:
        return ("scalar", name, None)
    if "*" in name:
        lhs, rhs = [part.strip() for part in name.split("*", 1)]
        vec = _parse_vector_alias(rhs, vectors)
        if lhs in scalars and vec is not None:
            return ("product", lhs, vec)
    vec = _parse_vector_alias(name, vectors)
    if vec is not None:
        base, suffix = vec
        return ("vector", base, suffix)
    known = sorted(scalars) + [base + s for base in vectors for s in
                               ("los", "los2", "imgx", "imgy")]
    known += [base + "vec" for base in vectors]
    known += ["scalar*" + base + s for base in vectors for s in
              ("los", "los2", "imgx", "imgy")]
    raise KeyError("unknown projection field {!r}; known fields include {}".format(name, known))


def project_shearing_periodic(
    scalars,
    bounds,
    fields=None,
    vectors=None,
    theta=0.0,
    phi=0.0,
    degrees=True,
    time=0.0,
    qshear=1.0,
    Omega0=1.0,
    nbinx=None,
    nbiny=None,
    binx_min=None,
    binx_max=None,
    biny_min=None,
    biny_max=None,
    dl_factor=0.5,
    los_length=None,
    axis_order="zyx",
):
    """Project scalar and vector cubes along a shearing-periodic LOS.

    Parameters
    ----------
    scalars : ndarray or dict[str, ndarray]
        Scalar cell-centered cubes. Arrays are shape ``(nz, ny, nx)`` by
        default. A bare ndarray is named ``"field"``.
    bounds : sequence
        ``((xlo, xhi), (ylo, yhi), (zlo, zhi))`` in code units.
    fields : sequence[str], optional
        Output field names. Defaults to all scalar names. Vector aliases are
        ``Vlos``, ``Vlos2``, ``Vimgx``, ``Vimgy`` for a vector named ``"V"``;
        ``los2`` is the squared LOS component and works for every vector. Aliases with
        underscores such as ``V_los`` are also accepted. A shorthand such as
        ``Vvec`` expands to ``Vlos,Vimgx,Vimgy``. Products such as
        ``ne*Blos`` multiply a scalar cube by a projected vector component at
        each sample before line integration.
    vectors : dict[str, tuple[ndarray, ndarray, ndarray]], optional
        Vector components in simulation x/y/z coordinates.
    theta, phi : float
        LOS angles. ``theta`` is measured from +z; ``phi`` is the azimuth in
        the x-y plane from +x toward +y.
    time, qshear, Omega0 : float
        Parameters for ``deltay = (qshear * Omega0 * Lx * time) mod Ly``.
    nbinx, nbiny : int, optional
        Image size. Defaults to the input cube nx, ny.
    bin*_min, bin*_max : float, optional
        Image footprint ranges in code units. Defaults to x/y domain bounds.
    dl_factor : float
        Sample spacing in units of the smallest cell size.
    los_length : float, optional
        Required for nearly horizontal rays; otherwise rays integrate through
        the finite z slab.
    axis_order : {'zyx', 'xyz'}
        Memory order of the input arrays.

    Returns
    -------
    dict
        Keys include ``maps`` (dict of 2D arrays), ``x_edges``, ``y_edges``,
        ``x_centers``, ``y_centers``, ``n_los``, ``e_imgx``, ``e_imgy``, and
        ``deltay``.
    """
    if isinstance(scalars, np.ndarray):
        scalars = {"field": scalars}
    scalars = {name: _as_zyx(arr, axis_order) for name, arr in scalars.items()}
    if not scalars:
        raise ValueError("at least one scalar cube is required")

    shape = next(iter(scalars.values())).shape
    for name, arr in scalars.items():
        if arr.shape != shape:
            raise ValueError("scalar cube {!r} has shape {}, expected {}".format(name, arr.shape, shape))

    vectors = vectors or {}
    vectors = {
        name: tuple(_as_zyx(comp, axis_order) for comp in comps)
        for name, comps in vectors.items()
    }
    for name, comps in vectors.items():
        if len(comps) != 3:
            raise ValueError("vector {!r} must have three components".format(name))
        for comp in comps:
            if comp.shape != shape:
                raise ValueError("vector {!r} component has shape {}, expected {}".format(name, comp.shape, shape))

    nz, ny, nx = shape
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = np.asarray(bounds, dtype=float)
    lx, ly = xhi - xlo, yhi - ylo
    dx, dy, dz = lx / nx, ly / ny, (zhi - zlo) / nz
    if min(dx, dy, dz) <= 0.0:
        raise ValueError("bounds must be increasing in every direction")
    if dl_factor <= 0.0:
        raise ValueError("dl_factor must be positive")

    nbinx = nx if nbinx is None else int(nbinx)
    nbiny = ny if nbiny is None else int(nbiny)
    binx_min = xlo if binx_min is None else float(binx_min)
    binx_max = xhi if binx_max is None else float(binx_max)
    biny_min = ylo if biny_min is None else float(biny_min)
    biny_max = yhi if biny_max is None else float(biny_max)
    if nbinx <= 0 or nbiny <= 0:
        raise ValueError("nbinx and nbiny must be positive")

    x_edges = np.linspace(binx_min, binx_max, nbinx + 1)
    y_edges = np.linspace(biny_min, biny_max, nbiny + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    n_los, e_imgx, e_imgy = basis_vectors(theta, phi, degrees=degrees)
    center = np.array([0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)])
    deltay = np.mod(qshear * Omega0 * lx * time, ly)

    if fields is None:
        fields = list(scalars)
    fields = _expand_fields(fields, vectors)
    parsed = [_parse_field(name, scalars, vectors) for name in fields]
    maps = {name: np.zeros((nbiny, nbinx), dtype=np.float64) for name in fields}

    base_dl = dl_factor * min(dx, dy, dz)
    if abs(n_los[2]) < 1.0e-12:
        if los_length is None:
            raise ValueError("los_length is required for nearly horizontal rays")
        slab_smin, slab_smax = -0.5 * los_length, 0.5 * los_length

    for iy_img, v in enumerate(y_centers):
        for ix_img, u in enumerate(x_centers):
            r0 = center + u * e_imgx + v * e_imgy
            if abs(n_los[2]) < 1.0e-12:
                smin, smax = slab_smin, slab_smax
            else:
                s_a = (zlo - r0[2]) / n_los[2]
                s_b = (zhi - r0[2]) / n_los[2]
                smin, smax = min(s_a, s_b), max(s_a, s_b)
            nsamp = int(np.ceil((smax - smin) / base_dl))
            if nsamp <= 0:
                continue
            dl = (smax - smin) / nsamp

            for isamp in range(nsamp):
                s = smin + (isamp + 0.5) * dl
                x, y, z = r0 + s * n_los
                if z < zlo or z >= zhi:
                    continue
                m = np.floor((x - xlo) / lx)
                xw = x - m * lx
                yw = _wrap_periodic(y + m * deltay, ylo, ly)

                ix = int(np.floor((xw - xlo) / dx))
                iy = int(np.floor((yw - ylo) / dy))
                iz = int(np.floor((z - zlo) / dz))
                if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
                    continue

                for out_name, spec in zip(fields, parsed):
                    kind, base, comp = spec
                    if kind == "scalar":
                        value = scalars[base][iz, iy, ix]
                    else:
                        if kind == "product":
                            scalar_name = base
                            vec_base, comp = comp
                        else:
                            scalar_name = None
                            vec_base = base
                        vx, vy, vz = vectors[vec_base]
                        vec = np.array([vx[iz, iy, ix], vy[iz, iy, ix], vz[iz, iy, ix]])
                        if comp == "los":
                            value = np.dot(vec, n_los)
                        elif comp == "los2":
                            value = np.dot(vec, n_los)**2
                        elif comp == "imgx":
                            value = np.dot(vec, e_imgx)
                        else:
                            value = np.dot(vec, e_imgy)
                        if scalar_name is not None:
                            value *= scalars[scalar_name][iz, iy, ix]
                    maps[out_name][iy_img, ix_img] += value * dl

    return {
        "maps": maps,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "x_centers": x_centers,
        "y_centers": y_centers,
        "n_los": n_los,
        "e_imgx": e_imgx,
        "e_imgy": e_imgy,
        "theta": theta,
        "phi": phi,
        "degrees": degrees,
        "deltay": deltay,
        "bounds": ((xlo, xhi), (ylo, yhi), (zlo, zhi)),
    }

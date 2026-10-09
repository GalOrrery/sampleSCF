# -*- coding: utf-8 -*-

"""Local Utilities."""


__all__ = ["plot_corner_samples", "log_prior", "log_prob"]


##############################################################################
# IMPORTS

# STDLIB
from typing import Optional, Tuple, Union

# THIRD PARTY
import astropy.units as u
import corner
from astropy.coordinates import BaseRepresentation, CartesianRepresentation
from galpy.potential import SCFPotential

# PROJECT-SPECIFIC
from matplotlib.figure import Figure
from numpy import abs, arctan2, floating, inf, isfinite, log, nan_to_num, ndarray, sign, sqrt
from numpy import square, sum

##############################################################################
# CODE
##############################################################################


def plot_corner_samples(
    samples: Union[BaseRepresentation, ndarray],
    r_limit: float = 1_000.0,
    *,
    figs: Optional[Tuple[Figure, Figure]] = None,
    include_log: bool = True,
    **kw,
) -> Tuple[Figure, Figure]:
    """Plot samples.

    Parameters
    ----------
    *samples : BaseRepresentation or (N, 3) ndarray
        If an `numpy.ndarray`, samples should be in Cartesian coordinates.
    r_limit : float
        Largerst radius that should be plotted.
        Values larger will be masked.
    figs : tuple[Figure, Figure] or None, optional keyword-only
    include_log : bool, optional keyword-only

    Returns
    -------
    tuple[Figure, Figure]
    """
    # Convert to ndarray
    arr: ndarray
    if isinstance(samples, BaseRepresentation):
        arr = samples.represent_as(CartesianRepresentation)._values.view(float).reshape(-1, 3)
    else:
        arr = samples

    # Correcting for large r
    r = sqrt(sum(square(arr), axis=1))
    mask = r <= r_limit

    # plot stuff
    truths = [0, 0, 0]
    hist_kwargs = {"density": True}
    hist_kwargs.update(kw.pop("hist_kwargs", {}))
    kw.pop("plot_contours", None)
    kw.pop("plot_density", None)

    # -----------
    # normal plot

    labels = ["x", "y", "z"]

    fig1 = corner.corner(
        arr[mask, :],
        labels=labels,
        raster=True,
        bins=50,
        truths=truths,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 13},
        hist_kwargs=hist_kwargs,
        plot_contours=False,
        plot_density=False,
        fig=None if figs is None else figs[0],
        **kw,
    )
    fig1.suptitle("Samples")

    # -----------
    # logarithmic plot

    if not include_log:
        fig2 = None

    else:

        labels = [r"$\log_{10}(x)$", r"$\log_{10}(y)$", r"$\log_{10}(z)$"]

        fig2 = corner.corner(
            nan_to_num(sign(arr) * log(abs(arr))),
            labels=labels,
            raster=True,
            bins=50,
            truths=truths,
            show_titles=True,
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 13},
            fig=None if figs is None else figs[1],
            plot_contours=False,
            plot_density=False,
            hist_kwargs=hist_kwargs,
            **kw,
        )
        fig2.suptitle("Samples")

    return fig1, fig2


def log_prior(R: floating, r_limit: floating) -> floating:
    """Log-Prior.

    Parameters
    ----------
    R : float
    r_limit : float

    Returns
    -------
    float
    """
    # outside
    if r_limit is not None and R > r_limit:
        return -inf
    return 0.0


def log_prob(
    x: ndarray, /, pot: SCFPotential, rho0: u.Quantity, r_limit: Optional[floating] = None
) -> floating:
    """Log-Probability.

    Parameters
    ----------
    x : (3, ) array
        Cartesian coordinates in kpc
    pot : `galpy.potential.SCFPotential`
    rho0 : Quantity
        The central density.
    r_limit : float

    Returns
    -------
    float
    """
    # Checks
    if rho0 == 0:
        raise ValueError("`mtot` cannot be 0.")
    elif r_limit == 0:
        raise ValueError("`r_limit` cannot be 0.")

    # convert Cartesian to Cylindrical coordinates
    R = sqrt(sum(square(x)))
    z = x[-1]
    phi = arctan2(x[1], x[0])

    # calculate log-prior
    lp = log_prior(R, r_limit)
    if not isfinite(lp):
        return lp

    # the density as a likelihood
    logrho0 = log(rho0.value)
    dens = pot.dens(R, z, phi).to_value(rho0.unit)

    logdens = nan_to_num(log(dens), copy=False, nan=logrho0, posinf=logrho0)
    ll = logdens - logrho0  # normalize the density

    return lp + ll

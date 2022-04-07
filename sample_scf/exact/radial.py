# -*- coding: utf-8 -*-

"""Exact sampling of radial coordinate."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import abc
from typing import Any, Optional, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import PhysicsSphericalRepresentation
from astropy.units import Quantity
from galpy.potential import SCFPotential
from numpy.typing import ArrayLike

# LOCAL
from sample_scf._typing import NDArrayF, RandomLike
from sample_scf.base_multivariate import SCFSamplerBase
from sample_scf.base_univariate import r_distribution_base

__all__ = ["exact_r_distribution"]


##############################################################################
# CODE
##############################################################################


class exact_r_distribution(r_distribution_base):
    """Sample radial coordinate from an SCF potential.

    Parameters
    ----------
    pot : `~galpy.potential.SCFPotential`
        A potential that can be used to calculate the enclosed mass.
    total_mass : Optional
    **kw
        Not used.
    """

    def __init__(
        self, potential: SCFPotential, total_mass: Optional[Quantity] = None, **kw: Any
    ) -> None:
        # make sampler
        kw["a"], kw["b"] = 0, np.inf  # allowed range of r
        super().__init__(potential, **kw)

        # normalization for total mass
        # TODO! if mass has units
        if total_mass is None:
            total_mass = potential._mass(np.inf)
        if np.isnan(total_mass):
            raise ValueError(
                "total mass is NaN. Need to pass kwarg `total_mass` with a non-NaN value.",
            )
        self._mtot = total_mass
        # vectorize mass function, which is scalar
        self._vec_cdf = np.vectorize(self._potential._mass)

    def _cdf(self, r: Quantity, *args: Any, **kw: Any) -> NDArrayF:
        """Cumulative Distribution Function.

        Parameters
        ----------
        r : Quantity ['length']
        *args
        **kwargs

        Returns
        -------
        mass : array-like
            Shape matches 'r'.
        """
        mass: NDArrayF = np.atleast_1d(self._vec_cdf(r)) / self._mtot
        mass[r == 0] = 0
        mass[r == np.inf] = 1
        return mass.item() if mass.shape == (1,) else mass

    cdf = _cdf

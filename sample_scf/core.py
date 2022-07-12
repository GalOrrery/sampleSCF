# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from typing import Any, Literal, Optional, Type, TypedDict, Union

# THIRD PARTY
from galpy.potential import SCFPotential

# LOCAL
from .base_multivariate import SCFSamplerBase
from .base_univariate import rv_potential
from .exact import ExactSCFSampler
from .interpolated import InterpolatedSCFSampler

__all__ = ["SCFSampler"]


##############################################################################
# Parameters


class MethodsMapping(TypedDict):
    r: rv_potential
    theta: rv_potential
    phi: rv_potential


##############################################################################
# CODE
##############################################################################


class SCFSampler(SCFSamplerBase):
    """Sample SCF in spherical coordinates.

    The coordinate system is:
    - r : [0, infinity)
    - theta : [0, pi]  (0 at the North pole)
    - phi : [0, 2pi)

    Parameters
    ----------
    pot : `galpy.potential.SCFPotential`
    method : {'interp', 'exact'} or mapping[str, type]
        If mapping, must have keys (r, theta, phi)
    **kwargs
        Passed to to the individual component sampler constructors.
    """

    _sampler: Optional[SCFSamplerBase]

    def __init__(
        self,
        potential: SCFPotential,
        method: Union[Literal["interp", "exact"], MethodsMapping],
        **kwargs: Any,
    ) -> None:
        super().__init__(potential, **kwargs)

        if isinstance(method, Mapping):  # mix and match exact and interpolated
            sampler = None
            r_distribution = method["r"](potential, **kwargs)
            theta_distribution = method["theta"](potential, **kwargs)
            phi_distribution = method["phi"](potential, **kwargs)

        else:  # either exact or interpolated
            sampler_cls: Type[SCFSamplerBase]
            if method == "interp":
                sampler_cls = InterpolatedSCFSampler
            elif method == "exact":
                sampler_cls = ExactSCFSampler
            else:
                raise ValueError(f"method = {method} not in " + "{'interp', 'exact'}")

            sampler = sampler_cls(potential, **kwargs)
            r_distribution = sampler.r_distribution
            theta_distribution = sampler.theta_distribution
            phi_distribution = sampler.phi_distribution

        self._sampler: Optional[SCFSamplerBase] = sampler
        self._r_distribution = r_distribution
        self._theta_distribution = theta_distribution
        self._phi_distribution = phi_distribution

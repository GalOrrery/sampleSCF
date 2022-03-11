# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
from collections.abc import Mapping
from typing import Any, Literal, Optional, Type, TypedDict, Union

# THIRD PARTY
from galpy.potential import SCFPotential

# LOCAL
from .base import SCFSamplerBase, rv_potential
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


class SCFSampler(SCFSamplerBase):  # metaclass=SCFSamplerSwitch
    """Sample SCF in spherical coordinates.

    The coordinate system is:
    - r : [0, infinity)
    - theta : [-pi/2, pi/2]  (positive at the North pole)
    - phi : [0, 2pi)

    Parameters
    ----------
    pot : `galpy.potential.SCFPotential`
    method : {'interp', 'exact'} or mapping[str, type]
        If mapping, must have keys (r, theta, phi)
    **kwargs
        Passed to to the individual component sampler constructors.
    """

    def __init__(
        self,
        potential: SCFPotential,
        method: Union[Literal["interp", "exact"], MethodsMapping],
        **kwargs: Any,
    ) -> None:
        super().__init__(potential)

        if isinstance(method, Mapping):  # mix and match exact and interpolated
            sampler = None
            rsampler = method["r"](potential, **kwargs)
            thetasampler = method["theta"](potential, **kwargs)
            phisampler = method["phi"](potential, **kwargs)

        else:  # either exact or interpolated
            sampler_cls: Type[SCFSamplerBase]
            if method == "interp":
                sampler_cls = InterpolatedSCFSampler
            elif method == "exact":
                sampler_cls = ExactSCFSampler
            else:
                raise ValueError(f"method = {method} not in " + "{'interp', 'exact'}")

            sampler = sampler_cls(potential, **kwargs)
            rsampler = sampler.rsampler
            thetasampler = sampler.thetasampler
            phisampler = sampler.phisampler

        self._sampler: Optional[SCFSamplerBase] = sampler
        self._r_distribution = rsampler
        self._theta_distribution = thetasampler
        self._phi_distribution = phisampler

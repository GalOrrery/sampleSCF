# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# BUILT-IN
import typing as T
from collections.abc import Mapping

# THIRD PARTY
from galpy.potential import SCFPotential

# LOCAL
from .base import SCFSamplerBase, rv_continuous_modrvs
from .sample_exact import SCFSampler as SCFSamplerExact
from .sample_intrp import SCFSampler as SCFSamplerIntrp

__all__: T.List[str] = ["SCFSampler"]


##############################################################################
# Parameters


class MethodsMapping(T.TypedDict):
    r: rv_continuous_modrvs
    theta: rv_continuous_modrvs
    phi: rv_continuous_modrvs


##############################################################################
# CODE
##############################################################################


# class SCFSamplerSwitch(ABCMeta):
#     def __new__(
#         cls: T.Type[SCFSamplerSwitch],
#         name: str,
#         bases: T.Tuple[type, ...],
#         dct: T.Dict[str, T.Any],
#         **kwds: T.Any
#     ) -> SCFSamplerSwitch:
#
#         method: str = dct["method"]
#
#         if method == "interp":
#             # LOCAL
#             from sample_scf.sample_intrp import SCFSampler as interpcls
#
#             bases = (interpcls,)
#
#         elif method == "exact":
#             # LOCAL
#             from sample_scf.sample_exact import SCFSampler as exactcls
#
#             bases = (exactcls,)
#         elif isinstance(method, Mapping):
#             pass
#         else:
#             raise ValueError("`method` must be {'interp', 'exact'} or mapping.")
#
#         self = super().__new__(cls, name, bases, dct)
#         return self
#
#     # /def


# /class


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
        pot: SCFPotential,
        method: T.Union[T.Literal["interp", "exact"], MethodsMapping],
        **kwargs: T.Any
    ) -> None:
        super().__init__(pot)

        if isinstance(method, Mapping):
            sampler = None
            rsampler = method["r"](pot, **kwargs)
            thetasampler = method["theta"](pot, **kwargs)
            phisampler = method["phi"](pot, **kwargs)
        else:
            sampler_cls: rv_continuous_modrvs
            if method == "interp":
                sampler_cls = SCFSamplerIntrp
            elif method == "exact":
                sampler_cls = SCFSamplerExact

            sampler = sampler_cls(pot, **kwargs)
            rsampler = self._sampler._rsampler
            thetasampler = self._sampler._thetasampler
            phisampler = self._sampler._phisampler

        self._sampler: T.Optional[SCFSamplerBase] = sampler
        self._rsampler = rsampler
        self._thetasampler = thetasampler
        self._phisampler = phisampler

    # /def


# /class

##############################################################################
# END

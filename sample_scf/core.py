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
from .base import SCFSamplerBase

__all__: T.List[str] = ["SCFSampler"]


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

    #     def __new__(
    #         cls,
    #         pot: SCFPotential,
    #         *args: T.Any,
    #         method: T.Union[T.Literal["interp", "exact"], T.Mapping[str, T.Callable]] = "interp",
    #         **kwargs: T.Any
    #     ) -> SCFSamplerBase:
    #
    #         self: SCFSamplerBase
    #         if method == "interp":
    #             # LOCAL
    #             from sample_scf.sample_intrp import SCFSampler as interpcls
    #
    #             self = interpcls(pot, *args, method=method, **kwargs)
    #         elif method == "exact":
    #             # LOCAL
    #             from sample_scf.sample_exact import SCFSampler as exactcls
    #
    #             self = exactcls(pot, *args, method=method, **kwargs)
    #         elif isinstance(method, Mapping):
    #             self = super().__new__(cls)
    #         else:
    #             raise ValueError("`method` must be {'interp', 'exact'} or mapping.")
    #
    #         return self
    #
    #     # /def

    def __init__(
        self, pot: SCFPotential, method: T.Literal["interp", "exact"], **kwargs: T.Any
    ) -> None:
        if not isinstance(method, Mapping):
            raise NotImplementedError

        self._rsampler = method["r"](pot, **kwargs)
        self._thetasampler = method["theta"](pot, **kwargs)
        self._phisampler = method["phi"](pot, **kwargs)

    # /def


# /class

##############################################################################
# END

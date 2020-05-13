"""
An implementation of Gaussian-Process Surrogate Optimisation - a tool for
optimizing large-scale biophysical networks.

Hadida, J., Sotiropoulos, S. N., Abeysuriya, R. G., Woolrich, M. W., & Jbabdi,
    S. (2018). Bayesian Optimisation of Large-Scale Biophysical Networks.
    NeuroImage, 174, 219-236.
"""

from .gp_surrogate import GPRSurrogate
from .optimisation import GPSOptimiser
from .param_space import ParameterSpace

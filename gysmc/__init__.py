# SPDX-License-Identifier: MIT

"""
magnet_config_GYSlib - Magnetic Configuration Library for GYSELA

This module provides classes for initializing and computing magnetic configurations
for tokamak simulations.

Classes:
    - MagnetConfig: Abstract base class for magnetic configurations
    - CircularMagnetConfig: Circular concentric flux surfaces configuration
    - CulhamMagnetConfig: Culham equilibrium configuration
    - GEQDSKMagnetConfig: GEQDSK file-based magnetic configuration
    - QProfile: Safety factor profile generator
    - PressureProfile: Pressure profile generator
"""

from .magnet_config import MagnetConfig
from .circular_magnetconfig import CircularMagnetConfig
from .culham_magnetconfig import CulhamMagnetConfig
from .geqdsk_magnetconfig import GEQDSKMagnetConfig
from .q_profile import QProfile
from .pressure_profile import PressureProfile
from .GYSmagnet_config import GYSMagnetConfig

__all__ = [
    'MagnetConfig',
    'CircularMagnetConfig',
    'CulhamMagnetConfig',
    'GEQDSKMagnetConfig',
    'QProfile',
    'PressureProfile',
    'GYSMagnetConfig',
]

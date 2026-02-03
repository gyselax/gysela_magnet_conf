# SPDX-License-Identifier: MIT

"""
Created on 2025-11-03

 Contains the class to initialise the magnetic configuration for circular concentric flux surfaces

@author: Z. S. Qu
"""

import numpy as np

from .magnet_config import MagnetConfig


class CircularMagnetConfig(MagnetConfig):
    """
    Class to initialise the magnetic configuration
    tor1_arr is the normalised minor radius r/a
    """

    def __init__(self, q_profile, aspect_ratio=3.0, thetastar=True):
        """
        Initialisation of the magnetic configuration
        :param q_profile: array of q profile
        :param aspect_ratio: aspect ratio of the tokamak
        :param thetastar: boolean indicating if theta* is used
        """
        self.q_profile = q_profile
        self.aspect_ratio = aspect_ratio
        self.thetastar = thetastar
        self.thetastar = thetastar

    def get_RZ(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the R and Z coordinates from toroidal coordinates
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: tuple of (R, Z) coordinates
        """
        R0 = self.aspect_ratio
        if not self.thetastar:
            R_coord = R0 + tor1_arr[:, None, None] * np.cos(tor2_arr[None, :, None]) + 0 * tor3_arr[None, None, :]
            Z_coord = tor1_arr[:, None, None] * np.sin(tor2_arr[None, :, None]) + 0 * tor3_arr[None, None, :]
        else:
            theta = self.get_theta_from_thetastar_(tor1_arr, tor2_arr)
            R_coord = R0 + tor1_arr[:, None, None] * np.cos(theta[:, :, None]) + 0 * tor3_arr[None, None, :]
            Z_coord = tor1_arr[:, None, None] * np.sin(theta[:, :, None]) + 0 * tor3_arr[None, None, :]
        return R_coord, Z_coord

    def get_gij(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the gij coefficients from toroidal coordinates
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: tuple of (ContravariantMetricTensor, CovariantMetricTensor)
        """
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_cocontra = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3, 3)
        CovariantMetricTensor = np.zeros(shape_cocontra, dtype=float)

        CovariantMetricTensor[..., 0, 0] = 1
        CovariantMetricTensor[..., 1, 1] = (tor1_arr[:, None, None]) ** 2
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        CovariantMetricTensor[..., 2, 2] = R**2

        if self.thetastar:
            Jacobimat = self.get_jacobi_theta_thetastar_(tor1_arr, tor2_arr, tor3_arr)
            CovariantMetricTensor = np.einsum("...ki,...kl,...lj->...ij", Jacobimat, CovariantMetricTensor, Jacobimat)

        ContravariantMetricTensor = np.linalg.inv(CovariantMetricTensor)
        return ContravariantMetricTensor, CovariantMetricTensor

    def get_Psi(self, tor1_arr):
        """
        Get Psi, dPsidr
        :param tor1_arr: array of toroidal coordinates 1
        :return: Psi, dPsidr
        """
        from scipy.interpolate import CubicSpline
        from scipy.integrate import cumtrapz

        Nrint = 1024
        rint = np.linspace(0, np.max(tor1_arr), Nrint, True)
        qint = self.get_q(rint)
        dPsidrint = rint / qint
        Psiint = cumtrapz(dPsidrint, rint, initial=0)
        cb = CubicSpline(rint, Psiint, bc_type=((1, 0), (1, dPsidrint[-1])))
        Psi = cb(tor1_arr)

        q = self.get_q(tor1_arr)
        dPsidr = tor1_arr / q

        return Psi, dPsidr

    def get_q(self, tor1_arr):
        """
        Get q
        :param tor1_arr: array of toroidal coordinates 1
        :return: q
        """
        qGYS, _ = self.q_profile.get_q(tor1_arr)
        if self.thetastar:
            q = qGYS / np.sqrt(1 - tor1_arr**2 / self.aspect_ratio**2)
        else:
            q = qGYS
        return q

    def get_Bcontra(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the magnetic field from toroidal coordinates, assuming self.ds_geometry is already initialised
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: B_contra magnetic field components
        """
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_B = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3)
        B_contra = np.zeros(shape_B, dtype=float)

        R0 = self.aspect_ratio
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        q_arr, _ = self.q_profile.get_q(tor1_arr)

        # B^r = 0
        B_contra[..., 0] = 0
        # B^\theta
        B_contra[..., 1] = 1 / R / q_arr[:, None, None]
        # B^\phi
        B_contra[..., 2] = R0 / R**2

        if self.thetastar:
            Jacobimat = self.get_jacobi_theta_thetastar_(tor1_arr, tor2_arr, tor3_arr)
            B_contra = np.linalg.solve(Jacobimat, B_contra[...])

        return B_contra

    def get_Jcontra(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the current density from toroidal coordinates, assuming self.ds_geometry is already initialised
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: J_contra current density components
        """
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_J = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3)
        J_contra = np.zeros(shape_J, dtype=float)

        theta = self.get_theta_from_thetastar_(tor1_arr, tor2_arr)
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        qGYS, dqGYSdr = self.q_profile.get_q(tor1_arr)

        # J^r = 0
        J_contra[..., 0] = 0
        # J^\theta
        J_contra[..., 1] = 0
        # J^\phi
        r = tor1_arr
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        factor1 = 1 / qGYS[:, None, None] / R**2
        if self.thetastar:
            theta = self.get_theta_from_thetastar_(tor1_arr, tor2_arr)
        else:
            theta = tor2_arr[None, :] + tor1_arr[:, None] * 0
        factor2 = 2 - r / qGYS * dqGYSdr
        factor2 = factor2[:, None, None] - r[:, None, None] * np.cos(theta[:, :, None]) / R
        J_contra[..., 2] = factor1 * factor2

        return J_contra

    def get_jacobi_theta_thetastar_(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Get the Jacobian between theta and theta*
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: Jacobi matrix
        """
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_cocontra = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3, 3)
        Jacobimat = np.zeros(shape_cocontra, dtype=float)

        R0 = self.aspect_ratio
        factor_sqrt = np.sqrt((R0 + tor1_arr) / (R0 - tor1_arr))
        Rminus = R0 - tor1_arr[:, None] * np.cos(tor2_arr[None, :])
        dtheta_dr = (R0 / Rminus) * np.sin(tor2_arr[None, :]) * factor_sqrt[:, None] / (R0 + tor1_arr[:, None])
        dtheta_dthetastar = (R0 + tor1_arr[:, None]) / Rminus / factor_sqrt[:, None]

        Jacobimat[..., 0, 0] = 1
        Jacobimat[..., 1, 0] = dtheta_dr[:, :, None]
        Jacobimat[..., 1, 1] = dtheta_dthetastar[:, :, None]
        Jacobimat[..., 2, 2] = 1

        return Jacobimat

    def get_theta_from_thetastar_(self, tor1_arr, tor2_arr):
        """
        Get theta from theta*
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :return: theta
        """
        R0 = self.aspect_ratio
        theta = 2 * np.arctan(
            np.tan(tor2_arr[None, :] / 2) * np.sqrt((R0 + tor1_arr[:, None]) / (R0 - tor1_arr[:, None]))
        )
        return theta

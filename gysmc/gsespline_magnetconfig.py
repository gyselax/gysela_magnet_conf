# SPDX-License-Identifier: MIT

"""
Created on 2025-11-08

 Contains the class to initialise the magnetic configuration from GSE solution

@author: Z. S. Qu
"""

import numpy as np
from abc import abstractmethod
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, CubicSpline
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz

from .magnet_config import MagnetConfig


class GSEMagnetConfig(MagnetConfig):
    """
    Class to initialise the magnetic configuration from a Grad-Shafranov equilibrium data
    
    The __init__ method must be implemented in the child class to load the specific data format.
    1. The child class must set the following attributes:
        - self.cb_R_r : CubicSpline object for R(r, theta), spline is over r, theta is a parameter
        - self.cb_Z_r : CubicSpline object for Z(r, theta), spline is over r, theta is a parameter
        - self.theta_vals : array of theta values used in the splines
        - self.q_cb : CubicSpline object for q(r)
        - self.F_cb : CubicSpline object for F(r)
        - self.FFprime_cb : CubicSpline object for FF'(r)
        - self.pprime_cb : CubicSpline object for p'(r)
        - self.psi1_norm : normalization factor for psi1
        - self.s_at_r_one : s value at r=1
    2. The child class must call super().__init__(cocos, verbose) to set the cocos convention.
    3. The child class can implement additional methods as needed.
    """
    @abstractmethod
    def __init__(self, cocos=2, verbose=True):
        """
        Initialisation of the magnetic configuration
        :param cocos: COCOS convection of the input, see [Sauter O, Medvedev SY. CPC 2013]
        :param verbose: whether to print verbose information

        see also set_cocos_convention method for a list of supported cocos conventions
        the output uses cocos = 2 convention (GYSELA/CHEASE standard)
        """

        self.set_cocos_convention(cocos, verbose=verbose) # set cocos convention

    def set_cocos_convention(self, cocos, verbose=True):
        """
        Set the cocos convention for the magnetic configuration
        :param cocos: COCOS convection of the input, see [Sauter O, Medvedev SY. CPC 2013]
        :param verbose: whether to print verbose information

        COCOS convention:
            2/12 : CHEASE/GYSELA standard (default) (R,Z,phi), (r,theta,phi), signma_phi = +1
            5/15 : AUG (R,phi,Z), (r,phi,theta), sigma_phi = -1
        """
        if cocos == 2 or cocos == 12:
            self.signma_phi = 1
            if verbose:
                print("INFO: cocos = 2/12, (R,Z,phi), (r,theta,phi)")
        elif cocos == 5 or cocos == 15:
            self.signma_phi = -1
            if verbose:
                print("INFO: cocos = 5/15, (R,phi,Z), (r,phi,theta)")
        else:
            raise ValueError("Unsupported COCOS convention: {}".format(cocos))

    def _compute_dRZ_drt_(self, r, theta):
        """
        Compute the derivatives of R and Z with respect to r and theta
        :param r: minor radius
        :param theta: poloidal angle
        :return: tuple of derivatives (dR_dr, dR_dtheta, dZ_dr, dZ_dtheta)
        """

        R_coord_theta_interp = self.cb_R_r(r)
        Z_coord_theta_interp = self.cb_Z_r(r)
        dRdr = self.cb_R_r(r, nu=1)
        dZdr = self.cb_Z_r(r, nu=1)

        cb_R = CubicSpline(self.theta_vals, R_coord_theta_interp, axis=1, bc_type='periodic')
        cb_Z = CubicSpline(self.theta_vals, Z_coord_theta_interp, axis=1, bc_type='periodic')
        cb_dRdr = CubicSpline(self.theta_vals, dRdr, axis=1, bc_type='periodic')
        cb_dZdr = CubicSpline(self.theta_vals, dZdr, axis=1, bc_type='periodic')

        dR_dr = cb_dRdr(theta)
        dZ_dr = cb_dZdr(theta)
        dR_dtheta = cb_R(theta, nu=1)
        dZ_dtheta = cb_Z(theta, nu=1)

        return dR_dr, dR_dtheta, dZ_dr, dZ_dtheta

    def get_RZ(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the R and Z coordinates from toroidal coordinates
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: tuple of (R, Z) coordinates
        """
        r = tor1_arr
        theta = tor2_arr
        R_coord_theta_interp = self.cb_R_r(r)
        Z_coord_theta_interp = self.cb_Z_r(r)

        cb_R = CubicSpline(self.theta_vals, R_coord_theta_interp, axis=1, bc_type='periodic')
        cb_Z = CubicSpline(self.theta_vals, Z_coord_theta_interp, axis=1, bc_type='periodic')

        R_coord = cb_R(theta)[:,:,None] + tor3_arr[None, None, :] * 0
        Z_coord = cb_Z(theta)[:,:,None] + tor3_arr[None, None, :] * 0

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

        dRdr, dRdtheta, dZdr, dZdtheta = self._compute_dRZ_drt_(tor1_arr, tor2_arr)

        # extend to 3D
        dRdr = dRdr[:, :, None] + tor3_arr[None, None, :] * 0
        dZdr = dZdr[:, :, None] + tor3_arr[None, None, :] * 0
        dRdtheta = dRdtheta[:, :, None] + tor3_arr[None, None, :] * 0
        dZdtheta = dZdtheta[:, :, None] + tor3_arr[None, None, :] * 0

        CovariantMetricTensor[..., 0, 0] = dRdr**2 + dZdr**2
        CovariantMetricTensor[..., 0, 1] = dRdr * dRdtheta + dZdr * dZdtheta
        CovariantMetricTensor[..., 1, 0] = CovariantMetricTensor[..., 0, 1]
        CovariantMetricTensor[..., 1, 1] = dRdtheta**2 + dZdtheta**2
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        CovariantMetricTensor[..., 2, 2] = R**2

        ContravariantMetricTensor = np.linalg.inv(CovariantMetricTensor)
        return ContravariantMetricTensor, CovariantMetricTensor

    def get_Psi(self, tor1_arr):
        """
        Get Psi, dPsidr
        :param tor1_arr: array of toroidal coordinates 1
        :return: Psi, dPsidr
        """
        r = tor1_arr
        s = r * self.s_at_r_one
        Psi = s**2 * self.psi1_norm * self.signma_phi
        dPsidr = 2 * s * self.s_at_r_one * self.psi1_norm * self.signma_phi

        return Psi, dPsidr

    def get_q(self, tor1_arr):
        """
        Get q
        :param tor1_arr: array of toroidal coordinates 1
        :return: q
        """

        return self.q_cb(tor1_arr) * self.signma_phi


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

        # Compute all transformation derivatives using vectorized function
        # Returns arrays of shape (Nr, Ntheta)
        dR_dr, dR_dtheta, dZ_dr, dZ_dtheta = self._compute_dRZ_drt_(tor1_arr, tor2_arr)
        
        # Get R coordinates (shape: Nr, Ntheta, Nphi)
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        
        # Compute Jacobian determinant from R-Z transformation (shape: Nr, Ntheta, Nphi)
        jacobian_det = (dR_dr * dZ_dtheta - dR_dtheta * dZ_dr)[:, :, None] * R

        # Get Psi and dPsi/dr (shape: Nr)
        Psi, dPsidr = self.get_Psi(tor1_arr)
        F = self.F_cb(tor1_arr)
        dPsidr = dPsidr[:, None, None]  # reshape for broadcasting

        # Contravariant magnetic field components (fully vectorized)
        # B^r = 0
        B_contra[:, :, :, 0] = 0.0
        
        # B^theta = dPsi/dr / J (shape: Nr, Ntheta, Nphi)
        B_contra[:, :, :, 1] = self.signma_phi * dPsidr / jacobian_det

        # B^phi = F/R**2 (shape: Nr, Ntheta, Nphi)
        B_contra[:, :, :, 2] = self.signma_phi * F[:, None, None] / R**2
        
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

        FFprime = (self.FFprime_cb(tor1_arr))[:, None, None]
        F = self.F_cb(tor1_arr)[:, None, None]
        pprime = (self.pprime_cb(tor1_arr))[:, None, None]
        
        B_contra = self.get_Bcontra(tor1_arr, tor2_arr, tor3_arr)
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)

        J_contra[..., 1] = -(FFprime / F) * B_contra[..., 1] * self.signma_phi
        
        # J^phi = -(FF'/R^2 + mu0*p')
        J_contra[..., 2] = -(FFprime / R**2 + pprime) * self.signma_phi
        
        return J_contra

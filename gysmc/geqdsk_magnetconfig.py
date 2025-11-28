# SPDX-License-Identifier: MIT

"""
Created on 2025-11-08

 Contains the class to initialise the magnetic configuration from a geqdsk file

@author: Z. S. Qu
"""

import numpy as np
import scipy.integrate
from scipy.interpolate import RectBivariateSpline, CubicSpline
try:
    from scipy.integrate import cumtrapz
except ImportError:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
try:
    from freeqdsk import geqdsk
except ImportError:
    raise ImportError("freeqdsk module is required for GEQDSKMagnetConfig. Please visit https://github.com/freegs-plasma/FreeQDSK")

from .magnet_config import MagnetConfig


class GEQDSKMagnetConfig(MagnetConfig):
    """
    Class to initialise the magnetic configuration from GEQDSK equilibrium data
    
    This class extends MagnetConfig to work with GEQDSK (GEometry Q-grid DiSpatching sKinetic) 
    format magnetic equilibrium data. It provides functionality to:
    
    1. Load magnetic equilibrium data from GEQDSK format
    2. Compute R(psi,theta) and Z(psi,theta) mappings
    3. Interpolate between flux surfaces and poloidal angles
    
    The R(psi,theta) and Z(psi,theta) functionality maps from:
    - psi: normalized poloidal flux (0 at magnetic axis, 1 at plasma boundary)
    - theta: poloidal angle with respect to magnetic axis (0 to 2π)
    
    To Cartesian coordinates (R,Z) in the poloidal plane.
    
    Attributes:
        tor1_arr: normalised minor radius r/a
        R_psi_theta: R coordinates as function of (psi, theta)
        Z_psi_theta: Z coordinates as function of (psi, theta)
        psi_levels: psi grid used for mapping
        theta_grid: theta grid used for mapping
    """

    def __init__(self, filename, cocos=2, thetastar=False, psinorm_max=0.95, rmax=1.0, Ninterp=1024, verbose=True):
        """
        Initialisation of the magnetic configuration
        :param geqdsk_data: GEQDSK data object
        :param cocos: COCOS convection of the input, see [Sauter O, Medvedev SY. CPC 2013]
        :param thetastar: whether to use thetastar coordinate
        :param psinorm_max: maximum value of normalized psi
        :param rmax: maximum value of minor radius
        :param Ninterp: number of interpolation points in s and theta
        :param verbose: whether to print verbose information

        see also set_cocos_convention method for a list of supported cocos conventions
        the output uses cocos = 2 convention (GYSELA/CHEASE standard)
        """
        from scipy.optimize import minimize

        print("Initialising GEQDSKMagnetConfig from file: {}".format(filename))
        print("cocos convention set to: {}".format(cocos))
        print("WARNING: r is related to sqrt(poloidal flux), not the geometrical minor radius!")
        print("WARNING: rmax is set at {:.4f}, linked to psinorm_max at {:.4f}".format(rmax, psinorm_max))

        with open(filename, 'r') as f:
            data = geqdsk.read(f, cocos=cocos)

        self.set_cocos_convention(cocos, verbose=verbose) # set cocos convention

        Nint = Ninterp+1 # number of interpolation points
        self.Nint = Nint

        self.thetastar = thetastar
        if self.thetastar:
            print("INFO: Using thetastar coordinate")
        else:
            print("INFO: Using geometric theta coordinate")

        if data.psi_boundary > data.psi_axis:
            psi_sign = 1.0
        else:
            psi_sign = -1.0

        # optimise the location of magnetic axis
        spline_psi = RectBivariateSpline(data.r_grid[:,0], data.z_grid[0], psi_sign *  data.psi)
        res  = minimize(lambda x: spline_psi(x[0], x[1], grid=False), 
                        np.array([data.rmaxis, data.zmaxis]), 
                        method='BFGS', 
                        options={'gtol': 1e-8, 'disp': False})
        
        r_axis_opt, z_axis_opt = res.x
        psi_axis_opt = res.fun
        
        if verbose:
            print("INFO: Optimized magnetic axis location: R_axis = {:.6f}, Z_axis = {:.6f}, Psi_axis = {:.6f}".format(
                r_axis_opt, z_axis_opt, psi_axis_opt))
            print("INFO: Original magnetic axis location: R_axis = {:.6f}, Z_axis = {:.6f}, Psi_axis = {:.6f}".format(
                data.rmaxis, data.zmaxis, data.psi_axis))

        self.geqdsk_data = data
        self.psi1_real = data.psi_boundary - psi_axis_opt
        self.psinorm_max = psinorm_max
        self.rmax = rmax
        self.smax = np.sqrt(psinorm_max)
        self.s_at_r_one = self.smax/rmax # value of s at r = 1.0

        # normalise psi to between 0 and 1
        psinorm = (data.psi - psi_axis_opt) / (data.psi_boundary - psi_axis_opt)
        psinorm = psinorm.T
        self.psinorm = psinorm
        self.Z_grid_original = data.z_grid
        self.R_grid_original = data.r_grid
        self.Rbdry = data.rbdry
        self.Zbdry = data.zbdry
        self.Rmaxis = r_axis_opt
        self.Zmaxis = z_axis_opt

        self.F = data.f
        self.FFprime = data.ffprime
        self.p = data.pres
        self.pprime = data.pprime

        # create spline for psi(x,y), x = R - Raxis, y = Z - Zaxis
        xdata = data.r_grid[:,0] - self.Rmaxis
        ydata = data.z_grid[0] - self.Zmaxis
        spline_psi = RectBivariateSpline(xdata, ydata, psinorm.T)
        
        # array for computed R(s, theta) and Z(s, theta)
        R_s_theta = np.zeros((Nint, Nint))
        Z_s_theta = np.zeros((Nint, Nint))

        # extract and sort boundary points
        xbdry = data.rbdry - self.Rmaxis
        zbdry = data.zbdry - self.Zmaxis
        if xbdry[0] == xbdry[-1] and zbdry[0] == zbdry[-1]:
            xbdry = xbdry[:-1]
            zbdry = zbdry[:-1]
        boundary_angle = np.mod(np.arctan2(zbdry, xbdry) + 2*np.pi, 2*np.pi)

        # sort boundary points by angle
        sorted_indices = np.argsort(boundary_angle)
        xbdry_sorted = xbdry[sorted_indices]
        ybdry_sorted = zbdry[sorted_indices]
        boundary_angle_sorted = boundary_angle[sorted_indices]

        # interpolate boundary points to get x and y at desired angles
        boundary_angle_sorted = np.append(boundary_angle_sorted, boundary_angle_sorted[0] + 2*np.pi)
        xbdry_sorted = np.append(xbdry_sorted, xbdry_sorted[0])
        ybdry_sorted = np.append(ybdry_sorted, ybdry_sorted[0])
        xbdry_cb = CubicSpline(boundary_angle_sorted, xbdry_sorted, bc_type='periodic')
        ybdry_cb = CubicSpline(boundary_angle_sorted, ybdry_sorted, bc_type='periodic')

        # interpolation grid for s and theta
        s_target = np.linspace(0, self.smax, Nint, True)
        theta_vals = np.linspace(0, 2*np.pi, Nint, endpoint=True)
        self.s_target = s_target
        self.theta_vals = theta_vals

        # R and Z at r=1.0
        R_at_r_one = np.zeros([Nint])
        Z_at_r_one = np.zeros([Nint])

        for i in range(Nint-1):
            theta = theta_vals[i]
            x_bdry = xbdry_cb(theta)
            y_bdry = ybdry_cb(theta)
            dis_bdry = np.sqrt(x_bdry**2 + y_bdry**2)

            dis_array = np.linspace(0, dis_bdry, Nint, True)[1:]  # exclude zero to avoid singularity
            x_array = dis_array * np.cos(theta)
            y_array = dis_array * np.sin(theta)
            s_array = np.concatenate([[0], np.sqrt(spline_psi(x_array, y_array, grid=False))])
            dis_array = np.concatenate([[0], dis_array])

            cb_s_to_dis = CubicSpline(s_array, dis_array)
            dis_at_s = cb_s_to_dis(s_target)
            R_at_s = dis_at_s * np.cos(theta) + self.Rmaxis
            Z_at_s = dis_at_s * np.sin(theta) + self.Zmaxis

            R_s_theta[:, i] = R_at_s
            Z_s_theta[:, i] = Z_at_s

            # compute R and Z at r=1.0
            dis_at_one = cb_s_to_dis(self.s_at_r_one)
            R_at_r_one[i] = dis_at_one * np.cos(theta) + self.Rmaxis
            Z_at_r_one[i] = dis_at_one * np.sin(theta) + self.Zmaxis

        # compute quantities of the boundary at r=1.0
        self.Rleft_real = np.min(R_at_r_one[:-1])
        self.Rright_real = np.max(R_at_r_one[:-1])
        self.Rgeo_real = 0.5 * (self.Rleft_real + self.Rright_real)
        self.ageo_real = 0.5 * (self.Rright_real - self.Rleft_real)
        self.aspect_ratio = self.Rgeo_real / self.ageo_real

        # compute vacuum toroidal field at Rgeo
        self.Bvac0_real = np.abs(data.fpol[-1] / self.Rgeo_real)
        # compute magnetic field at axis
        self.Bphyaxis_real = np.abs(data.fpol[0] / self.Rmaxis)
        # compute psi1 normalised to B0vac*a_geo^2
        self.psi1_norm = self.psi1_real / (self.Bvac0_real * self.ageo_real**2)

        if verbose:
            print('INFO: Computational boundary at r={:.4f}: s_max={:.4f}, psinorm_max={:.4f}'.format(
                rmax, self.smax, psinorm_max))
            print('INFO: Plasma boundary at r=1.0: s(r=1.0)={:.4f}, psinorm(r=1.0)={:.4f}'.format(
                self.s_at_r_one, self.s_at_r_one**2))
            print('INFO: Geometry in original unit: R_geo = {:.4f}, a_geo(at r=1) = {:.4f}, Aspect ratio = {:.4f}'.format(
                self.Rgeo_real, self.ageo_real, self.aspect_ratio))
            print('INFO: Magnetic field in original unit: Bvac_geocentre = {:.4f}, B_axis = {:.4f}'.format(
                self.Bvac0_real, self.Bphyaxis_real))
            print('INFO: Psi1: in original unit psi1 = {:.4f}, normalised to Bvac0*a_geo^2 psi1_norm = {:.4f}'.format(
                self.psi1_real, self.psi1_norm))

        print("INFO: B is normalised to Bvac0={:.4f} at Rgeo={:.4f}, length to a_geo={:.4f}".format(
            self.Bvac0_real, self.Rgeo_real, self.ageo_real))

        # normalise R and Z by a_geo
        R_s_theta /= self.ageo_real
        Z_s_theta /= self.ageo_real

        # handle the last theta = 2pi separately to ensure periodicity
        R_s_theta[:, -1] = R_s_theta[:, 0]
        Z_s_theta[:, -1] = Z_s_theta[:, 0]

        # normalise plasma profiles
        mu0 = 4e-7 * np.pi
        Npsi_grid = data.ffprime.shape[0]
        psi_grid_to_interp = np.linspace(0, 1, Npsi_grid, True)
        r_grid_to_interp = np.sqrt(psi_grid_to_interp) / self.s_at_r_one
        Funit = self.Bvac0_real * self.ageo_real
        psiunit = self.Bvac0_real * self.ageo_real**2
        mu0punit = self.Bvac0_real**2
        Fprofile_norm = data.fpol / Funit
        FFprimeprofile_norm = data.ffprime / (Funit**2/psiunit)
        pprime_norm = mu0 * data.pprime / (mu0punit / psiunit)

        self.F_cb = CubicSpline(r_grid_to_interp, Fprofile_norm, bc_type=((1,0), (2,0)))
        self.FFprime_cb = CubicSpline(r_grid_to_interp, FFprimeprofile_norm, bc_type='natural')
        self.pprime_cb = CubicSpline(r_grid_to_interp, pprime_norm, bc_type='natural')
        self.q_cb = CubicSpline(r_grid_to_interp, data.qpsi, bc_type=((1,0), (2,0)))

        # construct cubic splines for R(r) and Z(r) at each theta, extend r to negative for natural BCs
        self.r_grid = s_target / self.smax * self.rmax
        s_combined = np.concatenate([-s_target[::-1], s_target[1:]])
        r_combined = s_combined / self.smax * self.rmax
        R_combined = np.zeros([2*Nint-1, Nint])
        Z_combined = np.zeros([2*Nint-1, Nint])

        # combine R and Z for negative and positive s for smooth natural BCs
        for i in range(Nint//2):
            Nhalf = Nint // 2
            R_combined[:, i] = np.concatenate([R_s_theta[::-1, i+Nhalf], R_s_theta[1:, i]]) 
            Z_combined[:, i] = np.concatenate([Z_s_theta[::-1, i+Nhalf], Z_s_theta[1:, i]])
            R_combined[:, i+Nhalf] = R_combined[::-1, i]
            Z_combined[:, i+Nhalf] = Z_combined[::-1, i]

        R_combined[:, -1] = R_combined[:, 0]
        Z_combined[:, -1] = Z_combined[:, 0]

        self.cb_R_r = CubicSpline(r_combined, R_combined, axis=0)
        self.cb_Z_r= CubicSpline(r_combined, Z_combined, axis=0)

        if self.thetastar:
            r_grid_modified = self.r_grid
            r_grid_modified[0] = 1e-6  # avoid singularity at r=0

            B_contra = self.get_Bcontra(r_grid_modified, self.theta_vals, np.array([0.0]))
            q = self.get_q(r_grid_modified)

            dthetastar_dtheta =  B_contra[:, :, 0, 2] / B_contra[:, :, 0, 1] / q[:, None]
            thetastar_vals = cumtrapz(dthetastar_dtheta, self.theta_vals, initial=0.0, axis=1)

            # normalise thetastar to [0, 2pi]
            thetastar_vals = thetastar_vals / thetastar_vals[:, -1][:, None] * 2.0 * np.pi

            R_s_thetastar = np.zeros_like(R_s_theta)
            Z_s_thetastar = np.zeros_like(Z_s_theta)

            # reconstruct R and Z splines at thetastar values
            for i in range(Nint):
                cb_R_thetastar = CubicSpline(thetastar_vals[i], R_s_theta[i], axis=1, bc_type='periodic')
                cb_Z_thetastar = CubicSpline(thetastar_vals[i], Z_s_theta[i], axis=1, bc_type='periodic')

                # obtain R and Z at uniform thetastar grid
                R_s_thetastar[i] = cb_R_thetastar(self.theta_vals)
                Z_s_thetastar[i] = cb_Z_thetastar(self.theta_vals)

            # reconstruct R(r) and Z(r) splines at each thetastar
            R_combined = np.zeros([2*Nint-1, Nint])
            Z_combined = np.zeros([2*Nint-1, Nint])

            # combine R and Z for negative and positive s for smooth natural BCs
            for i in range(Nint//2):
                Nhalf = Nint // 2
                R_combined[:, i] = np.concatenate([R_s_thetastar[::-1, i+Nhalf], R_s_thetastar[1:, i]]) 
                Z_combined[:, i] = np.concatenate([Z_s_thetastar[::-1, i+Nhalf], Z_s_thetastar[1:, i]])
                R_combined[:, i+Nhalf] = R_combined[::-1, i]
                Z_combined[:, i+Nhalf] = Z_combined[::-1, i]

            R_combined[:, -1] = R_combined[:, 0]
            Z_combined[:, -1] = Z_combined[:, 0]

            self.cb_R_r = CubicSpline(r_combined, R_combined, axis=0)
            self.cb_Z_r= CubicSpline(r_combined, Z_combined, axis=0)

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

    def plot_flux_surfaces(self, num_surfaces=10, normalised_units=True):
        """
        Plot flux surfaces using the R(psi,theta) and Z(psi,theta) mappings
        :param num_surfaces: number of flux surfaces to plot
        """
        import matplotlib.pyplot as plt

        # plot psi
        if normalised_units:
            plt.pcolormesh(self.R_grid_original/self.ageo_real, self.Z_grid_original/self.ageo_real, self.psinorm.T)
            plt.colorbar(label='Normalized Psi')
            plt.plot(self.Rbdry/self.ageo_real, self.Zbdry/self.ageo_real, color='k')
            plt.scatter(self.Rmaxis/self.ageo_real, self.Zmaxis/self.ageo_real, marker='x', color='r')
        else:
            plt.pcolormesh(self.R_grid_original, self.Z_grid_original, self.psinorm.T*self.psi1_real)
            plt.colorbar(label='Psi')
            plt.plot(self.Rbdry, self.Zbdry, color='k')
            plt.scatter(self.Rmaxis, self.Zmaxis, marker='x', color='r')

        r_values = np.linspace(0, self.rmax, num_surfaces+1, True)[1:]  # exclude r=0

        R_vals = self.cb_R_r(r_values)
        Z_vals = self.cb_Z_r(r_values)
        R_vals_r_one = self.cb_R_r(1.0)
        Z_vals_r_one = self.cb_Z_r(1.0)

        if normalised_units:
            for i in range(num_surfaces-1):
                plt.plot(R_vals[i], Z_vals[i], 'w-')
            plt.plot(R_vals_r_one, Z_vals_r_one, 'r--', label='r=1.0')
            plt.plot(R_vals[-1], Z_vals[-1], 'r-', label='r=max')
            plt.xlabel('R/a')
            plt.ylabel('Z/a')
        else:
            for i in range(num_surfaces):
                plt.plot(R_vals[i]*self.ageo_real, Z_vals[i]*self.ageo_real, 'w-')
            plt.plot(R_vals_r_one*self.ageo_real, Z_vals_r_one*self.ageo_real, 'r--', label='r=1.0')
            plt.plot(R_vals[-1]*self.ageo_real, Z_vals[-1]*self.ageo_real, 'r-', label='r=max')
            plt.xlabel('R')
            plt.ylabel('Z')
        plt.title('Flux Surfaces from GEQDSK Data')
        plt.axis('equal')
        plt.legend()

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
        # assign Z=0 at magnetic axis
        Z_coord = cb_Z(theta)[:,:,None] + tor3_arr[None, None, :] * 0 - self.Zmaxis / self.ageo_real

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
        theta = 2 * np.arctan(np.tan(tor2_arr[None, :] / 2) * np.sqrt((R0 + tor1_arr[:, None]) / (R0 - tor1_arr[:, None])))
        return theta
# SPDX-License-Identifier: MIT

"""
Created on 2025-11-05

 Contains the GVEC magnetic equilibrium configuration class

@author: Z. S. Qu
"""

import numpy as np
import tempfile
from pathlib import Path
from .magnet_config import MagnetConfig

# Try to import GVEC
try:
    import gvec
    GVEC_AVAILABLE = True
except ImportError:
    GVEC_AVAILABLE = False


class GvecMagnetConfig(MagnetConfig):
    """
    GVEC magnetic equilibrium configuration class
    
    Implements the GVEC equilibrium model for tokamak magnetic configuration
    using the GVEC (Galerkin Variational Equilibrium Code) library.
    
    This class uses GVEC to compute 3D MHD equilibria and provides the same
    interface as CulhamMagnetConfig for compatibility.
    """

    def __init__(self, major_radius=3, q_profile=None, pressure_profile=None, 
                 rmax=1.2, beta_toro=0.0, kappa_elongation=1.0, 
                 delta_triangularity=0.0, r_array=None, runpath=None):
        """
        Initialize the GVEC equilibrium
        
        Parameters:
        -----------
        major_radius : float
            Major radius at magnetic axis
        q_profile : QProfile
            Safety factor profile object with get_q(r) method
        pressure_profile : array_like or callable
            Pressure profile p(r) - can be array or object with get_pressure(r) method
        rmax : float, optional
            Maximum radial coordinate (default: 1.2)
        beta_toro : float, optional
            Toroidal beta parameter (default: 0.0)
        kappa_elongation : float, optional
            Elongation at edge (default: 1.0)
        delta_triangularity : float, optional
            Triangularity at edge (default: 0.0)
        r_array : array_like, optional
            The radial coordinate array. If None, a default grid will be created.
        runpath : str or Path, optional
            Path for GVEC run directory. If None, uses temporary directory.
        """
        super().__init__()
        
        if not GVEC_AVAILABLE:
            raise ImportError("GVEC is not available. Please install gvec package.")
        
        if q_profile is None:
            from .q_profile import QProfile
            q_profile = QProfile()
        
        if r_array is None:
            # number of internal radial points
            self.nr = 1024  # Default number of radial points
            rmin = 0.00001
            self.r_array = np.linspace(rmin, rmax, self.nr, True)
        else:
            self.r_array = np.array(r_array)
            rmax = self.r_array[-1]
        
        self.R0 = major_radius
        self.q_profile_obj = q_profile  # Store QProfile object
        self.q_profile, self.dqdr_profile = q_profile.get_q(self.r_array)  # Get q values and derivatives
        self.pressure_profile_obj = pressure_profile
        self.minor_radius = 1.0
        self.beta_toro = beta_toro
        self.kappa_elongation = kappa_elongation
        self.delta_triangularity = delta_triangularity
        
        # Get pressure profile
        if self.pressure_profile_obj is None:
            p_profile = np.zeros(len(self.r_array))
        else:
            if hasattr(self.pressure_profile_obj, 'get_pressure'):
                p_profile, _ = self.pressure_profile_obj.get_pressure(self.r_array)
            elif callable(self.pressure_profile_obj):
                p_profile = self.pressure_profile_obj(self.r_array)
            else:
                p_profile = np.asarray(self.pressure_profile_obj)
                if len(p_profile) != len(self.r_array):
                    # Interpolate if needed
                    from scipy.interpolate import interp1d
                    p_interp = interp1d(np.linspace(0, rmax, len(p_profile)), p_profile, 
                                       bounds_error=False, fill_value=0.0)
                    p_profile = p_interp(self.r_array)
            
            p_profile *= self.beta_toro
        
        # Create GVEC parameters
        params = self._create_gvec_parameters(p_profile)
        
        # Run GVEC
        self.runpath = runpath
        self._run_gvec(params)
        
        # Pre-compute values on r_array grid for interpolation
        self._precompute_grid_values()
        
        # Construct splines for efficient interpolation
        self._construct_splines()

    def _create_gvec_parameters(self, p_profile):
        """
        Create GVEC parameters dictionary from the given parameters.
        
        Parameters:
        -----------
        p_profile : array_like
            Pressure profile values on r_array
            
        Returns:
        --------
        dict
            GVEC parameters dictionary
        """
        params = {}
        # Project name
        params["ProjectName"] = "GyselaX"
        # Physics parameters
        params["PhiEdge"] = np.pi  # toroidal flux at s=1
        # Rotational transform profile (iota = 1/q)
        iota_profile = 1.0 / self.q_profile
        # Avoid division by zero
        iota_profile = np.where(np.isfinite(iota_profile), iota_profile, 0.0)
        params["iota"] = {
            "type": "interpolation",
            "rho2": self.r_array**2,
            "vals": iota_profile,
        }
        # Pressure profile
        params["pres"] = {
            "type": "interpolation",
            "rho2": self.r_array**2,
            "vals": p_profile,
        }
        
        # Boundary shape parameters
        params["which_hmap"] = 1  # cylindrical coordinates: X1=R, X2=Z
        params["nfp"] = 1  # number of field periods
        
        # Culham parameters for the boundary shape
        a = self.minor_radius
        eps = a / self.R0  # aspect ratio
        Ea = a * (self.kappa_elongation - 1) / (self.kappa_elongation + 1)
        Ta = a * self.delta_triangularity / 4
        Pa = a * ((eps**2) / 8 - 0.5 * (Ea / a) ** 2 - (Ta / a) ** 2)
        
        # Fourier coefficients for boundary shape
        # R_edge(theta) = a_0 + a_1*cos(theta) + a_2*cos(2*theta)
        a_0 = self.R0
        a_1 = a - Ea - Pa
        a_2 = Ta
        params["X1_b_cos"] = {(0, 0): a_0, (1, 0): a_1, (2, 0): a_2}
        
        # Z_edge(theta) = b_1*sin(theta) + b_2*sin(2*theta)
        b_1 = a + Ea - Pa
        b_2 = -Ta
        params["X2_b_sin"] = {(1, 0): b_1, (2, 0): b_2}
        
        # Initial guess for magnetic axis
        params["X1_a_cos"] = {(0, 0): a_0, (1, 0): a_1, (2, 0): a_2}
        params["X2_a_sin"] = {(1, 0): b_1, (2, 0): b_2}
        
        # Numerical parameters
        params["X1_mn_max"] = [2, 0]  # maximum Fourier modes for X1
        params["X2_mn_max"] = [2, 0]  # maximum Fourier modes for X2
        params["LA_mn_max"] = [2, 0]  # maximum Fourier modes for LA
        
        params["sgrid_nElems"] = 2  # number of radial B-spline elements
        params["X1X2_deg"] = 5  # degree of B-splines for X1 and X2
        params["LA_deg"] = 5  # degree of B-splines for LA
        
        # Minimiser parameters
        params["totalIter"] = 10000  # maximum number of iterations
        params["minimize_tol"] = 1e-6  # stopping tolerance
        
        return params

    def _run_gvec(self, params):
        """
        Run GVEC to compute equilibrium.
        
        Parameters:
        -----------
        params : dict
            GVEC parameters dictionary
        """
        # Create temporary directory for GVEC run if not provided
        if self.runpath is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            gvec_run_dir = Path(self._temp_dir.name) / "gvec_run"
        else:
            self._temp_dir = None
            gvec_run_dir = Path(self.runpath)
            gvec_run_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            print("Running GVEC with parameters dictionary...")
            self.gvec_run = gvec.run(params, runpath=str(gvec_run_dir))
            print("GVEC run successful")
        except Exception as e:
            print(f"GVEC run failed: {e}")
            raise
        
        # Get state from run object
        self.gvec_state = self.gvec_run.state

    def _precompute_grid_values(self):
        """
        Pre-compute values on the r_array grid for efficient interpolation.
        """
        # Create evaluation points
        rho = self.r_array.copy()
        if rho[0] == 0.0:
            rho[0] = 1e-15
        
        # Use a single theta and zeta value for radial profiles
        theta = np.array([0.0])
        zeta = np.array([0.0])
        
        # Evaluate at grid points
        ev = self.gvec_state.evaluate(
            "X1",
            "X2",
            "dPhi_dr",
            "iota",
            "I_tor",
            "B_contra_t",
            "B_contra_z",
            "g_rr",
            "g_rt",
            "g_rz",
            "g_tt",
            "g_tz",
            "g_zz",
            "mod_B",
            rho=rho,
            theta=theta,
            zeta=zeta,
        )
        
        # Store radial profiles (extract first theta/zeta slice)
        self.dPsidr_r = ev.dPhi_dr.values[:, 0, 0]
        self.iota_r = ev.iota.values[:, 0, 0]
        self.current_r = ev.I_tor.values[:, 0, 0]
        self.B0 = ev.mod_B.values[0, 0, 0]  # Reference B field
        
        # Store metric tensor components
        self.g_rr_r = ev.g_rr.values[:, 0, 0]
        self.g_rt_r = ev.g_rt.values[:, 0, 0]
        self.g_rz_r = ev.g_rz.values[:, 0, 0]
        self.g_tt_r = ev.g_tt.values[:, 0, 0]
        self.g_tz_r = ev.g_tz.values[:, 0, 0]
        self.g_zz_r = ev.g_zz.values[:, 0, 0]
        
        # Compute Psi by integrating dPsi/dr
        from scipy.integrate import cumulative_trapezoid
        self.Psi_r = cumulative_trapezoid(self.dPsidr_r, self.r_array, initial=0.0)

    def _construct_splines(self):
        """
        Pre-construct cubic splines for all radial profiles for efficient interpolation.
        """
        from scipy.interpolate import CubicSpline
        
        # Create splines for all profiles
        self.spline_Psi = CubicSpline(self.r_array, self.Psi_r, bc_type='natural')
        self.spline_dPsidr = CubicSpline(self.r_array, self.dPsidr_r, bc_type='natural')
        self.spline_iota = CubicSpline(self.r_array, self.iota_r, bc_type='natural')
        self.spline_current = CubicSpline(self.r_array, self.current_r, bc_type='natural')
        self.spline_g_rr = CubicSpline(self.r_array, self.g_rr_r, bc_type='natural')
        self.spline_g_rt = CubicSpline(self.r_array, self.g_rt_r, bc_type='natural')
        self.spline_g_rz = CubicSpline(self.r_array, self.g_rz_r, bc_type='natural')
        self.spline_g_tt = CubicSpline(self.r_array, self.g_tt_r, bc_type='natural')
        self.spline_g_tz = CubicSpline(self.r_array, self.g_tz_r, bc_type='natural')
        self.spline_g_zz = CubicSpline(self.r_array, self.g_zz_r, bc_type='natural')

    def _evaluate_gvec(self, rho, theta, zeta, quantities):
        """
        Evaluate GVEC quantities at given coordinates.
        
        Parameters:
        -----------
        rho : array_like
            Radial coordinates
        theta : array_like
            Poloidal coordinates
        zeta : array_like
            Toroidal coordinates
        quantities : list of str
            Quantities to evaluate
            
        Returns:
        --------
        xarray.Dataset
            Evaluated quantities
        """
        # Ensure rho doesn't have zeros
        rho = np.asarray(rho).copy()
        rho[rho == 0.0] = 1e-15
        
        ev = self.gvec_state.evaluate(*quantities, rho=rho, theta=theta, zeta=zeta)
        return ev

    def get_RZ(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the R and Z coordinates from toroidal coordinates.
        Fully vectorized implementation.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r) - 1D array of shape (Nr,)
        tor2_arr : array_like
            Poloidal coordinates (theta) - 1D array of shape (Ntheta,)
        tor3_arr : array_like
            Toroidal coordinates (phi) - scalar or 1D array of shape (Nphi,)
            
        Returns:
        --------
        tuple : (R_coord, Z_coord) coordinates of shape (Nr, Ntheta, Nphi)
        """
        # Ensure inputs are arrays
        tor1_arr = np.asarray(tor1_arr)
        tor2_arr = np.asarray(tor2_arr)
        
        # Validate that r and theta are 1D
        if tor1_arr.ndim != 1 or tor2_arr.ndim != 1:
            raise ValueError("tor1_arr (r) and tor2_arr (theta) must be 1D arrays")
        
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        
        # Prepare coordinates for GVEC evaluation
        rho = tor1_arr.copy()
        if rho[0] == 0.0:
            rho[0] = 1e-15
        
        theta = tor2_arr
        zeta = np.array([0.0]) if nb_grid_tor3 == 1 else np.asarray(tor3_arr)
        
        # Evaluate X1 (R) and X2 (Z) from GVEC
        ev = self._evaluate_gvec(rho, theta, zeta, ["X1", "X2"])
        
        R_coord = ev.X1.values
        Z_coord = ev.X2.values
        
        # Reshape if needed
        if R_coord.ndim == 2 and nb_grid_tor3 == 1:
            R_coord = R_coord[:, :, np.newaxis]
            Z_coord = Z_coord[:, :, np.newaxis]
        elif R_coord.ndim == 2:
            # Broadcast zeta dimension
            R_coord = np.repeat(R_coord[:, :, np.newaxis], nb_grid_tor3, axis=2)
            Z_coord = np.repeat(Z_coord[:, :, np.newaxis], nb_grid_tor3, axis=2)
        
        return R_coord, Z_coord

    def get_gij(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the gij coefficients from toroidal coordinates.
        Fully vectorized implementation.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r) - 1D array of shape (Nr,)
        tor2_arr : array_like
            Poloidal coordinates (theta) - 1D array of shape (Ntheta,)
        tor3_arr : array_like
            Toroidal coordinates (phi) - scalar or 1D array of shape (Nphi,)
            
        Returns:
        --------
        tuple : (ContravariantMetricTensor, CovariantMetricTensor)
                Both of shape (Nr, Ntheta, Nphi, 3, 3)
        """
        # Ensure inputs are arrays
        tor1_arr = np.asarray(tor1_arr)
        tor2_arr = np.asarray(tor2_arr)
        
        # Validate that r and theta are 1D
        if tor1_arr.ndim != 1 or tor2_arr.ndim != 1:
            raise ValueError("tor1_arr (r) and tor2_arr (theta) must be 1D arrays")
        
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_metric = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3, 3)
        
        CovariantMetricTensor = np.zeros(shape_metric, dtype=float)
        ContravariantMetricTensor = np.zeros(shape_metric, dtype=float)
        
        # Prepare coordinates for GVEC evaluation
        rho = tor1_arr.copy()
        if rho[0] == 0.0:
            rho[0] = 1e-15
        
        theta = tor2_arr
        zeta = np.array([0.0]) if nb_grid_tor3 == 1 else np.asarray(tor3_arr)
        
        # Evaluate metric tensor components from GVEC
        ev = self._evaluate_gvec(rho, theta, zeta, 
                               ["g_rr", "g_rt", "g_rz", "g_tt", "g_tz", "g_zz"])
        
        g_rr = ev.g_rr.values
        g_rt = ev.g_rt.values
        g_rz = ev.g_rz.values
        g_tt = ev.g_tt.values
        g_tz = ev.g_tz.values
        g_zz = ev.g_zz.values
        
        # Get R coordinates for g33 component
        R_coord, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        
        # Fill covariant metric tensor
        # GVEC uses (rho, theta, zeta) coordinates which map to (r, theta, phi)
        CovariantMetricTensor[:, :, :, 0, 0] = g_rr  # g_rr
        CovariantMetricTensor[:, :, :, 0, 1] = g_rt  # g_rt
        CovariantMetricTensor[:, :, :, 1, 0] = g_rt  # g_tr (symmetric)
        CovariantMetricTensor[:, :, :, 0, 2] = g_rz  # g_rz
        CovariantMetricTensor[:, :, :, 2, 0] = g_rz  # g_zr (symmetric)
        CovariantMetricTensor[:, :, :, 1, 1] = g_tt  # g_tt
        CovariantMetricTensor[:, :, :, 1, 2] = g_tz  # g_tz
        CovariantMetricTensor[:, :, :, 2, 1] = g_tz  # g_zt (symmetric)
        CovariantMetricTensor[:, :, :, 2, 2] = g_zz  # g_zz
        
        # Compute contravariant metric tensor (inverse of covariant)
        # Vectorized inversion
        for i in range(nb_grid_tor1):
            for j in range(nb_grid_tor2):
                for k in range(nb_grid_tor3):
                    g_cov = CovariantMetricTensor[i, j, k, :, :]
                    try:
                        g_contra = np.linalg.inv(g_cov)
                    except np.linalg.LinAlgError:
                        # Fallback for singular matrices
                        g_contra = np.eye(3)
                    ContravariantMetricTensor[i, j, k, :, :] = g_contra
        
        return ContravariantMetricTensor, CovariantMetricTensor

    def get_Psi(self, tor1_arr):
        """
        Get Psi, dPsidr using precomputed spline interpolation
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r)
            
        Returns:
        --------
        tuple : (Psi, dPsidr)
        """
        tor1_arr = np.asarray(tor1_arr)
        
        # Use precomputed splines for fast and accurate interpolation
        Psi = self.spline_Psi(tor1_arr)
        dPsidr = self.spline_dPsidr(tor1_arr)
        
        return Psi, dPsidr

    def get_q(self, tor1_arr):
        """
        Get safety factor q
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r)
            
        Returns:
        --------
        array : q profile
        """
        # Convert from iota: q = 1/iota
        iota_values = self.spline_iota(tor1_arr)
        q_values = np.where(np.abs(iota_values) > 1e-12, 1.0 / iota_values, np.inf)
        return q_values

    def get_Bcontra(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the magnetic field from toroidal coordinates.
        Fully vectorized implementation.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r) - 1D array of shape (Nr,)
        tor2_arr : array_like
            Poloidal coordinates (theta) - 1D array of shape (Ntheta,)
        tor3_arr : array_like
            Toroidal coordinates (phi) - scalar or 1D array of shape (Nphi,)
            
        Returns:
        --------
        array : B_contra magnetic field components of shape (Nr, Ntheta, Nphi, 3)
        """
        # Ensure inputs are arrays
        tor1_arr = np.asarray(tor1_arr)
        tor2_arr = np.asarray(tor2_arr)
        
        # Validate that r and theta are 1D
        if tor1_arr.ndim != 1 or tor2_arr.ndim != 1:
            raise ValueError("tor1_arr (r) and tor2_arr (theta) must be 1D arrays")
        
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_B = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3)
        
        B_contra = np.zeros(shape_B, dtype=float)
        
        # Prepare coordinates for GVEC evaluation
        rho = tor1_arr.copy()
        if rho[0] == 0.0:
            rho[0] = 1e-15
        
        theta = tor2_arr
        zeta = np.array([0.0]) if nb_grid_tor3 == 1 else np.asarray(tor3_arr)
        
        # Evaluate B field components from GVEC
        ev = self._evaluate_gvec(rho, theta, zeta, ["B_contra_t", "B_contra_z", "mod_B"])
        
        B_contra_t = ev.B_contra_t.values
        B_contra_z = ev.B_contra_z.values
        B_norm = ev.mod_B.values
        
        # Normalize by B0
        B_contra_t = B_contra_t / self.B0
        B_contra_z = B_contra_z / self.B0
        
        # Fill B_contra array
        # Radial component vanishes by construction for flux coordinates
        B_contra[:, :, :, 0] = 0.0
        B_contra[:, :, :, 1] = B_contra_t  # Poloidal component
        B_contra[:, :, :, 2] = B_contra_z  # Toroidal component
        
        return B_contra

    def get_Jcontra(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the current density from toroidal coordinates.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r)
        tor2_arr : array_like
            Poloidal coordinates (theta)
        tor3_arr : array_like
            Toroidal coordinates (phi)
            
        Returns:
        --------
        array : J_contra current density components
        """
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_J = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3, 3)
        
        J_contra = np.zeros(shape_J, dtype=float)
        
        # Get magnetic field
        B_contra = self.get_Bcontra(tor1_arr, tor2_arr, tor3_arr)
        
        # Get current profile (I_tor) from GVEC
        rho = tor1_arr.copy()
        if rho[0] == 0.0:
            rho[0] = 1e-15
        
        # Evaluate current at grid points
        theta = tor2_arr
        zeta = np.array([0.0]) if nb_grid_tor3 == 1 else np.asarray(tor3_arr)
        ev = self._evaluate_gvec(rho, theta, zeta, ["I_tor", "dPhi_dr"])
        
        I_tor = ev.I_tor.values
        dPhi_dr = ev.dPhi_dr.values
        
        # Compute current density components
        # J^r = 0 (radial component vanishes)
        J_contra[:, :, :, 0] = 0.0
        
        # J^theta and J^phi require derivatives of I_tor
        # For simplicity, use finite differences on the radial grid
        from scipy.interpolate import CubicSpline
        
        # Interpolate I_tor and dPhi_dr for derivatives
        I_tor_spline = CubicSpline(self.r_array, self.current_r, bc_type='natural')
        dPhi_dr_spline = CubicSpline(self.r_array, self.dPsidr_r, bc_type='natural')
        
        for i, r_pos in enumerate(tor1_arr):
            dIdr = I_tor_spline(r_pos, nu=1)  # dI/dr
            dpsidr = dPhi_dr_spline(r_pos)  # dPsi/dr
            
            if abs(dpsidr) < 1e-12:
                continue
            
            for j in range(nb_grid_tor2):
                for k in range(nb_grid_tor3):
                    # J^theta = -dI/dpsi * B^theta
                    J_contra[i, j, k, 1] = -(dIdr / dpsidr) * B_contra[i, j, k, 1]
                    
                    # J^phi = -dI/dr * B^phi / (dpsi/dr)
                    # Note: This is a simplified version
                    # Full expression would include pressure gradient term
                    J_contra[i, j, k, 2] = -(dIdr * B_contra[i, j, k, 2]) / dpsidr
        
        return J_contra

    def __del__(self):
        """
        Cleanup temporary directory if it was created.
        """
        if hasattr(self, '_temp_dir') and self._temp_dir is not None:
            self._temp_dir.cleanup()


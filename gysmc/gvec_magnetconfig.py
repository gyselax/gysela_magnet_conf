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
import warnings
import matplotlib.pyplot as plt
import xarray as xr

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
                 beta_toro=0.0, kappa_elongation=1.0,
                 delta_triangularity=0.0, runpath=None, r_array=None, max_iter=10000, minimize_tol=1e-6, sgrid_nElems=2, X1X2_deg=5, LA_deg=5, rho_min=1e-13):
        """
        Initialize the GVEC equilibrium
        
        Parameters:
        -----------
        major_radius : float
            Major radius at magnetic axis
        q_profile : QProfile
            Safety factor profile object with get_q(r) method
            q_profile.grid_r is used as the radial coordinate array if r_array is None
        pressure_profile : array_like or callable
            Pressure profile p(r) - can be array or object with get_pressure(r) method
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
        rho_min : float, optional
            Minimum value for rho coordinate to avoid numerical issues at rho=0 (default: 1e-13)
        """
        super().__init__()
        
        if not GVEC_AVAILABLE:
            raise ImportError("GVEC is not available. Please install gvec package.")
        
        if q_profile is None:
            from .q_profile import QProfile
            q_profile = QProfile()

        # Check if the q profile is valid (must be between 0 and 1 for gvec)
        assert min(q_profile.grid_r) >= 0, "It's over, Anakin. min(q_profile.grid_r) must be non-negative"
        assert max(q_profile.grid_r) <= 1, "It's over, Anakin. max(q_profile.grid_r) must be less than or equal to 1"
        
        # note that q_profile.grid_r must not be the same as r_array
        if r_array is None:
            # number of internal radial points
            r_array = q_profile.grid_r
        
        assert min(r_array) >= 0, "It's over, Anakin. min(r_array) must be non-negative"
        assert max(r_array) <= 1, "It's over, Anakin. max(r_array) must be less than or equal to 1"
        # note that gvec solves the equilibrium on these r_array points, however the equilibrium can be evaluated later on any 
        # radial coordinate grid.
        # Therefore it makes sense to limit the numbers of grid points in r_array that are used to solve the equilibrium. 
        assert len(r_array) < 200, "It's over, Anakin. if len(r_array) is to large our gvec starship will explode"
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
        
        # Minimiser parameters for GVEC
        self.max_iter = max_iter
        self.minimize_tol = minimize_tol
        self.sgrid_nElems = sgrid_nElems
        self.X1X2_deg = X1X2_deg
        self.LA_deg = LA_deg
        
        # Numerical parameter for avoiding rho=0
        self.rho_min = rho_min

        # Get pressure profile
        if self.pressure_profile_obj is None:
            p_profile = np.zeros(len(self.r_array))
        else:
            p_profile, _ = self.pressure_profile_obj.get_pressure(self.r_array)
        
        # Create GVEC parameters
        params = self._create_gvec_parameters(p_profile)
        
        # Store params as class attribute
        self.params = params
        
        # Run GVEC
        self.runpath = runpath
        self._run_gvec(params)
        
        # Pre-compute values on r_array grid for interpolation
        self._precompute_grid_values()
        
        # Construct splines for efficient interpolation
        self._construct_splines()

    @classmethod
    def from_params(cls, params, runpath=None, stdout_path=None):
        """
        Initialize GVEC equilibrium from a parameters dictionary or .ini file.
        
        Parameters:
        -----------
        params : dict or str or Path
            GVEC parameters dictionary or path to a .ini parameter file
        runpath : str or Path, optional
            Path for GVEC run directory. If None, uses temporary directory.
        stdout_path : str or Path, optional
            Path for GVEC stdout output file. Only used when params is a .ini file.
            
        Returns:
        --------
        GvecMagnetConfig
            Initialized GVEC equilibrium configuration
        """
        if not GVEC_AVAILABLE:
            raise ImportError("GVEC is not available. Please install gvec package.")
        
        # Create instance
        instance = cls.__new__(cls)
        super(GvecMagnetConfig, instance).__init__()
        instance.runpath = runpath
        instance.rho_min = 1e-13  # Default value for rho_min
        
        # Handle .ini file
        if isinstance(params, (str, Path)) and Path(params).suffix.lower() == '.ini':
            params_path = Path(params)
            if not params_path.exists():
                raise FileNotFoundError(f"Parameter file not found: {params_path}")
            
            # Prepare runpath
            if runpath is None:
                instance._temp_dir = tempfile.TemporaryDirectory()
                gvec_run_dir = Path(instance._temp_dir.name) / "gvec_run"
            else:
                instance._temp_dir = None
                gvec_run_dir = Path(runpath)
                gvec_run_dir.mkdir(parents=True, exist_ok=True)
            
            # Run GVEC from .ini file
            print(f"Running GVEC from parameter file: {params_path}")
            kwargs = {"runpath": str(gvec_run_dir)}
            if stdout_path is not None:
                kwargs["stdout_path"] = str(stdout_path)
            instance.gvec_run = gvec.run(str(params_path), **kwargs)
            instance.gvec_state = instance.gvec_run.state
            print("GVEC run successful")
        else:
            # params is a dictionary
            if not isinstance(params, dict):
                raise TypeError(f"params must be a dictionary or path to .ini file, got {type(params)}")
            # Store params as class attribute
            instance.params = params
            instance._run_gvec(params)
        
        # Pre-compute and construct splines
        instance._precompute_grid_values()
        instance._construct_splines()
        
        return instance

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
        
        params["sgrid_nElems"] = self.sgrid_nElems  # number of radial B-spline elements
        params["X1X2_deg"] = self.X1X2_deg  # degree of B-splines for X1 and X2
        params["LA_deg"] = self.LA_deg  # degree of B-splines for LA
        
        # Minimiser parameters
        params["totalIter"] = self.max_iter  # maximum number of iterations
        params["minimize_tol"] = self.minimize_tol  # stopping tolerance
        
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
        # Extract r_array from state if not already set
        if not hasattr(self, 'r_array') or self.r_array is None:
            try:
                ev_test = self.gvec_state.evaluate("iota", rho="int", theta=[0.0], zeta=[0.0])
                self.r_array = np.array(ev_test.rho.values)
            except:
                # Fallback to default grid
                self.r_array = np.linspace(0.01, 1.0, 64)
        
        # Create evaluation points
        rho = self.r_array.copy()
        if rho[0] == 0.0:
            rho[0] = self.rho_min
        
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
        self.dPsidr_r = ev.dPhi_dr.values[:]
        self.current_r = ev.I_tor.values[:]
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
        rho[rho == 0.0] = self.rho_min
        
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
            rho[0] = self.rho_min
        
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
            rho[0] = self.rho_min
        
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
        ev = self._evaluate_gvec(np.asarray(tor1_arr), np.array([0.0]), np.array([0.0]), ["iota"])
        iota_values = ev.iota.values[:]
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
            rho[0] = self.rho_min
        
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

    def to_gyselaX(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Convert the magnetic configuration to a GyselaX dataset.
        
        This function creates ds_gvec_geometry and ds_magnetconf datasets
        compatible with the GyselaX initialization format.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r) - 1D array
        tor2_arr : array_like
            Poloidal coordinates (theta) - 1D array
        tor3_arr : array_like or None
            Toroidal coordinates (phi) - 1D array or None for 2D case
            
        Returns:
        --------
        tuple : (ds_gvec_geometry, ds_magnetconf)
            Two xarray.Dataset objects containing geometry and magnetic field data
        """
        # Ensure inputs are arrays
        tor1_arr = np.asarray(tor1_arr)
        tor2_arr = np.asarray(tor2_arr)
        
        # Validate that r and theta are 1D
        if tor1_arr.ndim != 1 or tor2_arr.ndim != 1:
            raise ValueError("tor1_arr (r) and tor2_arr (theta) must be 1D arrays")
        
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        # Handle None case for tor3_arr (convert to array for get_RZ)
        if tor3_arr is None:
            tor3_arr_for_eval = np.array([0.0])
            nb_grid_tor3 = 1
        else:
            tor3_arr_for_eval = tor3_arr
            nb_grid_tor3 = np.size(tor3_arr)
        
        # Get R and Z coordinates
        R_coord, Z_coord = self.get_RZ(tor1_arr, tor2_arr, tor3_arr_for_eval)
        
        # Get metric tensors (returns shape: (Nr, Ntheta, Nphi, 3, 3))
        ContravariantMetricTensor_gij, CovariantMetricTensor_gij = self.get_gij(tor1_arr, tor2_arr, tor3_arr_for_eval)
        
        # Transpose to match init_gvec_geometry format: (3, 3, Nr, Ntheta, Nphi)
        CovariantMetricTensor = np.transpose(CovariantMetricTensor_gij, (3, 4, 0, 1, 2))
        ContravariantMetricTensor = np.transpose(ContravariantMetricTensor_gij, (3, 4, 0, 1, 2))
        
        # Get B field (returns shape: (Nr, Ntheta, Nphi, 3))
        B_contra_gij = self.get_Bcontra(tor1_arr, tor2_arr, tor3_arr_for_eval)
        
        # Transpose to match init_gvec_geometry format: (3, Nr, Ntheta, Nphi)
        B_contra = np.transpose(B_contra_gij, (3, 0, 1, 2))
        
        # Get B_norm by evaluating mod_B at grid points
        rho = tor1_arr.copy()
        if rho[0] == 0.0:
            rho[0] = self.rho_min
        
        theta = tor2_arr
        zeta = np.array([0.0]) if nb_grid_tor3 == 1 else np.asarray(tor3_arr_for_eval)
        
        ev = self._evaluate_gvec(rho, theta, zeta, ["mod_B", "dPhi_dr", "I_tor"])
        
        B_norm = ev.mod_B.values
        # Extract B0 from first point (r=0 or r_min, theta=0, zeta=0)
        B0 = B_norm[0, 0, 0]
        B_norm = B_norm / B0
        
        # Extract radial profiles (dPhi_dr and I_tor are flux-surface quantities)
        # Take first slice along theta and phi (they should be constant along these)
        dPhi_dr = ev.dPhi_dr.values
        current_tor1 = ev.I_tor.values
        
        # Extract 1D radial profiles (take first slice along theta and phi)
        if dPhi_dr.ndim > 1:
            dPhi_dr = dPhi_dr[:, 0, 0]
        if current_tor1.ndim > 1:
            current_tor1 = current_tor1[:, 0, 0]
        
        # Handle squeezing when nb_grid_tor3 == 1 (similar to init_gvec_geometry)
        if nb_grid_tor3 == 1:
            R_coord = R_coord.squeeze()
            Z_coord = Z_coord.squeeze()
            B_contra = B_contra.squeeze()
            B_norm = B_norm.squeeze()
            ContravariantMetricTensor = ContravariantMetricTensor.squeeze()
            CovariantMetricTensor = CovariantMetricTensor.squeeze()
        
        # Determine toroidal coordinates
        tor_coord = ["tor1", "tor2", "tor3"] if tor3_arr is not None else ["tor1", "tor2"]
        metric_array = list(range(3))
        metric_coord = ["metric1", "metric2"] + tor_coord
        
        # Create coordinate dictionary
        coords_dict = {"tor1": tor1_arr, "tor2": tor2_arr}
        if tor3_arr is not None:
            # Use original tor3_arr for coordinates (not the evaluation version)
            if np.size(tor3_arr) > 1:
                coords_dict["tor3"] = np.asarray(tor3_arr)
            else:
                coords_dict["tor3"] = np.array([tor3_arr]) if not isinstance(tor3_arr, np.ndarray) else tor3_arr
        
        # Create ds_gvec_geometry
        ds_gvec_geometry = xr.Dataset(
            data_vars={
                "R0": ((), self.R0),
                "kappa": ((), self.kappa_elongation),
                "delta": ((), self.delta_triangularity),
                "beta": ((), self.beta_toro),
            },
            coords=coords_dict,
        )
        
        # Save R and Z coordinates
        ds_gvec_geometry = ds_gvec_geometry.assign(R_coord=(tor_coord, R_coord))
        ds_gvec_geometry = ds_gvec_geometry.assign(Z_coord=(tor_coord, Z_coord))
        
        # Save metric tensors
        ds_gvec_geometry = ds_gvec_geometry.assign_coords(metric1=("metric1", metric_array))
        ds_gvec_geometry = ds_gvec_geometry.assign_coords(metric2=("metric2", metric_array))
        ds_gvec_geometry = ds_gvec_geometry.assign(
            CovariantMetricTensor=(
                metric_coord,
                CovariantMetricTensor[: len(metric_array), : len(metric_array), ...],
            )
        )
        ds_gvec_geometry = ds_gvec_geometry.assign(
            ContravariantMetricTensor=(
                metric_coord,
                ContravariantMetricTensor[: len(metric_array), : len(metric_array), ...],
            )
        )
        
        # Individual metric tensor components for compatibility
        ds_gvec_geometry = ds_gvec_geometry.assign(CovariantMetricTensor_11=(tor_coord, CovariantMetricTensor[0, 0, ...]))
        ds_gvec_geometry = ds_gvec_geometry.assign(CovariantMetricTensor_12=(tor_coord, CovariantMetricTensor[0, 1, ...]))
        ds_gvec_geometry = ds_gvec_geometry.assign(CovariantMetricTensor_21=(tor_coord, CovariantMetricTensor[1, 0, ...]))
        ds_gvec_geometry = ds_gvec_geometry.assign(CovariantMetricTensor_22=(tor_coord, CovariantMetricTensor[1, 1, ...]))
        ds_gvec_geometry = ds_gvec_geometry.assign(CovariantMetricTensor_33=(tor_coord, CovariantMetricTensor[2, 2, ...]))
        ds_gvec_geometry = ds_gvec_geometry.assign(
            ContravariantMetricTensor_11=(tor_coord, ContravariantMetricTensor[0, 0, ...])
        )
        ds_gvec_geometry = ds_gvec_geometry.assign(
            ContravariantMetricTensor_12=(tor_coord, ContravariantMetricTensor[0, 1, ...])
        )
        ds_gvec_geometry = ds_gvec_geometry.assign(
            ContravariantMetricTensor_21=(tor_coord, ContravariantMetricTensor[1, 0, ...])
        )
        ds_gvec_geometry = ds_gvec_geometry.assign(
            ContravariantMetricTensor_22=(tor_coord, ContravariantMetricTensor[1, 1, ...])
        )
        ds_gvec_geometry = ds_gvec_geometry.assign(
            ContravariantMetricTensor_33=(tor_coord, ContravariantMetricTensor[2, 2, ...])
        )
        
        # Initialisation of the magnetic field
        ds_magnetconf = ds_gvec_geometry[tor_coord].copy()
        ds_magnetconf = ds_magnetconf.assign(dPsidr_tor1=("tor1", dPhi_dr))
        ds_magnetconf = ds_magnetconf.assign(current_tor1=("tor1", current_tor1))
        ds_magnetconf = ds_magnetconf.assign(B_contra=(["metric1"] + tor_coord, B_contra))
        
        # Extract individual B components (ellipsis handles both 2D and 3D cases)
        ds_magnetconf = ds_magnetconf.assign(B_tor1_contra=(tor_coord, B_contra[0, ...]))
        ds_magnetconf = ds_magnetconf.assign(B_tor2_contra=(tor_coord, B_contra[1, ...]))
        ds_magnetconf = ds_magnetconf.assign(B_tor3_contra=(tor_coord, B_contra[2, ...]))
        ds_magnetconf = ds_magnetconf.assign(B_norm=(tor_coord, B_norm))
        
        return ds_gvec_geometry, ds_magnetconf

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
            rho[0] = self.rho_min
        
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

    def get_params(self):
        """
        Get the GVEC parameters dictionary used to create this equilibrium.
        
        Returns:
        --------
        dict
            GVEC parameters dictionary, or None if not available
        """
        if hasattr(self, 'params'):
            return self.params
        return None

    def to_hdf5(self, filename='magnet_config.h5', Nr=16, Ntheta=16, Nzeta=16):
        """
        Save the magnetic configuration to an HDF5 file.
        
        Parameters:
        -----------
        filename : str, optional
            Name of the HDF5 file (default: 'magnet_config.h5')
        Nr : int, optional
            Number of radial grid points (default: 16)
        Ntheta : int, optional
            Number of poloidal grid points (default: 16)
        Nzeta : int, optional
            Number of toroidal grid points (default: 16)
        """
        import h5py
        
        # Get radial range from self.r_array
        rmin = self.r_array[0] if len(self.r_array) > 0 else self.rho_min
        rmax = self.r_array[-1] if len(self.r_array) > 0 else 1.0
        
        # Create mesh
        dr = (rmax - rmin) / (Nr + 1.0)
        tor1 = np.linspace(dr/2, rmax, Nr + 1, True)
        tor2 = np.linspace(0.0, 2.0 * np.pi, Ntheta+1, True)
        tor3 = np.linspace(0.0, 2.0 * np.pi, Nzeta+1, True)
        
        # Compute magnet configuration
        _, CovariantMetricTensor = self.get_gij(tor1, tor2, tor3)
        R_coord, Z_coord = self.get_RZ(tor1, tor2, tor3)
        Psi, dPsidr = self.get_Psi(tor1)
        q = self.get_q(tor1)
        B_contra = self.get_Bcontra(tor1, tor2, tor3)
        J_contra = self.get_Jcontra(tor1, tor2, tor3)
        
        # Save to HDF5
        # Save all toroidal slices (3D geometry)
        nb_tor3 = len(tor3)
        
        with h5py.File(filename, 'w') as f:
            # R and Z coordinates: save all toroidal slices
            f.create_dataset('R', data=R_coord[:,:,:].T)
            f.create_dataset('Z', data=Z_coord[:,:,:].T)
            
            f.create_dataset('psi', data=Psi)
            f.create_dataset('safety_factor', data=q)
            
            # B and J fields: save all toroidal slices
            f.create_dataset('B_gradr', data=B_contra[:,:,:,0].T)
            f.create_dataset('B_gradtheta', data=B_contra[:,:,:,1].T)
            f.create_dataset('B_gradphi', data=B_contra[:,:,:,2].T)
            f.create_dataset('mu0J_gradr', data=J_contra[:,:,:,0].T)
            f.create_dataset('mu0J_gradtheta', data=J_contra[:,:,:,1].T)
            f.create_dataset('mu0J_gradphi', data=J_contra[:,:,:,2].T)
            
            # Metric tensor components: save all toroidal slices
            f.create_dataset('g11', data=CovariantMetricTensor[:,:,:,0,0].T)
            f.create_dataset('g12', data=CovariantMetricTensor[:,:,:,0,1].T)
            f.create_dataset('g13', data=CovariantMetricTensor[:,:,:,0,2].T)
            f.create_dataset('g21', data=CovariantMetricTensor[:,:,:,1,0].T)
            f.create_dataset('g22', data=CovariantMetricTensor[:,:,:,1,1].T)
            f.create_dataset('g23', data=CovariantMetricTensor[:,:,:,1,2].T)
            f.create_dataset('g31', data=CovariantMetricTensor[:,:,:,2,0].T)
            f.create_dataset('g32', data=CovariantMetricTensor[:,:,:,2,1].T)
            f.create_dataset('g33', data=CovariantMetricTensor[:,:,:,2,2].T)

    def plot_geometry(self, N_surf=16, N_theta=32, N_toroidal_plots=3, Nr=128, Ntheta=128):
        """
        Plot the geometry of the magnetic configuration
        
        Parameters:
        -----------
        N_surf : int, optional
            Number of magnetic surfaces to plot (default: 16)
        N_theta : int, optional
            Number of poloidal angles to plot (default: 32)
        N_toroidal_plots : int, optional
            Number of toroidal plots to make (default: 3)
        Nr : int, optional
            Number of radial grid points for plotting (default: 128)
        Ntheta : int, optional
            Number of poloidal grid points for plotting (default: 128)
        """
        # Create coordinate grids for plotting
        tor1 = np.linspace(self.rho_min, 1, Nr)
        tor2 = np.linspace(0.0, 2.0 * np.pi, Ntheta)
        
        # Create toroidal coordinate array
        if N_toroidal_plots > 1:
            tor3 = np.linspace(0.0, 2.0 * np.pi, N_toroidal_plots)
        else:
            tor3 = np.array([0.0])
        
        # Compute R and Z coordinates
        R_coord, Z_coord = self.get_RZ(tor1, tor2, tor3)
        
        nb_grid_tor1 = len(tor1)
        nb_grid_tor2 = len(tor2)
        nb_grid_tor3 = len(tor3)
        
        delta_Nr = nb_grid_tor1 // N_surf
        delta_Ntheta = nb_grid_tor2 // N_theta
        delta_Nphi = nb_grid_tor3 // N_toroidal_plots if nb_grid_tor3 > 1 else 1

        # Create an array of surface indices
        surface_indices = (np.arange(1, N_surf) * delta_Nr)
        surface_indices_theta = (np.arange(0, N_theta) * delta_Ntheta)

        # Create separate figures for each toroidal plot
        for plot_idx in range(N_toroidal_plots):
            if N_toroidal_plots > 1:
                plt.figure(figsize=(8, 8))
                iphi = plot_idx * delta_Nphi
            else:
                iphi = 0
            
            # Plot magnetic surfaces
            for i in surface_indices:
                plt.plot(R_coord[i, :, iphi], Z_coord[i, :, iphi], 'k-', linewidth=0.5)
            
            # Plot outermost surface in red
            plt.plot(R_coord[-1, :, iphi], Z_coord[-1, :, iphi], 'r-', linewidth=1.5)

            # Plot poloidal field lines
            for i in surface_indices_theta:
                plt.plot(R_coord[:, i, iphi], Z_coord[:, i, iphi], 'k-', linewidth=0.5, alpha=0.5)

            plt.axis('equal')
            plt.xlabel('R')
            plt.ylabel('Z')
            if N_toroidal_plots > 1:
                plt.title(f'Magnetic geometry (φ = {tor3[iphi]:.2f})')
            else:
                plt.title('Magnetic geometry')
            plt.grid(True, alpha=0.3)
        
        plt.show()

    def plot_3D(self, Nr=50, Ntheta=90, Nzeta=93):
        """
        Create a 3D plot of the magnetic configuration.
        
        Parameters:
        -----------
        Nr : int, optional
            Number of radial grid points (default: 50)
        Ntheta : int, optional
            Number of poloidal grid points (default: 90)
        Nzeta : int, optional
            Number of toroidal grid points (default: 93)
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create coordinate grids
        rho_3d = np.linspace(self.rho_min, 1.0, Nr)
        theta_3d = np.linspace(0, 2*np.pi, Ntheta)
        zeta_3d = np.linspace(0, 2*np.pi/2, Nzeta)
        
        # Evaluate position vectors
        ev_surface = self._evaluate_gvec(rho_3d, theta_3d, zeta_3d, ["pos"])
        x, y, z = np.asarray(ev_surface.pos)
        
        # Evaluate magnetic axis
        rho_axis = np.array([self.rho_min])
        zeta_axis = np.linspace(0, 2*np.pi, 133)
        ev_axis = self._evaluate_gvec(rho_axis, np.array([0.0]), zeta_axis, ["pos"])
        x_axis, y_axis, z_axis = np.asarray(ev_axis.pos)
        
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot boundary surface
        ax.plot_surface(x[-1, :, :], y[-1, :, :], z[-1, :, :], alpha=0.7, color="blue")
        
        # Plot surface in center
        zeta_3d = np.linspace(-0.2*np.pi, 2*np.pi/1.8, 33)
        ev_mid = self._evaluate_gvec( rho_3d, theta_3d, zeta_3d, ["pos"])
        xmid, ymid, zmid = np.asarray(ev_mid.pos)
        mid_idx = np.argmin(np.abs(rho_3d - 0.5))
        ax.plot_surface(xmid[mid_idx, :, :], ymid[mid_idx, :, :], zmid[mid_idx, :, :], alpha=0.3, color="red")

        # Plot boundary cuts
        for iz in range(0, z.shape[2], 4):
            ax.plot3D(x[-1, :, iz], y[-1, :, iz], z[-1, :, iz], color="k", linewidth=1)
        
        # Plot magnetic axis
        ax.plot3D(x_axis, y_axis, z_axis, color="green", linewidth=2)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Magnetic Configuration')
        ax.set(aspect="equal")
        ax.view_init(25, 140, 0)
        
        plt.tight_layout()
        plt.show()

    def __del__(self):
        """
        Cleanup temporary directory if it was created.
        """
        if hasattr(self, '_temp_dir') and self._temp_dir is not None:
            self._temp_dir.cleanup()


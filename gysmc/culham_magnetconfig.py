# SPDX-License-Identifier: MIT

"""
Created on 2025-11-05

 Contains the Culham magnetic equilibrium configuration class

@author: Z. S. Qu
"""

import numpy as np
from .magnet_config import MagnetConfig


class CulhamMagnetConfig(MagnetConfig):
    """
    Culham magnetic equilibrium configuration class
    
    Implements the Culham equilibrium model for tokamak magnetic configuration
    based on the analytical formulation with shaping parameters.
    
    The transformation is defined as:
        R = R0 + r*cos(theta) + Delta(r) - E(r)*cos(theta) + T(r)*cos(2*theta) - P(r)*cos(theta)
        Z = r*sin(theta) + E(r)*sin(theta) - T(r)*sin(2*theta) - P(r)*sin(theta)
    
    where:
        - Delta(r): Shafranov shift
        - E(r): function related to elongation
        - T(r): function related to triangularity  
        - P(r): P function (epsilon^3 correction)
    """

    def __init__(self, thetastar=True, major_radius=3, q_profile=None, pressure_profile=None, 
                 rmax=1.2, beta_toro=0.0, Shafranov_shift=True,
                 kappa_elongation=1.0, delta_triangularity=0.0, circularJ=False, r_array=None):
        """
        Initialize the Culham equilibrium
        
        All radial functions (g, f, Delta, E, T, P) are computed automatically
        using the exact same numerical methods as the Fortran code.
        
        Parameters:
        -----------
        thetastar : bool, optional
            Use theta* coordinate (default: True)
        major_radius : float
            Major radius at magnetic axis
        q_profile : QProfile
            Safety factor profile object with get_q(r) method
        pressure_profile : array_like
            Pressure profile p(r)
        rmax : float, optional
            Maximum radial coordinate (default: 1.2)
        beta_toro : float, optional
            Toroidal beta parameter (default: 0.0)
        Shafranov_shift : bool, optional
            Enable Shafranov shift computation (default: True)
        kappa_elongation : float, optional
            Elongation at edge (default: 1.0)
        delta_triangularity : float, optional
            Triangularity at edge (default: 0.0)
        circularJ : bool, optional
            Use a dummy pressure profile so the current profile is circular (default: False)
        r_array : array_like, optional
            The radial coordinate array. If None, a default grid will be created.
        """
        super().__init__()

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

        self.thetastar = thetastar
        self.R0 = major_radius
        self.q_profile_obj = q_profile  # Store QProfile object
        self.q_profile, self.dqdr_profile = q_profile.get_q(self.r_array)  # Get q values and derivatives
        self.pressure_profile_obj = pressure_profile
        self.minor_radius = 1.0
        self.beta_toro = beta_toro
        self.Shafranov_shift = Shafranov_shift
        self.kappa_elongation = kappa_elongation
        self.delta_triangularity = delta_triangularity
        
        # Initialize arrays - all functions will be computed automatically
        nr = len(self.r_array)
        self.Delta_r = np.zeros(nr)
        self.Deltaprime_r = np.zeros(nr)
        self.E_r = np.zeros(nr)
        self.Eprime_r = np.zeros(nr)
        self.T_r = np.zeros(nr)
        self.Tprime_r = np.zeros(nr)
        self.Pfunc_r = np.zeros(nr)
        self.Pfuncprime_r = np.zeros(nr)
        self.gfunc_r = np.zeros(nr)
        self.gfuncprime_r = np.zeros(nr)
        self.ffunc_r = np.zeros(nr)
        self.ffuncprime_r = np.zeros(nr)
        self.Psi_r = np.zeros(nr)
        self.dPsidr_r = np.zeros(nr)
        
        # Pressure term - must be computed first
        if self.pressure_profile_obj is None:
            if circularJ:
                # Dummy pressure term computation for circular current case
                self.pressterm_r, self.pressterm_prime_r = self._compute_dummy_pressterm()
            else:
                self.pressterm_prime_r = np.zeros(nr)
                self.pressterm_r = np.zeros(nr)
        else:
            self.pressterm_r, self.pressterm_prime_r = self.pressure_profile_obj.get_pressure(self.r_array)
            self.pressterm_r *= self.beta_toro
            self.pressterm_prime_r *= self.beta_toro
        
        # Compute all magnetic field functions and radial profiles automatically
        # using exact Fortran methods
        self._compute_magnetic_functions()
        self._compute_all_profiles()
        
        # Pre-construct cubic splines for efficient interpolation
        self._construct_splines()

        # Compute thetastar to theta transformation if needed
        if self.thetastar:
            self._compute_thetastar_transformation()

    def _compute_magnetic_functions(self):
        """
        Compute magnetic field functions g(r) and f(r) using exact Fortran methods
        """
        # Compute g(r) using scipy's RK45 method (replacing manual RK4)
        self.gfunc_r = self._compute_g_function_RK4()
        self.gfuncprime_r = self._compute_derivative(self.gfunc_r)
        
        # Compute f(r) = r/(q(r)*R0) * g(r)
        self.ffunc_r = self.r_array * self.gfunc_r / (self.q_profile * self.R0)
        self.ffuncprime_r = self._compute_derivative(self.ffunc_r)

    def _compute_derivative(self, func_array):
        """
        Compute derivative using 4th order finite differences (same as UTL_deriv1)
        """
        nr = len(func_array)
        dr = self.r_array[1] - self.r_array[0]  # Assuming uniform grid
        derivative = np.zeros(nr)
        
        # Interior points - exact match to Fortran UTL_deriv1
        # cdx = 2.0/(3.0*dx) from Fortran: cdx = TW/(TH*dx) with TW=2, TH=3
        cdx = 2.0 / (3.0 * dr)
        for i in range(2, nr-2):
            derivative[i] = cdx * (func_array[i+1] - func_array[i-1] + 
                                 (func_array[i-2] - func_array[i+2]) / 8.0)
        
        # Boundary points (4th order forward/backward difference)
        # Left boundary
        derivative[0] = (-25.0/12.0 * func_array[0] + 4.0 * func_array[1] - 
                        3.0 * func_array[2] + 4.0/3.0 * func_array[3] - 
                        func_array[4]/4.0) / dr
        derivative[1] = (-func_array[0]/4.0 - 5.0/6.0 * func_array[1] + 
                        3.0/2.0 * func_array[2] - 0.5 * func_array[3] + 
                        func_array[4]/12.0) / dr
        
        # Right boundary  
        derivative[nr-1] = (25.0/12.0 * func_array[nr-1] - 4.0 * func_array[nr-2] + 
                           3.0 * func_array[nr-3] - 4.0/3.0 * func_array[nr-4] + 
                           func_array[nr-5]/4.0) / dr
        derivative[nr-2] = (func_array[nr-1]/4.0 + 5.0/6.0 * func_array[nr-2] - 
                           3.0/2.0 * func_array[nr-3] + 0.5 * func_array[nr-4] - 
                           func_array[nr-5]/12.0) / dr
        
        return derivative

    def _compute_g_function_RK4(self):
        """
        Compute g(r) using scipy's RK45 method (replacing manual RK4)
        """
        from scipy.integrate import solve_ivp
        
        def g_ode(r, y):
            """
            ODE system for g(r): dg/dr = _estimate_gprime_scipy(r, g)
            y[0] = g(r)
            """
            g_val = y[0]
            dgdr = self._estimate_gprime_scipy(r, g_val)
            return [dgdr]
        
        # Initial condition: g(0) = 1
        y0 = [1.0]
        
        # Solve the ODE system
        r_span = (self.r_array[0], self.r_array[-1])
        solution = solve_ivp(
            g_ode, 
            r_span, 
            y0, 
            method='RK45',
            t_eval=self.r_array,
            rtol=1e-10,
            atol=1e-12
        )
        
        return solution.y[0]

    def _estimate_gprime_scipy(self, r_pos, g_val):
        """
        Estimate g'(r) for scipy integration
        Based on the differential equation for g(r) from Fortran code
        """
        if r_pos <= 0:
            return 0.0
        
        # Interpolate values at r_pos
        zeta_i, dzetadr_i = self._approx_zeta_and_derivative(r_pos)
        pressterm_prime_i = np.interp(r_pos, self.r_array, self.pressterm_prime_r)
        
        f_i = g_val * zeta_i
        
        if abs(f_i) < 1e-12:
            return 0.0
        
        # g'(r) = -(g*(zeta/r+dzeta/dr)+mu0*p'/(B0^2*zeta*g))/(zeta + 1/(zeta*g))
        numerator = (f_i/r_pos + g_val*dzetadr_i + pressterm_prime_i/f_i)
        denominator = (zeta_i + 1.0/(zeta_i*g_val))
        
        if abs(denominator) < 1e-12:
            return 0.0
        
        return -numerator / denominator

    def _approx_zeta_and_derivative(self, r_pos):
        """
        Compute zeta(r) = r/(q*R0) and dzeta/dr = 1/(R0*q)*(1-s) with s=r/q*dq/dr
        """
        q_interp = np.interp(r_pos, self.r_array, self.q_profile)
        
        # Use the derivative from QProfile object for higher accuracy
        dq_dr = np.interp(r_pos, self.r_array, self.dqdr_profile)
        
        zeta = r_pos / (q_interp * self.R0)
        
        if abs(q_interp) < 1e-12:
            return zeta, 0.0
        
        s = r_pos / q_interp * dq_dr  # Shear parameter
        dzetadr = 1.0 / (self.R0 * q_interp) * (1.0 - s)
        
        return zeta, dzetadr

    def _compute_betap_profile(self):
        """
        Compute beta_p(r) profile as in Fortran:
        beta_p(r) = -2/(r^2*f(r)^2) * integral_0^r r'^2 * beta_toro * p'(r') dr'
        """
        nr = len(self.r_array)
        betap = np.zeros(nr)
        dr = self.r_array[1] - self.r_array[0]
        
        # Extrapolation of pressure at r=0 from first two points
        pressterm0 = (self.pressterm_prime_r[0] - 
                     self.r_array[0]/dr * (self.pressterm_prime_r[1] - self.pressterm_prime_r[0]))
        pressterm1 = (self.pressterm_prime_r[1] - self.pressterm_prime_r[0]) / dr
        
        for i in range(nr):
            r_pos = self.r_array[i]
            if abs(r_pos) < 1e-12 or abs(self.ffunc_r[i]) < 1e-12:
                betap[i] = 0.0
                continue
                
            inv_r2f2 = 1.0 / (r_pos**2 * self.ffunc_r[i]**2)
            
            # Numerical integration of non-constant part
            integral_val = 0.0
            for j in range(i+1):
                r_tmp = self.r_array[j]
                integr_elem = dr  # Integration element
                pressterm_prime_tmp = (self.pressterm_prime_r[j] - pressterm0 - 
                                     pressterm1 * r_tmp)
                integral_val += integr_elem * r_tmp**2 * pressterm_prime_tmp
            
            # Analytical part for constant and linear terms
            integral_val += (pressterm0 * r_pos**3 / 3.0 + 
                           pressterm1 * r_pos**4 / 4.0)
            
            betap[i] = -2.0 * inv_r2f2 * integral_val
        
        return betap

    def _compute_li_profile(self):
        """
        Compute li(r) profile as in Fortran:
        li(r) = 2/(r^2*f(r)^2) * integral_0^r r'*f^2(r') dr'
        """
        nr = len(self.r_array)
        li = np.zeros(nr)
        dr = self.r_array[1] - self.r_array[0]
        
        # Slope of f function
        f1 = (self.ffunc_r[1] - self.ffunc_r[0]) / dr
        
        for i in range(nr):
            r_pos = self.r_array[i]
            if abs(r_pos) < 1e-12 or abs(self.ffunc_r[i]) < 1e-12:
                li[i] = 0.0
                continue
                
            inv_r2f2 = 1.0 / (r_pos**2 * self.ffunc_r[i]**2)
            
            # Numerical integration of higher order terms
            integral_val = 0.0
            for j in range(i+1):
                r_tmp = self.r_array[j]
                integr_elem = dr
                f_tmp = self.ffunc_r[j]
                integral_val += integr_elem * r_tmp * (f_tmp**2 - f1**2 * r_tmp**2)
            
            # Analytical part for f^2 up to r^2
            integral_val += f1**2 * r_pos**4 / 4.0
            
            li[i] = 2.0 * inv_r2f2 * integral_val
        
        return li

    def _compute_Shafranov_shift(self, betap, li, minor_radius):
        """
        Compute Shafranov shift Delta(r) and its derivative
        """
        nr = len(self.r_array)
        Deltaprime = np.zeros(nr)
        Delta = np.zeros(nr)
        dr = self.r_array[1] - self.r_array[0]
        
        # Compute Delta'(r) = -r/R0*(0.5*li(r) + beta_p(r))
        for i in range(nr):
            r_pos = self.r_array[i]
            Deltaprime[i] = -(betap[i] + 0.5 * li[i]) * r_pos / self.R0
        
        # Integrate to get Delta(r), normalized so Delta(minor_radius) = 0
        Delta[0] = 0.0
        for i in range(1, nr):
            Delta[i] = Delta[i-1] + dr * Deltaprime[i]
        
        # Find edge value for normalization
        if minor_radius <= self.r_array[-1]:
            Delta_edge = np.interp(minor_radius, self.r_array, Delta)
            Delta = Delta - Delta_edge
        
        return Delta, Deltaprime

    def _compute_all_profiles(self):
        """
        Compute all radial profiles using the same methods as Fortran code
        """
        # Compute betap and li profiles
        betap = self._compute_betap_profile()
        li = self._compute_li_profile()
        
        # Compute Shafranov shift if enabled and not already provided
        if self.Shafranov_shift and np.all(self.Delta_r == 0) and np.all(self.Deltaprime_r == 0):
            self.Delta_r, self.Deltaprime_r = self._compute_Shafranov_shift(betap, li, self.minor_radius)
        elif not self.Shafranov_shift:
            self.Delta_r = np.zeros(len(self.r_array))
            self.Deltaprime_r = np.zeros(len(self.r_array))
        
        # Compute E and T functions using differential equations
        if np.all(self.E_r == 0) and np.all(self.Eprime_r == 0):
            self.E_r, self.Eprime_r = self._compute_E_function()
        
        if np.all(self.T_r == 0) and np.all(self.Tprime_r == 0):
            self.T_r, self.Tprime_r = self._compute_T_function()
        
        # Normalize E and T functions
        self._normalize_E_and_T_functions()
        
        # Compute P function  
        if np.all(self.Pfunc_r == 0) and np.all(self.Pfuncprime_r == 0):
            self.Pfunc_r, self.Pfuncprime_r = self._compute_P_function()
        
        # Compute Psi profile (depends on f function, so must be computed after magnetic functions)
        self.Psi_r, self.dPsidr_r = self._compute_Psi_profile()

    def _compute_E_function(self):
        """
        Compute E function by solving differential equation:
        E'' + (1/r + 2*f'/f)*E' - 3*E/r^2 = 0
        with initial conditions E(r_min) = r_min, E'(r_min) = 1
        """
        return self._integrate_ET_equation(coeff_0=3.0, init_u=self.r_array[0], init_v=1.0)

    def _compute_T_function(self):
        """
        Compute T function by solving differential equation:
        T'' + (1/r + 2*f'/f)*T' - 8*T/r^2 = 0
        with initial conditions T(r_min) = r_min^2, T'(r_min) = 2*r_min
        """
        return self._integrate_ET_equation(coeff_0=8.0, init_u=self.r_array[0]**2, init_v=2.0*self.r_array[0])

    def _integrate_ET_equation(self, coeff_0, init_u, init_v):
        """
        Integrate the differential equation:
        y'' + (1/r + 2*f'/f)*y' - coeff_0*y/r^2 = 0
        using scipy's RK45 integrator
        
        Rewritten as a first-order system:
        y'[0] = y[1]              (u' = v)
        y'[1] = (coeff_0/r^2)*y[0] - (1/r + 2*f'/f)*y[1]
        """
        from scipy.integrate import solve_ivp
        
        def ode_system(r, y):
            """
            ODE system for E or T function
            y[0] = u (the function)
            y[1] = v (the derivative)
            """
            u, v = y
            
            if abs(r) < 1e-12:
                return [v, 0.0]
            
            # Interpolate f and f' at current r
            f_interp = np.interp(r, self.r_array, self.ffunc_r)
            fp_interp = np.interp(r, self.r_array, self.ffuncprime_r)
            
            if abs(f_interp) < 1e-12:
                logf_approx = 0.0
            else:
                logf_approx = fp_interp / f_interp
            
            du_dr = v
            dv_dr = (coeff_0 / r**2) * u - (1.0/r + 2*logf_approx) * v
            
            return [du_dr, dv_dr]
        
        # Initial conditions
        y0 = [init_u, init_v]
        
        # Solve the ODE system
        r_span = (self.r_array[0], self.r_array[-1])
        solution = solve_ivp(
            ode_system, 
            r_span, 
            y0, 
            method='RK45',
            t_eval=self.r_array,
            rtol=1e-8,
            atol=1e-10
        )
        
        u_prof = solution.y[0]
        v_prof = solution.y[1]
        
        return u_prof, v_prof

    def _compute_P_function(self):
        """
        Compute P function and its derivative:
        P(r) = r^3/(8*R0^2) - r*Delta(r)/(2*R0) - E(r)^2/(2*r) - T(r)^2/r
        P'(r) = 3*r^2/(8*R0^2) - r*Delta'(r)/(2*R0) - Delta(r)/(2*R0) -
                E(r)*E'(r)/r + E(r)^2/(2*r^2) - 2*T(r)*T'(r)/r + T(r)^2/r^2
        """
        nr = len(self.r_array)
        Pfunc = np.zeros(nr)
        Pfuncprime = np.zeros(nr)
        
        for i in range(nr):
            r_pos = self.r_array[i]
            if abs(r_pos) < 1e-12:
                Pfunc[i] = 0.0
                Pfuncprime[i] = 0.0
                continue
            
            # P function
            Pfunc[i] = (r_pos**3 / (8.0 * self.R0**2) - 
                       0.5 * r_pos * self.Delta_r[i] / self.R0 -
                       0.5 * self.E_r[i]**2 / r_pos -
                       self.T_r[i]**2 / r_pos)
            
            # P' function
            Pfuncprime[i] = (3.0 * r_pos**2 / (8.0 * self.R0**2) -
                            0.5 / self.R0 * (r_pos * self.Deltaprime_r[i] + self.Delta_r[i]) -
                            self.E_r[i] / r_pos * (self.Eprime_r[i] - 0.5 * self.E_r[i] / r_pos) -
                            self.T_r[i] / r_pos * (2.0 * self.Tprime_r[i] - self.T_r[i] / r_pos))
        
        return Pfunc, Pfuncprime

    def _compute_dummy_pressterm(self):
        """
        Compute dummy pressure term p'(r) = zeta^2 / (2*r) with zeta = r/(q*R0)
        for circular current case
        """
        nr = len(self.r_array)
        pressterm = np.zeros(nr)
        pressterm_prime = np.zeros(nr)
        dr = self.r_array[1] - self.r_array[0]
        
        # Compute p'(r) = zeta^2 / (2*r)
        for i in range(nr):
            r_pos = self.r_array[i]
            if abs(r_pos) < 1e-12:
                pressterm_prime[i] = 0.0
            else:
                zeta = r_pos / (self.R0 * self.q_profile[i])
                pressterm_prime[i] = zeta**2 / (2.0 * r_pos)
        
        # Integrate to get p(r)
        pressterm[0] = 0.0
        for i in range(1, nr):
            pressterm[i] = pressterm[i-1] + dr * pressterm_prime[i]
        
        return pressterm, pressterm_prime

    def _compute_Psi_profile(self):
        """
        Compute Psi(r) and dPsi/dr profiles using scipy integration
        Psi is the poloidal flux: dPsi/dr = R0 * f(r)
        """
        from scipy.integrate import cumulative_trapezoid
        
        nr = len(self.r_array)
        
        # Compute dPsi/dr = R0 * f(r)
        dPsidr = self.R0 * self.ffunc_r
        
        # Integrate to get Psi using scipy's cumulative trapezoidal integration
        Psi = cumulative_trapezoid(dPsidr, self.r_array, initial=0.0)
        
        return Psi, dPsidr

    def _normalize_E_and_T_functions(self):
        """
        Normalize E and T functions according to elongation and triangularity parameters
        using scipy's nonlinear solver (fsolve)
        """
        from scipy.optimize import fsolve
        
        # Find normalization coefficients at edge
        E_edge = np.interp(self.minor_radius, self.r_array, self.E_r)
        T_edge = np.interp(self.minor_radius, self.r_array, self.T_r)
        
        # Initial guess for E_a and T_a
        E_a_init = (self.kappa_elongation - 1.0) / (self.kappa_elongation + 1.0)
        T_a_init = self.delta_triangularity / 4.0
        
        if abs(self.delta_triangularity) < 1e-3:
            # Approximation used when triangularity is 0 (avoids singularity)
            Pfunc_a = (self.minor_radius / self.R0)**2 / 8.0 - T_a_init**2
            if abs(self.kappa_elongation - 1.0) < 1e-3:
                E_a = E_a_init * (1.0 - Pfunc_a)
            else:
                E_a = (1.0 - np.sqrt(1.0 - 2.0 * (1.0 - Pfunc_a) * E_a_init**2)) / E_a_init
            T_a = T_a_init
        else:
            # Use scipy's fsolve to solve the nonlinear system
            def equations(x):
                """
                System of equations to solve:
                fE(E_a, T_a) = kappa_elongation
                fT(E_a, T_a) = delta_triangularity
                """
                E_a, T_a = x
                
                # Compute P function
                Pfunc_a = (self.minor_radius / self.R0)**2 / 8.0 - E_a**2 / 2.0 - T_a**2
                
                # Compute cos(theta_Z) using the exact formula
                if abs(T_a) < 1e-12:
                    # Handle T_a = 0 case
                    return [E_a - E_a_init, T_a]
                
                discriminant = 1.0 + 32.0 * T_a**2 / (1.0 + E_a - Pfunc_a)**2
                cos_thz = (1.0 + E_a - Pfunc_a) / T_a / 8.0 * (1.0 - np.sqrt(discriminant))
                sin_thz = np.sqrt(max(0.0, 1.0 - cos_thz**2))  # Ensure non-negative
                
                # Compute the functions fE and fT
                fE = ((1.0 + E_a - Pfunc_a) / (1.0 - E_a - Pfunc_a) * 
                      (1.0 - 2.0 * T_a / (1.0 + E_a - Pfunc_a) * cos_thz) * sin_thz)
                fT = (T_a / (1.0 - E_a - Pfunc_a) - 
                      (1.0 + (1.0 + E_a - Pfunc_a) / (1.0 - E_a - Pfunc_a) / 2.0) * cos_thz)
                
                # Return residuals
                return [fE - self.kappa_elongation, fT - self.delta_triangularity]
            
            # Solve the system
            solution = fsolve(equations, [E_a_init, T_a_init], xtol=1e-10, maxfev=1000)
            E_a, T_a = solution
        
        # Compute normalization coefficients
        if abs(E_edge) > 1e-12:
            C_E = self.minor_radius * E_a / E_edge
            self.E_r *= C_E
            self.Eprime_r *= C_E
        
        if abs(T_edge) > 1e-12:
            C_T = self.minor_radius * T_a / T_edge
            self.T_r *= C_T
            self.Tprime_r *= C_T

    def _construct_splines(self):
        """
        Pre-construct cubic splines for all radial profiles for efficient interpolation
        """
        from scipy.interpolate import CubicSpline
        
        # Create splines for all profiles
        self.spline_Delta = CubicSpline(self.r_array, self.Delta_r, bc_type='natural')
        self.spline_Deltaprime = CubicSpline(self.r_array, self.Deltaprime_r, bc_type='natural')
        self.spline_E = CubicSpline(self.r_array, self.E_r, bc_type='natural')
        self.spline_Eprime = CubicSpline(self.r_array, self.Eprime_r, bc_type='natural')
        self.spline_T = CubicSpline(self.r_array, self.T_r, bc_type='natural')
        self.spline_Tprime = CubicSpline(self.r_array, self.Tprime_r, bc_type='natural')
        self.spline_Pfunc = CubicSpline(self.r_array, self.Pfunc_r, bc_type='natural')
        self.spline_Pfuncprime = CubicSpline(self.r_array, self.Pfuncprime_r, bc_type=((1,0.0),(2,0.0)))
        self.spline_gfunc = CubicSpline(self.r_array, self.gfunc_r, bc_type='natural')
        self.spline_gfuncprime = CubicSpline(self.r_array, self.gfuncprime_r, bc_type='natural')
        self.spline_ffunc = CubicSpline(self.r_array, self.ffunc_r, bc_type='natural')
        self.spline_ffuncprime = CubicSpline(self.r_array, self.ffuncprime_r, bc_type='natural')
        self.spline_pressterm = CubicSpline(self.r_array, self.pressterm_r, bc_type='natural')
        self.spline_pressterm_prime = CubicSpline(self.r_array, self.pressterm_prime_r, bc_type='natural')
        self.spline_Psi = CubicSpline(self.r_array, self.Psi_r, bc_type='natural')
        self.spline_dPsidr = CubicSpline(self.r_array, self.dPsidr_r, bc_type='natural')

    def _compute_dR_dZ_transformation(self, r, theta):
        """
        Compute the transformation derivatives dR/dr, dR/dtheta, dZ/dr, dZ/dtheta
        for the transformation from (r, theta) to (R, Z).
        Fully vectorized implementation.

        Parameters:
        -----------
        r : Radial coordinates (r). Can be 1D array of shape (Nr,)
        theta : Poloidal coordinates (theta). Can be 1D array of shape (Ntheta,) or 2D grid of shape (Nr, Ntheta)
        
        Returns:
        --------
        dR_dr : array_like of shape (Nr, Ntheta)
            Derivative of R with respect to r.
        dR_dtheta : array_like of shape (Nr, Ntheta)
            Derivative of R with respect to theta.
        dZ_dr : array_like of shape (Nr, Ntheta)
            Derivative of Z with respect to r.
        dZ_dtheta : array_like of shape (Nr, Ntheta)
            Derivative of Z with respect to theta.
        """
        # Ensure inputs are arrays
        r = np.asarray(r)
        theta = np.asarray(theta)
        
        # Get radial functions using pre-constructed splines (vectorized)
        Deltaprime_pos = self.spline_Deltaprime(r)
        E_pos = self.spline_E(r)
        Eprime_pos = self.spline_Eprime(r)
        T_pos = self.spline_T(r)
        Tprime_pos = self.spline_Tprime(r)
        Pfunc_pos = self.spline_Pfunc(r)
        Pfuncprime_pos = self.spline_Pfuncprime(r)
        
        # If theta is 1D, broadcast to 2D grid (Nr, Ntheta)
        if theta.ndim == 1:
            theta_bc = theta[np.newaxis, :]
        else:
            # theta is already 2D (Nr, Ntheta)
            theta_bc = theta

        r_bc = r[:, np.newaxis]
        # Reshape radial functions to match (Nr, 1) for broadcasting
        Deltaprime_pos = Deltaprime_pos[:, np.newaxis]
        E_pos = E_pos[:, np.newaxis]
        Eprime_pos = Eprime_pos[:, np.newaxis]
        T_pos = T_pos[:, np.newaxis]
        Tprime_pos = Tprime_pos[:, np.newaxis]
        Pfuncprime_pos = Pfuncprime_pos[:, np.newaxis]
        Pfunc_pos = Pfunc_pos[:, np.newaxis]

        # Precompute trigonometric functions (fully vectorized)
        sin_theta = np.sin(theta_bc)
        cos_theta = np.cos(theta_bc)
        sin_2theta = np.sin(2 * theta_bc)
        cos_2theta = np.cos(2 * theta_bc)
        
        # Compute derivatives (fully vectorized)
        # dR/dr = Delta' + cos(theta) * (1 - E' - P') + T' * cos(2*theta)
        dR_dr = (Deltaprime_pos + 
                 cos_theta * (1.0 - Eprime_pos - Pfuncprime_pos) + 
                 Tprime_pos * cos_2theta)

        # dR/dtheta = -sin(theta) * (r - E - P) - 2*T * sin(2*theta)
        dR_dtheta = (-sin_theta * (r_bc - E_pos - Pfunc_pos) - 
                     2.0 * T_pos * sin_2theta)
        
        # dZ/dr = sin(theta) * (1 + E' - P') - T' * sin(2*theta)
        dZ_dr = (sin_theta * (1.0 + Eprime_pos - Pfuncprime_pos) - 
                 Tprime_pos * sin_2theta)
        
        # dZ/dtheta = cos(theta) * (r + E - P) - 2*T * cos(2*theta)
        dZ_dtheta = (cos_theta * (r_bc + E_pos - Pfunc_pos) - 
                     2.0 * T_pos * cos_2theta)
        
        return dR_dr, dR_dtheta, dZ_dr, dZ_dtheta

    def _compute_thetastar_transformation(self):
        """
        Compute the transformation from theta to theta* and vice versa.
        This method sets up the necessary functions for the transformation.
        """
        from scipy.interpolate import CubicSpline
        Nr = self.r_array.size
        Ntheta = 1025  # High resolution for theta grid
        theta_grid = np.linspace(0, 2.0 * np.pi, Ntheta, endpoint=True)

        thetastar = self._get_thetastar_from_theta_(self.r_array, theta_grid)
        thetastar[:,-1] = 2.0 * np.pi # ensure periodicity
        
        thetadiff = thetastar - theta_grid[None, :]
        self.cb_thetadiff = []
        for i in range(Nr):
            cb = CubicSpline(thetastar[i], thetadiff[i], bc_type='periodic')
            self.cb_thetadiff.append(cb)    
        

    def _get_thetastar_from_theta_(self, r, theta):
        """
        Convert theta to theta* using the analytical formula:
        theta* = theta + (Delta' + Delta' E / r - r/R0) * sin(theta)
                       - 1/2 * (E' - E/r) * sin(2*theta)
                       + 1/3 * (T' - 2*T/r) * sin(3*theta)
        
        Fully vectorized implementation that handles grid inputs efficiently.
        
        Parameters:
        -----------
        r : Radial coordinates (r). Can be 1D array of shape (Nr,)
        theta : Poloidal coordinates (theta). Can be 1D array of shape (Ntheta,)

        Returns:
        --------
        array_like : thetastar coordinates with same shape as input
        """
        
        # Get radial functions using pre-constructed splines (vectorized)
        Deltaprime_pos = self.spline_Deltaprime(r)
        E_pos = self.spline_E(r)
        Eprime_pos = self.spline_Eprime(r)
        T_pos = self.spline_T(r)
        Tprime_pos = self.spline_Tprime(r)
        
        # Apply the transformation formula (fully vectorized)
        term1 = (-Deltaprime_pos * ( 1 + E_pos / r) + r / self.R0)[:,None] * np.sin(theta[None, :])
        term2 = 0.5 * (Eprime_pos - E_pos / r)[:,None] * np.sin(2 * theta[None, :])
        term3 = (1.0 / 3.0) * (Tprime_pos - 2.0 * T_pos / r)[:,None] * np.sin(3 * theta[None, :])
        
        thetastar = theta[None, :] - term1 - term2 + term3
        
        return thetastar

    def _get_theta_from_thetastar_(self, r, thetastar):
        """
        Convert theta* to theta by inverting the transformation using spline interpolation.
        Uses _get_thetastar_from_theta_ for the forward transformation.
        
        Parameters:
        -----------
        r : Radial coordinates (r). Can be 1D array of shape (Nr,)
        thetastar : Poloidal coordinates (thetastar). Can be 1D array of shape (Ntheta,)

        Returns:
        --------
        array_like : theta coordinates 2D
        """
        from scipy.interpolate import CubicSpline

        Nrgrid = self.r_array.size
        thetediff_grid = np.zeros((Nrgrid, thetastar.size))
        for i in range(Nrgrid):
            thetediff_grid[i,:] = self.cb_thetadiff[i](thetastar)

        # perform radial interpolation
        cb_thetadiff = CubicSpline(self.r_array, thetediff_grid, axis=0, bc_type='natural')
        thetediff = cb_thetadiff(r)
        theta = thetastar[None, :] - thetediff
        
        return theta

    def _get_jacobian_theta_thetastar_(self, r, thetastar, phi):
        """
        Compute the Jacobian matrix for the transformation from (r, theta*, phi) to (r, theta, phi).
        
        This is the inverse transformation, where:
        r = r*  (unchanged)
        theta = theta* + thetadiff(r, theta*)  (computed via spline interpolation)
        phi = phi*  (unchanged)
        
        The Jacobian matrix is:
        J = [[dr/dr*,      dr/dtheta*,      dr/dphi*    ],
             [dtheta/dr*, dtheta/dtheta*, dtheta/dphi*],
             [dphi/dr*,   dphi/dtheta*,   dphi/dphi*  ]]
        
        where:
        dr/dr* = 1,                  dr/dtheta* = 0,                  dr/dphi* = 0
        dtheta/dr* = ...,            dtheta/dtheta* = ...,            dtheta/dphi* = 0
        dphi/dr* = 0,                dphi/dtheta* = 0,                dphi/dphi* = 1
        
        Parameters:
        -----------
        r : Radial coordinates (r). Can be 1D array of shape (Nr,)
        thetastar : Poloidal coordinates (thetastar). Can be 1D array of shape (Ntheta,)

        Returns:
        --------
        array : Jacobian matrix of shape (..., 3, 3)
                For scalar inputs: (3, 3)
                For array inputs: (N, 3, 3) where N is the number of points
        """
        # Get the dimensions of input
        Nr = np.size(r)
        Ntheta = np.size(thetastar)
        Nphi = np.size(phi)
        
        # inintialize Jacobian matrix
        jacobian = np.zeros((Nr, Ntheta, Nphi, 3, 3))

        # convert thetastar to theta
        theta = self._get_theta_from_thetastar_(r, thetastar)
        
        # Get radial functions and their derivatives using pre-constructed splines
        Deltaprime_pos = self.spline_Deltaprime(r)
        E_pos = self.spline_E(r)
        Eprime_pos = self.spline_Eprime(r)
        T_pos = self.spline_T(r)
        Tprime_pos = self.spline_Tprime(r)
        
        # For second derivatives, use the derivative of splines
        Deltaprimeprime_pos = self.spline_Deltaprime(r, nu=1)  # d²Delta/dr²
        Eprimeprime_pos = self.spline_Eprime(r, nu=1)          # d²E/dr²
        Tprimeprime_pos = self.spline_Tprime(r, nu=1)          # d²T/dr²
        
        # We construct d(r*, theta*, phi*)/d(r, theta, phi) then invert it
        
        # First row: dr*/dr = 1, dr*/dtheta = 0, dr*/dphi = 0
        jacobian[..., 0, 0] = 1.0
        jacobian[..., 0, 1] = 0.0
        jacobian[..., 0, 2] = 0.0
        
        # Second row: dtheta*/dr, dtheta*/dtheta, dtheta*/dphi = 0
        # Compute dtheta*/dr
        # d/dr[(Delta' + r/R0) * sin(theta)]
        dterm1_dr = (-Deltaprimeprime_pos * (1.0 + E_pos/r) 
                     -Deltaprime_pos * Eprime_pos/r 
                     -Deltaprime_pos * (-E_pos/r**2)
                     + 1.0/self.R0)[:, None] * np.sin(theta)
        
        # d/dr[1/2 * (E' - E/r) * sin(2*theta)]
        dterm2_dr = 0.5 * (Eprimeprime_pos - Eprime_pos/r + E_pos/r**2)[:, None] * np.sin(2*theta)
        
        # d/dr[1/3 * (T' - 2*T/r) * sin(3*theta)]
        dterm3_dr = (1.0/3.0) * (Tprimeprime_pos - 2.0*Tprime_pos/r + 2.0*T_pos/r**2)[:, None] * np.sin(3*theta)

        jacobian[..., 1, 0] = (-dterm1_dr - dterm2_dr + dterm3_dr)[:, :, None] + phi[None, None, :] * 0.0  # broadcast to match shape

        # Compute dtheta*/dtheta
        # d/dtheta[(Delta' + r/R0) * sin(theta)]
        dterm1_dtheta = (-Deltaprime_pos * (1.0 + E_pos/r) + r/self.R0)[:, None] * np.cos(theta)
        
        # d/dtheta[1/2 * (E' - E/r) * sin(2*theta)]
        dterm2_dtheta = 0.5 * (Eprime_pos - E_pos/r)[:, None]  * 2.0 * np.cos(2*theta)
        
        # d/dtheta[1/3 * (T' - 2*T/r) * sin(3*theta)]
        dterm3_dtheta = (1.0/3.0) * (Tprime_pos - 2.0*T_pos/r)[:, None] * 3.0 * np.cos(3*theta)
        
        jacobian[..., 1, 1] = (1.0 - dterm1_dtheta - dterm2_dtheta + dterm3_dtheta)[:, :, None] + phi[None, None, :] * 0.0
        
        # dtheta*/dphi = 0 (theta* doesn't depend on phi)
        jacobian[..., 1, 2] = 0.0
        
        # Third row: dphi*/dr = 0, dphi*/dtheta = 0, dphi*/dphi = 1
        jacobian[..., 2, 0] = 0.0
        jacobian[..., 2, 1] = 0.0
        jacobian[..., 2, 2] = 1.0
        
        # Now invert the Jacobian at each point
        jacobian = np.linalg.inv(jacobian)
        
        return jacobian

    def get_RZ(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the R and Z coordinates from toroidal coordinates.
        Fully vectorized implementation.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r) - 1D array of shape (Nr,)
        tor2_arr : array_like
            Poloidal coordinates (theta or theta*) - 1D array of shape (Ntheta,)
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
        
        # Convert theta* to theta if needed (returns 2D array of shape (Nr, Ntheta))
        if self.thetastar:
            theta = self._get_theta_from_thetastar_(tor1_arr, tor2_arr)
        else:
            # Broadcast to 2D: theta[None, :] creates (1, Ntheta), tor1_arr[:, None] creates (Nr, 1)
            theta = tor2_arr[None, :] + tor1_arr[:, None] * 0.0
        
        # Get radial functions using pre-constructed splines (vectorized) - shape (Nr,)
        Delta_pos = self.spline_Delta(tor1_arr)
        E_pos = self.spline_E(tor1_arr)
        T_pos = self.spline_T(tor1_arr)
        Pfunc_pos = self.spline_Pfunc(tor1_arr)
        
        # Reshape radial functions to (Nr, 1) for broadcasting
        r_bc = tor1_arr[:, np.newaxis]
        Delta_bc = Delta_pos[:, np.newaxis]
        E_bc = E_pos[:, np.newaxis]
        T_bc = T_pos[:, np.newaxis]
        Pfunc_bc = Pfunc_pos[:, np.newaxis]
        
        # Precompute trigonometric functions (fully vectorized) - shape (Nr, Ntheta)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_2theta = np.cos(2 * theta)
        sin_2theta = np.sin(2 * theta)
        
        # Compute R and Z coordinates (fully vectorized) - shape (Nr, Ntheta)
        # R = R0 + r*cos(theta) + Delta - E*cos(theta) + T*cos(2*theta) - P*cos(theta)
        R_2d = (self.R0 + 
                r_bc * cos_theta + 
                Delta_bc - 
                E_bc * cos_theta + 
                T_bc * cos_2theta - 
                Pfunc_bc * cos_theta)
        
        # Z = r*sin(theta) + E*sin(theta) - T*sin(2*theta) - P*sin(theta)
        Z_2d = (r_bc * sin_theta + 
                E_bc * sin_theta - 
                T_bc * sin_2theta - 
                Pfunc_bc * sin_theta)
        
        # Add phi dimension: broadcast from (Nr, Ntheta) to (Nr, Ntheta, Nphi)
        if nb_grid_tor3 == 1:
            # Single phi value - just add dimension
            R_coord = R_2d[:, :, np.newaxis]
            Z_coord = Z_2d[:, :, np.newaxis]
        else:
            # Multiple phi values - replicate along phi axis
            R_coord = np.repeat(R_2d[:, :, np.newaxis], nb_grid_tor3, axis=2)
            Z_coord = np.repeat(Z_2d[:, :, np.newaxis], nb_grid_tor3, axis=2)
        
        return R_coord, Z_coord

    def get_gij(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the gij coefficients from toroidal coordinates.
        Fully vectorized implementation using _compute_dR_dZ_transformation.
        
        Parameters:
        -----------
        tor1_arr : array_like
            Radial coordinates (r) - 1D array of shape (Nr,)
        tor2_arr : array_like
            Poloidal coordinates (theta or theta*) - 1D array of shape (Ntheta,)
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
        
        # Convert theta* to theta if needed (returns 2D array of shape (Nr, Ntheta))
        if self.thetastar:
            theta = self._get_theta_from_thetastar_(tor1_arr, tor2_arr)
        else:
            # Broadcast to 2D: theta[None, :] creates (1, Ntheta), tor1_arr[:, None] creates (Nr, 1)
            theta = tor2_arr[None, :] + tor1_arr[:, None] * 0.0
        
        # Compute all transformation derivatives using vectorized function
        # Returns arrays of shape (Nr, Ntheta)
        dR_dr, dR_dtheta, dZ_dr, dZ_dtheta = self._compute_dR_dZ_transformation(tor1_arr, theta)
        
        # Get R coordinates for g33 component (vectorized)
        R_coord_2d, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        R_coord_2d = R_coord_2d[:, :, 0]  # Extract 2D slice (Nr, Ntheta)
        
        # Compute covariant metric components (fully vectorized) - shape (Nr, Ntheta)
        g11 = dR_dr**2 + dZ_dr**2
        g12 = dR_dr * dR_dtheta + dZ_dr * dZ_dtheta
        g13 = np.zeros_like(g11)
        g22 = dR_dtheta**2 + dZ_dtheta**2
        g23 = np.zeros_like(g11)
        g33 = R_coord_2d**2
        
        # Broadcast to 3D and fill covariant metric tensor
        for k in range(nb_grid_tor3):
            # Fill the metric tensor for all (i, j) at once
            CovariantMetricTensor[:, :, k, 0, 0] = g11
            CovariantMetricTensor[:, :, k, 0, 1] = g12
            CovariantMetricTensor[:, :, k, 0, 2] = g13
            CovariantMetricTensor[:, :, k, 1, 0] = g12
            CovariantMetricTensor[:, :, k, 1, 1] = g22
            CovariantMetricTensor[:, :, k, 1, 2] = g23
            CovariantMetricTensor[:, :, k, 2, 0] = g13
            CovariantMetricTensor[:, :, k, 2, 1] = g23
            CovariantMetricTensor[:, :, k, 2, 2] = g33

        if self.thetastar:
            Jacobimat = self._get_jacobian_theta_thetastar_(tor1_arr, tor2_arr, tor3_arr)
            # Adjust covariant metric for thetastar coordinates
            CovariantMetricTensor = np.einsum('...ki,...kl,...lj->...ij', Jacobimat, CovariantMetricTensor, Jacobimat)

        # Compute contravariant metric as inverse (vectorized over all points)
        ContravariantMetricTensor = np.linalg.inv(CovariantMetricTensor)
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
        q_values, _ = self.q_profile_obj.get_q(tor1_arr)
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
            Poloidal coordinates (theta or theta*) - 1D array of shape (Ntheta,)
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

        # Convert theta* to theta if needed and compute Jacobian
        if self.thetastar:
            theta = self._get_theta_from_thetastar_(tor1_arr, tor2_arr)
            Jacobimat = self._get_jacobian_theta_thetastar_(tor1_arr, tor2_arr, tor3_arr)
            Jacobian_det_theta_thetastar = np.linalg.det(Jacobimat)
        else:
            # Broadcast to 2D: theta[None, :] creates (1, Ntheta), tor1_arr[:, None] creates (Nr, 1)
            theta = tor2_arr[None, :] + tor1_arr[:, None] * 0.0
        
        # Compute all transformation derivatives using vectorized function
        # Returns arrays of shape (Nr, Ntheta)
        dR_dr, dR_dtheta, dZ_dr, dZ_dtheta = self._compute_dR_dZ_transformation(tor1_arr, theta)
        
        # Get R coordinates (shape: Nr, Ntheta, Nphi)
        R, _ = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        
        # Compute Jacobian determinant from R-Z transformation (shape: Nr, Ntheta, Nphi)
        Jacobian_det_RZ = (dR_dr * dZ_dtheta - dR_dtheta * dZ_dr)[:, :, None] * R
        
        # Adjust for theta* coordinate if needed
        if self.thetastar:
            jacobian_det = Jacobian_det_RZ * Jacobian_det_theta_thetastar
        else:
            jacobian_det = Jacobian_det_RZ
        
        # Get magnetic field functions using pre-constructed splines (vectorized) - shape (Nr,)
        ffunc_arr = self.spline_ffunc(tor1_arr)
        gfunc_arr = self.spline_gfunc(tor1_arr)
        
        # Reshape for broadcasting: (Nr, 1, 1)
        ffunc_bc = ffunc_arr[:, None, None]
        gfunc_bc = gfunc_arr[:, None, None]
        
        # Compute contravariant metric component g^{phi,phi} = 1/R^2 (vectorized)
        g_phi_phi = 1.0 / R**2
        
        # Contravariant magnetic field components (fully vectorized)
        # B^r = 0
        B_contra[:, :, :, 0] = 0.0
        
        # B^theta = R0 * f(r) / J (shape: Nr, Ntheta, Nphi)
        B_contra[:, :, :, 1] = self.R0 * ffunc_bc / jacobian_det
        
        # B^phi = R0 * g(r) * g^{phi,phi} (shape: Nr, Ntheta, Nphi)
        B_contra[:, :, :, 2] = self.R0 * gfunc_bc * g_phi_phi
        
        return B_contra

    def get_Jcontra(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the current density from toroidal coordinates
        
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
        
        # Get magnetic field for current calculation
        B_contra = self.get_Bcontra(tor1_arr, tor2_arr, tor3_arr)
        
        for i, r_pos in enumerate(tor1_arr):
            # Use cubic splines directly for better performance
            ffunc_pos = self.spline_ffunc(r_pos)
            gfuncprime_pos = self.spline_gfuncprime(r_pos)
            pressterm_prime_pos = self.spline_pressterm_prime(r_pos)
            
            for j, theta_pos in enumerate(tor2_arr):
                for k in range(nb_grid_tor3):
                    phi_pos = tor3_arr if nb_grid_tor3 == 1 else tor3_arr[k]
                    
                    # Current density components
                    # J^r = 0
                    J_contra[i, j, k, 0] = 0.0
                    
                    # J^theta = -dI/dpsi * B^theta
                    dpsidr = self.R0 * ffunc_pos
                    dIdr = self.R0 * gfuncprime_pos
                    J_contra[i, j, k, 1] = -(dIdr / dpsidr) * B_contra[i, j, k, 1]
                    
                    # J^phi = -(dI/dr * B^phi + mu0 * P'(r)) / (dpsi/dr)
                    J_contra[i, j, k, 2] = -(dIdr * B_contra[i, j, k, 2] + 
                                           pressterm_prime_pos) / dpsidr
        
        return J_contra
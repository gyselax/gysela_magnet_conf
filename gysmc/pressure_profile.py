# SPDX-License-Identifier: MIT

"""
Created on 2025-11-06

 Contains the class to construct the q profile

@author: Z. S. Qu
"""

import numpy as np


class PressureProfile():
    """
    Class to initialise the pressure profile
    """

    def __init__(self, nspecies=2, configs=(None, None), rmax=1.2, R0=2.78):
        """
        Initialisation of the pressure profile
        :param nspecies: number of species
        :param configs: tuple of pressure profile configurations for each species
        :param rmax: maximum radius for profiles
        :param R0: major radius

        config structure:
        {
        'fraction': fraction of this species density (for quasi-neutrality)
        'Zs': charge number of species
        'kappa_Ts': temperature profile stiffness
        'deltar_Ts': temperature profile width
        'kappa_ns': density profile stiffness  
        'deltar_ns': density profile width
        'rpeak': peak position for the profile
        'read_ns': flag to read ns profile from file
        'read_Ts': flag to read Ts profile from file
        'ns_filename': filename for ns profile
        'Ts_filename': filename for Ts profile
        }
        """
        from scipy.interpolate import CubicSpline
        import os

        if nspecies < 1:
            raise ValueError("Number of species must be at least 1")
        elif nspecies == 1:
            if not isinstance(configs, tuple):
                configs = (configs,)

            if configs[0]['Zs'] is None:
                raise ValueError("Zs must be provided!")
        
            configs[0]['fraction'] = 1.0 / np.abs(configs[0]['Zs'])
            configs = (configs[0], configs[0])
            configs[1]['Zs'] = -1.0  # assume electron for the second species
            configs[1]['fraction'] = 1.0  # electron density fraction
            nspecies = 2

        if len(configs) != nspecies:
            raise ValueError("Length of configs must match nspecies")
        
        self.configs = configs
        self.nspecies = nspecies
        self.R0 = R0
        self.Zs = np.zeros(nspecies, dtype=int)
        self.ns_spline = []
        self.Ts_spline = []
        self.fraction = np.zeros(nspecies, dtype=float)

        for ispec in range(nspecies):
            config = configs[ispec]
            if 'fraction' not in config:
                raise ValueError(f"fraction must be provided in config for species {ispec}")
            if 'Zs' not in config:
                raise ValueError(f"Zs must be provided in config for species {ispec}")
            self.fraction[ispec] = config.get('fraction', 1.0)
            self.Zs[ispec] = config.get('Zs', 1.0)
            profile_choice = config.get('profile_choice', 1)
            
            # Temperature gradient parameters (kappa_Ts corresponds to kappa_Ts0 in Fortran)
            kappa_Ts = config.get('kappa_Ts', 0.0)
            deltar_Ts = config.get('deltar_Ts', 0.33)
            
            # Density gradient parameters (kappa_ns corresponds to kappa_ns0 in Fortran)  
            kappa_ns = config.get('kappa_ns', 0.0)
            deltar_ns = config.get('deltar_ns', 0.33)
            
            rpeak = config.get('rpeak', 0.5)
            read_ns = config.get('read_ns', False)
            read_Ts = config.get('read_Ts', False)
            ns_filename = config.get('ns_filename', None)
            Ts_filename = config.get('Ts_filename', None)
            if read_ns:
                # Reading of the pressure profile
                if ns_filename is None:
                    raise ValueError(f"ns_filename must be provided when read_ns=True for species {ispec}")
                if os.path.exists(ns_filename):
                    data = np.loadtxt(ns_filename)
                    grid_r = data[:, 0]
                    ns_r = data[:, 1]
                    if grid_r[-1] < rmax:
                        raise ValueError(f"ns profile in {ns_filename} does not cover up to rmax={rmax}")

                else:
                    raise FileNotFoundError(f"n profile file {ns_filename} not found")
            else:
                Nr = 1024
                grid_r = np.linspace(0.0, rmax, Nr + 1, True)

                # Construct analytical ns profile
                ns_r = self._construct_analytical_ns(ispec, grid_r, profile_choice, kappa_ns, deltar_ns, rpeak)

            spline_ns = CubicSpline(grid_r, ns_r, bc_type=('clamped', 'natural'))
            self.ns_spline.append(spline_ns)
                
            if read_Ts:
                if Ts_filename is None:
                    raise ValueError(f"Ts_filename must be provided when read_Ts=True for species {ispec}")
                if os.path.exists(Ts_filename):
                    data = np.loadtxt(Ts_filename)
                    grid_r = data[:, 0]
                    Ts_r = data[:, 1]
                    spline_Ts = CubicSpline(grid_r, Ts_r, bc_type=('clamped', 'natural'))
                    self.Ts_spline.append(spline_Ts)
                    if grid_r[-1] < rmax:
                        raise ValueError(f"Ts profile in {Ts_filename} does not cover up to rmax={rmax}")
                else:
                    raise FileNotFoundError(f"T profile file {Ts_filename} not found")
            else:
                Nr = 1024
                grid_r = np.linspace(0.0, rmax, Nr + 1, True)
            
                # Construct analytical Ts profile
                Ts_r = self._construct_analytical_Ts(ispec, grid_r, profile_choice, kappa_Ts, deltar_Ts, rpeak)
                
            spline_Ts = CubicSpline(grid_r, Ts_r, bc_type=('clamped', 'natural'))
            self.Ts_spline.append(spline_Ts)
        
        # ensure quasi-neutrality
        total_charge = np.sum(self.fraction * self.Zs)
        if np.abs(total_charge) > 1e-6:
            raise ValueError("Sum of total charge is not zero, quasi-neutrality violated")
        
        # enforce the last species density profile to be the sum of the first nspecies-1 species
        Nr = 1024
        grid_r = np.linspace(0.0, rmax, Nr + 1, True)
        ns_r_last = np.zeros(Nr + 1, dtype=float)
        for i in range(nspecies - 1):
            ns_r_last += self.ns_spline[i](grid_r) * self.fraction[i] * self.Zs[i]
        self.ns_spline[-1] = CubicSpline(grid_r, ns_r_last, bc_type=('clamped', 'natural'))

    def _construct_analytical_ns(self, ispec, grid_r, profile_choice, kappa_ns, deltar_ns, rpeak):
        """
        Construct analytical ns profile
        :param ispec: species index
        :param grid_r: radial grid
        :param profile_choice: choice of profile
        :param kappa_ns: gradient parameter
        :param deltar_ns: width parameter
        :param rpeak: peak radius
        """
        Nr = len(grid_r) - 1
        ns_r = np.zeros(Nr + 1, dtype=float)

        inv_Lns0_spec = kappa_ns / self.R0 
        
        if profile_choice == 1:
            # Profile choice 1: sech^2 profile 
            # 1/ns(r) dns(r)/dr = -(1/Lns)*sech^2((r-rpeak)/deltar_ns)
            ns_rmin = 1e19  # Initial value at r=0
            ns_r[0] = ns_rmin
            
            for ir in range(1, Nr + 1):
                r_tmp = grid_r[ir-1]
                dr = grid_r[ir] - grid_r[ir-1]
                rth = r_tmp + dr * 0.5
                # Using sech^2(x) = 1/cosh^2(x)
                tmp = -0.5 * dr * inv_Lns0_spec * (1.0 / np.cosh((rth - rpeak) / deltar_ns))**2
                ns_r[ir] = (1.0 + tmp) / (1.0 - tmp) * ns_r[ir-1]
                
            # Normalize density: int(ns0(r)*r*dr)/int(r*dr) = 1
            self._normalize_density(grid_r, ns_r)
            
        elif profile_choice in [2, 4]:
            # Profile choice 2 & 4: Uses func_tanh and func_sech
            # 1/ns(r) dns(r)/dr = -(1/Lns)*func_sech(r)*func_tanh(r)
            # where func_sech(r) = -1 + sech^2((r-rmin)/deltar) + sech^2((r-rmax)/deltar)
            ns_rmin = 1e19
            ns_r[0] = ns_rmin
            
            # Generate func_tanh for this profile choice
            func_tanh = self._compute_func_tanh(grid_r, profile_choice, deltar_ns)
            
            rmin = grid_r[0]
            rmax = grid_r[Nr]
            
            for ir in range(1, Nr + 1):
                r_tmp = grid_r[ir-1]
                dr = grid_r[ir] - grid_r[ir-1]
                rth = r_tmp + dr * 0.5
                
                # Compute func_sech
                func_sech = (-1.0 + 
                           (1.0 / np.cosh((rth - rmin) / deltar_ns))**2 + 
                           (1.0 / np.cosh((rth - rmax) / deltar_ns))**2)
                
                tmp = 0.5 * dr * inv_Lns0_spec * func_sech * func_tanh[ir]
                ns_r[ir] = (1.0 + tmp) / (1.0 - tmp) * ns_r[ir-1]
                
            # Normalize density
            self._normalize_density(grid_r, ns_r)
            
        elif profile_choice == 3:
            # Profile choice 3: Uses func_tanh_3
            # 1/ns(r) dns(r)/dr = -(1/Lns)*func_tanh_3(r)
            ns_rmin = 1e19
            ns_r[0] = ns_rmin
            
            # Generate func_tanh_3 for this profile choice
            func_tanh = self._compute_func_tanh(grid_r, profile_choice, deltar_ns)
            
            for ir in range(1, Nr + 1):
                dr = grid_r[ir] - grid_r[ir-1]
                tmp = -0.5 * dr * inv_Lns0_spec * func_tanh[ir]
                ns_r[ir] = (1.0 + tmp) / (1.0 - tmp) * ns_r[ir-1]
                
            # Normalize density
            self._normalize_density(grid_r, ns_r)
            
        elif profile_choice == 5:
            # CYCLONE base case profile
            # Exact analytical solution: ns(r) = exp(-kappa*deltar*tanh((r-rpeak)/deltar))
            # NOTE: NO normalization applied (following Fortran implementation)
            for ir in range(Nr + 1):
                r_point = grid_r[ir]
                ns_r[ir] = np.exp(-inv_Lns0_spec * deltar_ns * np.tanh((r_point - rpeak) / deltar_ns))
            
        else:
            # Default case - flat profile
            ns_r[:] = 1.0

        return ns_r

    def _construct_analytical_Ts(self, ispec, grid_r, profile_choice, kappa_Ts, deltar_Ts, rpeak):
        """
        Construct analytical Ts profile
        :param ispec: species index
        :param grid_r: radial grid
        :param profile_choice: choice of profile
        :param kappa_Ts: gradient parameter
        :param deltar_Ts: width parameter
        :param rpeak: peak radius
        """
        Nr = len(grid_r) - 1
        Ts_r = np.zeros(Nr + 1, dtype=float)
        inv_LTs0_spec = kappa_Ts / self.R0
        
        if profile_choice == 1:
            # Profile choice 1: sech^2 profile
            # 1/Ts(r) dTs(r)/dr = -(1/LTs)*sech^2((r-rpeak)/deltar_Ts)
            Ts_rmin = 1.3  # Initial value in keV
            Ts_r[0] = Ts_rmin
            
            for ir in range(1, Nr + 1):
                r_tmp = grid_r[ir-1]
                dr = grid_r[ir] - grid_r[ir-1]
                rth = r_tmp + dr * 0.5
                # Using sech^2(x) = 1/cosh^2(x)
                tmp = -0.5 * dr * inv_LTs0_spec * (1.0 / np.cosh((rth - rpeak) / deltar_Ts))**2
                Ts_r[ir] = (1.0 + tmp) / (1.0 - tmp) * Ts_r[ir-1]
                
            # Normalize temperature to 1 at r=rpeak
            self._normalize_temperature(grid_r, Ts_r, rpeak)
            
        elif profile_choice in [2, 4]:
            # Profile choice 2 & 4: Uses func_tanh and func_sech
            # 1/Ts(r) dTs(r)/dr = -(1/LTs)*func_sech(r)*func_tanh(r)
            # where func_sech(r) = -1 + sech^2((r-rmin)/deltar) + sech^2((r-rmax)/deltar)
            Ts_rmin = 2.0  # Initial value in keV
            Ts_r[0] = Ts_rmin
            
            # Generate func_tanh for this profile choice
            func_tanh = self._compute_func_tanh(grid_r, profile_choice, deltar_Ts)
            
            rmin = grid_r[0]
            rmax = grid_r[Nr]
            
            for ir in range(1, Nr + 1):
                r_tmp = grid_r[ir-1]
                dr = grid_r[ir] - grid_r[ir-1]
                rth = r_tmp + dr * 0.5
                
                # Compute func_sech
                func_sech = (-1.0 + 
                           (1.0 / np.cosh((rth - rmin) / deltar_Ts))**2 + 
                           (1.0 / np.cosh((rth - rmax) / deltar_Ts))**2)
                
                tmp = 0.5 * dr * inv_LTs0_spec * func_sech * func_tanh[ir]
                Ts_r[ir] = (1.0 + tmp) / (1.0 - tmp) * Ts_r[ir-1]
                
            # Normalize temperature to 1 at r=rpeak
            self._normalize_temperature(grid_r, Ts_r, rpeak)
            
        elif profile_choice == 3:
            # Profile choice 3: Uses func_tanh_3 with flux-driven considerations
            # 1/Ts(r) dTs(r)/dr = -invLTs0*func_tanh_3(r)
            Ts_rmin = 2.0  # Initial value in keV
            Ts_r[0] = Ts_rmin
            
            # Generate func_tanh_3 for this profile choice
            func_tanh = self._compute_func_tanh(grid_r, profile_choice, deltar_Ts)
            
            # Flux driven parameters (simplified - assuming not flux driven)
            R0 = 1.0  # Major radius - should be passed as parameter
            invLTs0_min = 1.0 / R0
            shift = inv_LTs0_spec / (inv_LTs0_spec / invLTs0_min - 1.0) if inv_LTs0_spec != invLTs0_min else 0.0
            
            for ir in range(1, Nr + 1):
                dr = grid_r[ir] - grid_r[ir-1]
                invLTs0_r = (inv_LTs0_spec * func_tanh[ir] + shift) / (inv_LTs0_spec + shift) * inv_LTs0_spec
                tmp = -0.5 * dr * invLTs0_r
                Ts_r[ir] = (1.0 + tmp) / (1.0 - tmp) * Ts_r[ir-1]
                
            # Normalize temperature to 1 at r=rpeak
            self._normalize_temperature(grid_r, Ts_r, rpeak)
            
        elif profile_choice == 5:
            # CYCLONE base case profile
            # Exact analytical solution: Ts(r) = exp(-kappa*deltar*tanh((r-rpeak)/deltar))
            # NOTE: NO normalization applied (following Fortran implementation)
            for ir in range(Nr + 1):
                r_point = grid_r[ir]
                Ts_r[ir] = np.exp(-inv_LTs0_spec * deltar_Ts * np.tanh((r_point - rpeak) / deltar_Ts))
            
        else:
            # Default case - flat profile
            Ts_r[:] = 1.0

        return Ts_r

    def _normalize_density(self, grid_r, ns_r):
        """
        Normalize density profile: int(ns0(r)*r*dr)/int(r*dr) = 1
        Based on density_normalization subroutine from init_prof_func.F90
        """
        Nr = len(grid_r) - 1
        dr = grid_r[1] - grid_r[0]  # Assuming uniform grid
        
        # Compute int(ns0(r)*r*dr)
        ns0norm_tmp = 0.0
        for ir in range(1, Nr):
            ns0norm_tmp += ns_r[ir] * grid_r[ir]
        ns0norm_tmp += 0.5 * (ns_r[0] * grid_r[0] + ns_r[Nr] * grid_r[Nr])
        
        # Divide by int(r*dr) = (rmax^2 - rmin^2) / 2
        ns0norm_tmp = ns0norm_tmp * 2.0 * dr / (grid_r[Nr]**2 - grid_r[0]**2)
        
        # Normalize
        ns_r[:] = ns_r[:] / ns0norm_tmp

    def _normalize_temperature(self, grid_r, Ts_r, rpeak):
        """
        Normalize temperature profile to 1 at r=rpeak
        Based on temperature_normalization subroutine from init_prof_func.F90
        """
        Nr = len(grid_r) - 1
        dr = grid_r[1] - grid_r[0]  # Assuming uniform grid
        
        # Find interpolation weights for rpeak
        ir = int((rpeak - grid_r[0]) / dr)
        if ir >= Nr:
            ir = Nr - 1
        if ir < 0:
            ir = 0
            
        w1 = (rpeak - grid_r[ir]) / dr
        w0 = 1.0 - w1
        
        # Interpolate temperature at rpeak
        if ir + 1 <= Nr:
            Ts0_norm = w0 * Ts_r[ir] + w1 * Ts_r[ir + 1]
        else:
            Ts0_norm = Ts_r[ir]
            
        # Normalize
        if Ts0_norm != 0:
            Ts_r[:] = Ts_r[:] / Ts0_norm

    def _compute_func_tanh(self, grid_r, profile_choice, deltar):
        """
        Compute the hyperbolic tangent functions used in profile choices 2, 3, and 4
        Based on init_prof_func_tanh_* subroutines from init_prof_func.F90
        """
        Nr = len(grid_r) - 1
        func_tanh = np.zeros(Nr + 1, dtype=float)
        
        rmin = grid_r[0]
        rmax = grid_r[Nr]
        Lr = rmax - rmin
        
        if profile_choice == 1:
            # func_tanh_1: Used for profile choice 1
            rbuff = 0.25 * Nr * (grid_r[1] - grid_r[0])
            for ir in range(Nr + 1):
                ri = grid_r[ir]
                func_tanh_tmp = (np.tanh(0.5 * (ri - rmin - rbuff)) - 
                                np.tanh(0.5 * (ri - rmax + rbuff)))
                func_tanh[ir] = func_tanh_tmp
            # Normalize by max value
            max_func = np.max(func_tanh)
            if max_func != 0:
                func_tanh = func_tanh / max_func
                
        elif profile_choice == 2:
            # func_tanh_2: Required for invariance test comparisons
            rbuff1 = 0.2 * Lr
            rbuff2 = 0.2 * Lr
            dr = grid_r[1] - grid_r[0]
            for ir in range(Nr + 1):
                ri = grid_r[ir]
                func_tanh_tmp = (np.tanh(0.5 * (ri - rmin - rbuff1) / dr) - 
                                np.tanh(0.5 * (ri - rmax + rbuff2) / dr))
                func_tanh[ir] = func_tanh_tmp
            # Normalize by max value
            max_func = np.max(func_tanh)
            if max_func != 0:
                func_tanh = func_tanh / max_func
                
        elif profile_choice == 3:
            # func_tanh_3: Flat logarithmic gradients outside buffer regions
            # Simplified version without all parameters
            rbuff = 0.15 * Lr  # Simplified buffer size
            lambda_val = deltar * Lr
            minor_radius = rmax  # Simplified assumption
            
            for ir in range(Nr + 1):
                ri = grid_r[ir]
                func_tanh_tmp = 0.5 * (np.tanh((ri - rmin - rbuff) / lambda_val) - 
                                     np.tanh((ri - rmax + rbuff) / lambda_val))
                # Apply radial correction factor
                func_tanh[ir] = func_tanh_tmp * (1.0 + 0.7 * (0.5 - ri / minor_radius))
            
            # Normalize by central value
            central_idx = Nr // 2
            norm_val = 0.5 * (func_tanh[central_idx] + func_tanh[central_idx + 1])
            if norm_val != 0:
                func_tanh = 2.0 * func_tanh / norm_val
                
        elif profile_choice == 4:
            # func_tanh_4: Same as func_tanh_2 but without normalization by dr
            rbuff1 = 0.2 * Lr
            rbuff2 = 0.2 * Lr
            for ir in range(Nr + 1):
                ri = grid_r[ir]
                func_tanh_tmp = (np.tanh(0.5 * (ri - rmin - rbuff1)) - 
                                np.tanh(0.5 * (ri - rmax + rbuff2)))
                func_tanh[ir] = func_tanh_tmp
            # Normalize by max value
            max_func = np.max(func_tanh)
            if max_func != 0:
                func_tanh = func_tanh / max_func
                
        elif profile_choice == 5:
            # For CYCLONE profiles, func_tanh = 1 (no hyperbolic tangent used)
            func_tanh[:] = 1.0
            
        return func_tanh


    def get_Ts(self, ispecies, tor1_arr):
        """
        Create the Ts profile from toroidal coordinate 1
        :param tor1_arr: array of toroidal coordinates 1
        :return: Ts, dTsdr
        """
        # Use spline interpolation for smooth Ts profile and derivatives
        Ts = self.Ts_spline[ispecies](tor1_arr)
        dTsdr = self.Ts_spline[ispecies](tor1_arr, nu=1)  # First derivative

        return Ts, dTsdr
    
    def get_ns(self, ispecies, tor1_arr):
        """
        Create the ns profile from toroidal coordinate 1
        :param tor1_arr: array of toroidal coordinates 1
        :return: ns, dnsdr
        """
        # Use spline interpolation for smooth ns profile and derivatives
        ns = self.ns_spline[ispecies](tor1_arr)
        dnsdr = self.ns_spline[ispecies](tor1_arr, nu=1)  # First derivative

        return ns, dnsdr
    
    def get_pressure(self, tor1_arr):
        """
        Create the total pressure profile from toroidal coordinate 1
        :param tor1_arr: array of toroidal coordinates 1
        :return: total pressure, dPdr
        """
        total_pressure = np.zeros_like(tor1_arr)
        dPdr = np.zeros_like(tor1_arr)

        for ispec in range(self.nspecies):
            ns, dnsdr = self.get_ns(ispec, tor1_arr)
            Ts, dTsdr = self.get_Ts(ispec, tor1_arr)
            pressure_spec = ns * Ts * self.fraction[ispec]
            dPdr_spec = (dnsdr * Ts + ns * dTsdr) * self.fraction[ispec]

            total_pressure += pressure_spec
            dPdr += dPdr_spec

        return total_pressure, dPdr

    @classmethod
    def from_ds_radialprofiles(cls, ds_radialprofiles):
        """
        Create a PressureProfile from ds_radialprofiles Dataset
        Similar to QProfile but reads pressure_tor1 and tor1 from ds_radialprofiles
        
        :param ds_radialprofiles: xarray Dataset containing pressure_tor1 and tor1 coordinates
        :return: PressureProfile instance with get_pressure method
        """
        from scipy.interpolate import CubicSpline
        
        # Extract pressure_tor1 and tor1 values from ds_radialprofiles
        if "pressure_tor1" not in ds_radialprofiles.data_vars:
            raise ValueError("pressure_tor1 not found in ds_radialprofiles")
        if "tor1" not in ds_radialprofiles.coords:
            raise ValueError("tor1 coordinate not found in ds_radialprofiles")

        tor1 = ds_radialprofiles.coords["tor1"].values
        pressure_tor1 = ds_radialprofiles["pressure_tor1"].values

        # Create a simple wrapper class instance
        instance = cls.__new__(cls)
        
        # Store the spline interpolation
        spline = CubicSpline(tor1, pressure_tor1, bc_type='natural')
        instance.spline = spline
        instance.tor1 = tor1
        instance.pressure_tor1 = pressure_tor1
        
        # Override get_pressure to use the spline interpolation
        def get_pressure(tor1_arr):
            """
            Create the pressure profile from toroidal coordinate 1
            :param tor1_arr: array of toroidal coordinates 1
            :return: pressure, dpdr
            """
            pprof = spline(tor1_arr)
            dpdr = spline(tor1_arr, nu=1)  # First derivative
            return pprof, dpdr
        
        instance.get_pressure = get_pressure
        
        return instance
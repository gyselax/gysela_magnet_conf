# SPDX-License-Identifier: MIT

"""
Created on 2025-11-03

 Contains the class to construct the q profile

@author: Z. S. Qu
"""

import numpy as np


class QProfile():
    """
    Class to initialise the q profile
    """

    def __init__(self, option=2, q_param1=1.0, q_param2=2.7858, q_param3=2.8, q_param4=1.0,
                 read_q=False, q_filename=None, rmax=1.2, rmin=0.0, number_radial_points=1024):
        """
        Initialisation of the q profile
        :param option: option for q profile (int or str). 
                       Can be: 0 or "infinity", 1 or "flat", 2 or "parabolic", 
                       3 or "reversed shear", 4 or "cyclone"/"cycone", 5 or "wesson"
        :param q_param1,q_param2,q_param3,q_param4: parameter for q profile
        :param read_q: file to read q profile from, if None, use analytical profile
        :param q_filename: file to load q profile
        :param rmax: maximum radius for q profile
        """
        from scipy.interpolate import CubicSpline
        import os

        # Map string options to numeric options
        option_map = {
            "infinity": 0,
            "flat": 1,
            "parabolic": 2,
            "reversed shear": 3,
            "cyclone": 4,
            "cycone": 4,  # Support both spellings
            "wesson": 5
        }
        
        # Convert string option to number if needed
        if isinstance(option, str):
            option_lower = option.lower().strip()
            if option_lower in option_map:
                option = option_map[option_lower]
            else:
                raise ValueError(f"Unknown q_profile option '{option}'. "
                               f"Valid options are: {list(option_map.keys())} or integers 0-5")
        
        self.option = option
        self.q_param1 = q_param1
        self.q_param2 = q_param2
        self.q_param3 = q_param3
        self.q_param4 = q_param4

        if read_q:
            # Reading of the q profile
            if q_filename is None:
                raise ValueError("q_filename must be provided when read_q is True")
            if os.path.exists(q_filename):
                data = np.loadtxt(q_filename)
                self.grid_r = data[:, 0]
                self.q_r = data[:, 1]
            else:
                raise FileNotFoundError(f"Q profile file {q_filename} not found")
        else:
            # the following code was converted from FORTRAN by Copilot
            # Use default grid
    
            Nr = number_radial_points
            self.grid_r = np.linspace(0.0, rmax, Nr + 1, True)
            minor_radius = 1.0
            
            self.q_r = np.zeros(Nr + 1)
            
            if option == 0:
                # Infinity q profile -> iota = 0
                INF = 1e30
                self.q_r[:] = INF
                
            elif option == 1:
                # Flat case
                self.q_r[:] = q_param1
                
            elif option == 2:
                # Parabolic case
                for ir in range(Nr + 1):
                    ri = self.grid_r[ir]
                    if ri != 0:
                        qr_tmp = q_param2 * (ri / minor_radius) ** q_param3
                    else:
                        qr_tmp = 0
                    self.q_r[ir] = q_param1 + qr_tmp
      
            elif option == 3:
                # Reversed shear, Garbet et al. PoP2001 version
                rhomin2 = q_param4 ** 2
                rhomin4 = rhomin2 ** 2
                rhomin6 = rhomin4 * rhomin2
                var = rhomin4 * (1 - rhomin2) ** 2
                C2 = (rhomin6 * (q_param3 - q_param2) + 
                      (1 - rhomin2) ** 3 * (q_param1 - q_param2)) / var
                C3 = (rhomin4 * (q_param3 - q_param2) - 
                      (1 - rhomin2) ** 2 * (q_param1 - q_param2)) / var
                      
                for ir in range(Nr + 1):
                    ri = self.grid_r[ir]
                    rho = ri / minor_radius
                    rho2 = rho ** 2
                    qr_tmp = (q_param2 + C2 * (rho2 - rhomin2) ** 2 + 
                             C3 * (rho2 - rhomin2) ** 3)
                    self.q_r[ir] = qr_tmp
                    
            elif option == 4:
                # Profile used for CYCLONE base case comparison
                # q(r) = q_param1 + q_param2 * (ri/a) + q_param3 * (ri/a)^2
                for ir in range(Nr + 1):
                    ri = self.grid_r[ir]
                    ri_on_a = ri / minor_radius
                    qr_tmp = q_param1 + q_param2 * ri_on_a + q_param3 * ri_on_a ** 2
                    self.q_r[ir] = qr_tmp
                    
            elif option == 5:
                # Profile of WESSON type for TEARING modes
                # q(r) = q_param1 (r/a)^2 / (1- (1-(r/a)^2)^(q_param2+1))
                # q_param1 is the value at r=a
                for ir in range(Nr + 1):
                    ri = self.grid_r[ir]
                    ri_on_a = ri / minor_radius
                    ri_on_a2 = ri_on_a ** 2
                    qr_tmp = (q_param1 * ri_on_a2 / 
                             (1 - (1 - ri_on_a2) ** (q_param2 + 1)))
                    self.q_r[ir] = qr_tmp
                    
            else:
                raise ValueError(f"q_profile option {option} is not supported")
                
        # Create spline interpolation for smooth derivatives
        # use natural boundary conditions, same as GYSELA
        self.spline = CubicSpline(self.grid_r, self.q_r, bc_type='natural')

    def get_q(self, tor1_arr):
        """
        Create the q profile from toroidal coordinate 1
        :param tor1_arr: array of toroidal coordinates 1
        :return: q, dqdr
        """
        # Use spline interpolation for smooth q profile and derivatives
        qprof = self.spline(tor1_arr)
        dqdr = self.spline(tor1_arr, nu=1)  # First derivative
        
        return qprof, dqdr
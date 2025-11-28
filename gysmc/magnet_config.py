# SPDX-License-Identifier: MIT

"""
Created on 2025-07-15

 Contains the abstract class to initialise the magnetic configuration

@author: Z. S. Qu
"""

import numpy as np

from abc import ABC, abstractmethod

class MagnetConfig(ABC):
    """
    Class to initialise the magnetic configuration
    Must implement the following methods:
        - get_RZ
        - get_gij
        - get_Psi
        - get_q
        - get_Bcontra
        - get_Jcontra
    """

    def __init__(self):
        """
        Initialisation of the magnetic configuration
        """
        pass
  
    @abstractmethod
    def get_RZ(self, tor1_arr, tor2_arr, tor3_arr):
        """
        Create the R and Z coordinates from toroidal coordinates
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: array of toroidal coordinates 3
        :return: tuple of (R, Z) coordinates
        """
        nb_grid_tor1 = len(tor1_arr)
        nb_grid_tor2 = len(tor2_arr)
        nb_grid_tor3 = np.size(tor3_arr)
        shape_RZ = (nb_grid_tor1, nb_grid_tor2, nb_grid_tor3)
        R_coord = np.zeros(shape_RZ, dtype=float)
        Z_coord = np.zeros(shape_RZ, dtype=float)
        return R_coord, Z_coord
    
    @abstractmethod
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
        ContravariantMetricTensor = np.zeros(shape_cocontra, dtype=float)
        return ContravariantMetricTensor, CovariantMetricTensor

    @abstractmethod
    def get_Psi(self, tor1_arr):
        """
        Get Psi, dPsidr
        :param tor1_arr: array of toroidal coordinates 1
        :return: Psi, dPsidr
        """
        nb_grid_tor1 = len(tor1_arr)
        shape_dPsidr = (nb_grid_tor1)
        dPsidr = np.zeros(shape_dPsidr, dtype=float)
        Psi = np.zeros(shape_dPsidr, dtype=float)
        return Psi, dPsidr

    @abstractmethod
    def get_q(self, tor1_arr):
        """
        Get q
        :param tor1_arr: array of toroidal coordinates 1
        :return: q
        """
        nb_grid_tor1 = len(tor1_arr)
        shape_q = (nb_grid_tor1,)
        q = np.zeros(shape_q, dtype=float)
        return q

    @abstractmethod
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
        return B_contra

    @abstractmethod
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

        return J_contra
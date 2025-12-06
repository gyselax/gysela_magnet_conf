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
    
    def test_gij2D(self, tor1_arr=None, tor2_arr=None, tor3_arr=0.0):
        """
        Compare g_ij to finite difference from R,Z, for a tokamak configuration
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: the slice of toroidal coordinate 3 (default 0.0)
        """
        print("INFO: Testing gij comparison with finite difference")
        if tor1_arr is None:
            tor1_arr = np.linspace(0.001, 1.0, 257)
        if tor2_arr is None:
            tor2_arr = np.linspace(0.0, 2.0*np.pi, 256, endpoint=False)
        tor3_arr = np.array([tor3_arr])
        ContravariantMetricTensor, CovariantMetricTensor = self.get_gij(tor1_arr, tor2_arr, tor3_arr)
        R_coord, Z_coord = self.get_RZ(tor1_arr, tor2_arr, tor3_arr)
        R_coord = np.concatenate((R_coord[:,-1:,0], R_coord[:,:,0]), axis=1)
        Z_coord = np.concatenate((Z_coord[:,-1:,0], Z_coord[:,:,0]), axis=1)
        tor2_arrext = np.concatenate((tor2_arr[-1:] - 2.0*np.pi, tor2_arr))
        # compute gij from finite difference
        dR_dtor1 = np.gradient(R_coord, tor1_arr, axis=0)[:,1:]
        dR_dtor2 = np.gradient(R_coord, tor2_arrext, axis=1)[:,1:]
        dZ_dtor1 = np.gradient(Z_coord, tor1_arr, axis=0)[:,1:]
        dZ_dtor2 = np.gradient(Z_coord, tor2_arrext, axis=1)[:,1:]
        g11_fd = dR_dtor1**2 + dZ_dtor1**2
        g12_fd = dR_dtor1 * dR_dtor2 + dZ_dtor1 * dZ_dtor2
        g22_fd = dR_dtor2**2 + dZ_dtor2**2

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title('g11 comparison')
        npolshow = 3
        for i in range(0, len(tor2_arr)-1, len(tor2_arr)//npolshow):
            plt.plot(tor1_arr, CovariantMetricTensor[:,i,0,0,0], label=r'get_gij, $\theta={:.2f}$'.format(tor2_arr[i]))
            plt.plot(tor1_arr, g11_fd[:,i], '--', label=r'FD, $\theta={:.2f}$'.format(tor2_arr[i]))
        plt.xlabel('r')
        plt.legend()
        plt.subplot(1,3,2)
        plt.title('g12/r comparison')
        for i in range(0, len(tor2_arr)-1, len(tor2_arr)//npolshow):
            plt.plot(tor1_arr, CovariantMetricTensor[:,i,0,0,1]/tor1_arr, label=r'get_gij, $\theta={:.2f}$'.format(tor2_arr[i]))
            plt.plot(tor1_arr, g12_fd[:,i]/tor1_arr, '--', label=r'FD, $\theta={:.2f}$'.format(tor2_arr[i]))
        plt.xlabel('r')
        plt.legend()
        plt.subplot(1,3,3)
        plt.title('g22/r^2 comparison')
        for i in range(0, len(tor2_arr)-1, len(tor2_arr)//npolshow):
            plt.plot(tor1_arr, CovariantMetricTensor[:,i,0,1,1]/tor1_arr**2, label=r'get_gij, $\theta={:.2f}$'.format(tor2_arr[i]))
            plt.plot(tor1_arr, g22_fd[:,i]/tor1_arr**2, '--', label=r'FD, $\theta={:.2f}$'.format(tor2_arr[i]))
        plt.xlabel('r')
        plt.legend()
        plt.tight_layout()

    def test_current2D(self, tor1_arr=None, tor2_arr=None, tor3_arr=0.0):
        """
        Compare current to finite difference from B field, for tokamak configuration
        :param tor1_arr: array of toroidal coordinates 1
        :param tor2_arr: array of toroidal coordinates 2
        :param tor3_arr: the slice of toroidal coordinate 3 (default 0.0)
        """

        print("INFO: Testing current comparison with finite difference")
        if tor1_arr is None:
            tor1_arr = np.linspace(0.001, 1.0, 257)
        if tor2_arr is None:
            tor2_arr = np.linspace(0.0, 2.0*np.pi, 256, endpoint=False)
        tor3_arr = np.array([tor3_arr])

        tor2_arr = np.concatenate((tor2_arr[-1:] - 2.0*np.pi, tor2_arr))

        B_contra = self.get_Bcontra(tor1_arr, tor2_arr, tor3_arr)
        J_contra = self.get_Jcontra(tor1_arr, tor2_arr, tor3_arr)
        ContravariantMetricTensor, CovariantMetricTensor = self.get_gij(tor1_arr, tor2_arr, tor3_arr)
        jacobian_det = np.sqrt(np.linalg.det(CovariantMetricTensor))
        B_co = np.einsum('...ij,...j->...i', CovariantMetricTensor, B_contra)

        # compute curl of B in toroidal coordinates
        dB3_dtor2 = np.gradient(B_co[:,:,:,2], tor2_arr, axis=1)
        dB2_dtor3 = np.zeros_like(B_co[:,:,:,0])
        dB1_dtor3 = np.zeros_like(B_co[:,:,:,0])
        dB3_dtor1 = np.gradient(B_co[:,:,:,2], tor1_arr, axis=0, edge_order=2)
        dB2_dtor1 = np.gradient(B_co[:,:,:,1], tor1_arr, axis=0, edge_order=2)
        dB1_dtor2 = np.gradient(B_co[:,:,:,0], tor2_arr, axis=1, edge_order=2)

        curlB_contra = np.zeros_like(B_contra)
        curlB_contra[:,:,:,0] = (1.0/jacobian_det) * (dB3_dtor2 - dB2_dtor3)
        curlB_contra[:,:,:,1] = (1.0/jacobian_det) * (dB1_dtor3 - dB3_dtor1)
        curlB_contra[:,:,:,2] = (1.0/jacobian_det) * (dB2_dtor1 - dB1_dtor2)

        curlB_contra = curlB_contra[:,1:,0]  # remove the extra tor2 point
        J_contra = J_contra[:,1:,0]  # remove the extra tor2 point
        tor2_arr = tor2_arr[1:]  # remove the extra tor2 point

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.title('J1 comparison')
        npolshow = 4
        for i in range(0, len(tor2_arr)-1, len(tor2_arr)//npolshow):
            plt.plot(tor1_arr, J_contra[:,i,0], label=r'get_Jcontra, $\theta={:.2f}$'.format(tor2_arr[i]))
            plt.plot(tor1_arr, curlB_contra[:,i,0], '--', label=r'curlB/μ0, $\theta={:.2f}$'.format(tor2_arr[i]))
        plt.xlabel('r')
        plt.legend()
        plt.subplot(1,3,2)
        plt.title('J2 comparison')
        for i in range(0, len(tor2_arr)-1, len(tor2_arr)//npolshow):
            plt.plot(tor1_arr, J_contra[:,i,1], label=r'get_Jcontra, $\theta={:.2f}$'.format(tor2_arr[i]))
            plt.plot(tor1_arr, curlB_contra[:,i,1], '--', label=r'curlB/μ0, $\theta={:.2f}$'.format(tor2_arr[i]))
        plt.xlabel('r')
        plt.legend()
        plt.subplot(1,3,3)
        plt.title('J3 comparison')
        for i in range(0, len(tor2_arr)-1, len(tor2_arr)//npolshow):
            plt.plot(tor1_arr, J_contra[:,i,2], label=r'get_Jcontra, $\theta={:.2f}$'.format(tor2_arr[i]))
            plt.plot(tor1_arr, curlB_contra[:,i,2], '--', label=r'curlB/μ0, $\theta={:.2f}$'.format(tor2_arr[i]))
        plt.xlabel('r')
        plt.legend()
        plt.tight_layout()


   

"""
Created on 2025-11-4

 Contains the class to initialise the magnetic configuration for GYSELA simulations

@author: P. Donnel, V. Grandgirard, Z. S. Qu
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

class GYSMagnetConfig:

    def __init__(self, magnetconfig, Nr=255, Ntheta=256, minor_radius=180.0, rmin=0.0, rmax=1.0, skiphole=True):
        """
        Initialisation of the magnetic configuration from the magnetconfig class
        :param magconfig: the magnetconfig class
        """
        self.magnetconfig = magnetconfig
        self.Nr = Nr
        self.Ntheta = Ntheta
        self.minor_radius = minor_radius
        self.rmin = rmin
        self.rmax = rmax
        self.skiphole = skiphole

        self.compute_mesh()
        self.compute_magnet_config()

    def compute_mesh(self):
        """
        Compute of the mesh
        """
        if self.skiphole:
            dr = (self.rmax) / (self.Nr + 0.5)
            self.torr1 = np.linspace(dr/2, self.rmax, self.Nr + 1, True)
        else:
            dr = (self.rmax - self.rmin) / (self.Nr + 1.0)
            self.torr1 = np.linspace(self.rmin, self.rmax, self.Nr + 1, True)
        
        self.torr2 = np.linspace(0.0, 2.0 * np.pi, self.Ntheta+1, True)
        self.torr3 = np.array([0.0])  # 2D geometry

    def compute_magnet_config(self):
        """
        Compute magnet configuration
        """

        _, self.CovariantMetricTensor= self.magnetconfig.get_gij(self.torr1, self.torr2, self.torr3)
        self.R_coord, self.Z_coord = self.magnetconfig.get_RZ(self.torr1, self.torr2, self.torr3)
        self.Psi, self.dPsidr = self.magnetconfig.get_Psi(self.torr1)
        self.q = self.magnetconfig.get_q(self.torr1)
        self.B_contra = self.magnetconfig.get_Bcontra(self.torr1, self.torr2, self.torr3)
        self.J_contra = self.magnetconfig.get_Jcontra(self.torr1, self.torr2, self.torr3)

        # need to scale everything by minor radius
        self.r_coord = self.torr1 * self.minor_radius
        self.R_coord *= self.minor_radius
        self.Z_coord *= self.minor_radius
        self.dPsidr *= self.minor_radius
        self.Psi *= self.minor_radius**2
        self.B_contra[:,:,:,1:] /= self.minor_radius
        self.J_contra[:,:,:,0] /= self.minor_radius
        self.J_contra[:,:,:,1:] /= self.minor_radius**2
        self.CovariantMetricTensor[:,:,:,0,1] *= self.minor_radius
        self.CovariantMetricTensor[:,:,:,1,0] *= self.minor_radius
        self.CovariantMetricTensor[:,:,:,1,1] *= self.minor_radius**2
        self.CovariantMetricTensor[:,:,:,1,2] *= self.minor_radius**2
        self.CovariantMetricTensor[:,:,:,2,1] *= self.minor_radius**2
        self.CovariantMetricTensor[:,:,:,2,2] *= self.minor_radius**2

    def plot_q(self):

        plt.plot(self.torr1, self.q, 'k')
        plt.xlabel('r/a')
        plt.ylabel('q(r)')
        plt.title('q profile')
        plt.grid()

    def plot_geometry(self, N_surf=16, N_theta_plot=32):
        """
        Plot the geometry of the magnetic configuration
        :param N_surf: number of magnetic surfaces to plot
        :param N_theta_plot: number of poloidal angles to plot
        """

        nb_grid_tor1 = np.size(self.torr1)
        nb_grid_tor2 = np.size(self.torr2)
        delta_Nr = nb_grid_tor1 // N_surf
        delta_Ntheta = nb_grid_tor2 // N_theta_plot

        # Create an array of surface indices
        surface_indices = (np.arange(1, N_surf) * delta_Nr)
        surface_indices_theta = (np.arange(0, N_theta_plot) * delta_Ntheta)

        # Use .isel() to select the relevant slices for R and Z
        iphi = 0   # Fix toroidal direction to phi=0 if 3D geometry
        for i in surface_indices:
            plt.plot(self.R_coord[i, :, iphi], self.Z_coord[i, :, iphi], 'k-')
        
        plt.plot(self.R_coord[-1, :, iphi], self.Z_coord[-1, :, iphi], 'r-')

        for i in surface_indices_theta:
            plt.plot(self.R_coord[:, i, iphi], self.Z_coord[:, i, iphi], 'k-')

        plt.axis('equal')
        plt.xlabel('R')
        plt.ylabel('Z')
        plt.title('Magnetic geometry')


    def to_hdf5(self, filename='magnet_config.h5'):
        """
        Save the magnetic configuration to an HDF5 file
        :param filename: name of the HDF5 file
        """
        import h5py

        with h5py.File(filename, 'w') as f:
            f.create_dataset('R', data=self.R_coord[:,:,0].T)
            f.create_dataset('Z', data=self.Z_coord[:,:,0].T)
            f.create_dataset('psi', data=self.Psi)
            f.create_dataset('safety_factor', data=self.q)
            f.create_dataset('B_gradr', data=self.B_contra[:,:,0,0].T)
            f.create_dataset('B_gradtheta', data=self.B_contra[:,:,0,1].T)
            f.create_dataset('B_gradphi', data=self.B_contra[:,:,0,2].T)
            f.create_dataset('mu0J_gradr', data=self.J_contra[:,:,0,0].T)
            f.create_dataset('mu0J_gradtheta', data=self.J_contra[:,:,0,1].T)
            f.create_dataset('mu0J_gradphi', data=self.J_contra[:,:,0,2].T)
            f.create_dataset('g11', data=self.CovariantMetricTensor[:,:,0,0,0].T)
            f.create_dataset('g12', data=self.CovariantMetricTensor[:,:,0,0,1].T)
            f.create_dataset('g13', data=self.CovariantMetricTensor[:,:,0,0,2].T)
            f.create_dataset('g21', data=self.CovariantMetricTensor[:,:,0,1,0].T)
            f.create_dataset('g22', data=self.CovariantMetricTensor[:,:,0,1,1].T)
            f.create_dataset('g23', data=self.CovariantMetricTensor[:,:,0,1,2].T)
            f.create_dataset('g31', data=self.CovariantMetricTensor[:,:,0,2,0].T)
            f.create_dataset('g32', data=self.CovariantMetricTensor[:,:,0,2,1].T)
            f.create_dataset('g33', data=self.CovariantMetricTensor[:,:,0,2,2].T)
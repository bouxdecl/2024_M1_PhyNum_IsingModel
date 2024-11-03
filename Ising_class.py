import numpy as np
import scipy
import scipy.signal as signal
import scipy.constants
import scipy.optimize
import scipy.ndimage

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from IPython.display import HTML
import ipywidgets as widgets

from utils import *


class IsingModel:
    
    def __init__(self, N=50, kernel=None, J=1, h=0, T=1, lattice_init=None, boundary:str='periodic'):
        self.N = N
        self.T = T
        self.h = h
        self.J = J
        self.kB = 1 #scipy.constants.k

        if lattice_init is None:
            lattice_init = np.random.choice([-1, 1], size=(self.N, self.N))
        
        elif lattice_init.shape != (self.N,self.N):
            raise ValueError("lattice_init.shape has to be ({},{})".format(self.N,self.N))
        
        self.lattice = lattice_init
        self.boundaries = boundary

        if kernel is None:
            kernel = np.array([[0, 1, 0],
                               [1, 0, 1],
                               [0, 1, 0]])
        self.kernel = kernel
        self.sublattice_size = self.kernel.shape[0]//2 +1
        
        self.frames = []
        self.h_frames = []
        self.T_frames = []
        self.M_frames = []
        self.E_frames = []

    def plot(self):
        fig, ax = plt.subplots()
        img = ax.imshow(self.lattice, cmap='gray', interpolation='nearest', vmin=-1, vmax=1)

        # Custom legend
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Spin Up'),
            Patch(facecolor='black', label='Spin Down')]
        
        # Position the legend outside the plot area
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)

        ax.set_title('2D Ising Model Animation, T={:.2f}, M={:.2f}'.format(self.T, self.magnetization()))

        ax.set_xticks([])  # Hide x ticks
        ax.set_yticks([])  # Hide y ticks
        
        plt.tight_layout()  # Adjust layout to make room for the legend
        plt.show()

    
    def magnetization(self):
        return np.sum(self.lattice) / self.lattice.shape[0]**2

    def graph_magnetization(self, Ti, Tf, dT, n_samples, n_sample_close=100, n_eq=1000, n_first_eq=3000):

        if Tf > 2.4 :
            T_list = np.sort( np.concatenate([np.arange(Ti, Tf, dT), np.arange(2.2, 2.4, 0.01)]) )

        else :
            T_list = np.arange(Ti, Tf, dT)
            
        M_measurements = np.zeros((T_list.shape[0], 2))

        self.T = Ti
        self.simulate(n_first_eq)

        for i, T in enumerate(T_list):
            self.T = T
            self.simulate(n_eq)

            M_list = []

            n_mean = n_samples
            if 2.2 < T and T < 2.4:
                n_mean = n_sample_close
            for _ in range(n_mean):
                M_list.append(self.magnetization())
                self.simulate(n_eq)
            M_measurements[i, 0] = float(np.mean(M_list))
            M_measurements[i, 1] = float(np.std(M_list))
        
        return T_list, M_measurements

   
    def correlation(self,neighbour_choice):

        nb_distances = self.N//2 + 1
        correlation_values = np.zeros(nb_distances)

        for d in range(nb_distances):
            neighbour = neighbour_choice(d)
            correlation_matrix = self.lattice * signal.convolve2d(self.lattice, neighbour, mode='same', boundary='wrap')
            correlation_values[d] = np.sum(correlation_matrix)
        
        return correlation_values/(self.N**2)
        

    def calculate_xi(self,correlation):
        TF_0 = np.sum(correlation)  
        TF_2pi_L = np.sum([correlation[d] * 2 * np.cos(2*np.pi*d / self.N) for d in range(len(correlation))])
        
        xi = (self.N / (2 * np.pi)) * np.sqrt(TF_0 / TF_2pi_L - 1)
    
        return xi
        
    def get_main_interface(self, plot=False):
        
        # find the main connexe cluster
        A = padd_opposite_bound(self.lattice.copy())
        A[A==-1] = 0

        f1 = remove_non_border_components(A)
        f  = remove_non_border_components(np.logical_not(f1))

        # find the lenght of the interface
        kernel = self.kernel
        cov = signal.convolve2d(f, kernel, mode='valid')

        cov[cov==4] = 0
        interface = cov*A[1:-1, 1:-1]
        
        if plot:
            plt.imshow(~f, cmap='gray')
            plt.title('Main connex cluster, lenght={}'.format(np.sum(interface)))

        return interface
    
    def get_fractal_dimension(self, plot=False):
        interface = self.get_main_interface(plot=plot)
        if plot:
            plt.show()
            plt.imshow(interface)
            plt.show()
        
        curve = interface.copy()
        curve[curve!=0] = 1

        curve_coordinates = np.column_stack(np.nonzero(curve))

        box_sizes = np.linspace(1, 20, 100)
        dimension = fractal_dimension(curve_coordinates, box_sizes=box_sizes, plot=plot)
        
        return dimension
    
    def graph_fractal_dimension(self, T_list=np.arange(0.001, 3, 0.05), n_sample=10, n_eq=2000, plot=False):
        
        D_measurements = np.zeros((len(T_list), 2))

        for i, T in enumerate(T_list):
            self.T = T
            D_T_list = []
            for _ in range(n_sample):
                self.simulate(n_eq)
                D_T_list.append(self.get_fractal_dimension())
            D_measurements[i, 0] = float(np.mean(D_T_list))
            D_measurements[i, 1] = float(np.std (D_T_list))

        if plot:
            plt.errorbar(T_list, D_measurements[:, 0], yerr=D_measurements[:, 1],
             
             fmt='o', color='red', markersize=3, 
             ecolor='orange', capsize=2,
             label='Mean over {} points with {} equilibrations steps'.format(n_sample, n_eq)
             )
            plt.axvline(x=2.269, label='Tc', ls='--', c='gray')
            plt.xlabel('T')
            plt.ylabel('fractal dimension')
            plt.title('Fractal dimension around the transition')
            plt.legend()
            
        return T_list, D_measurements


    def energy_grid(self):
        if self.boundaries == 'periodic':
            return -self.h*self.lattice - self.J* self.lattice* signal.convolve2d(self.lattice, self.kernel, mode='same', boundary='wrap')
        if self.boundaries == 'fixed_opposite':
            lattice_pad = padd_opposite_bound(self.lattice)
            return -self.h*self.lattice - self.J* self.lattice* signal.convolve2d(lattice_pad, self.kernel, mode='valid')

    def get_energy(self):
        if self.boundaries == 'periodic':
            return float(np.sum(-self.h*self.lattice - 1/4 *self.J* self.lattice* signal.convolve2d(self.lattice, self.kernel, mode='same', boundary='wrap'))) / self.N**2
        if self.boundaries == 'fixed_opposite':
            lattice_pad = padd_opposite_bound(self.lattice)
            return float(np.sum(-self.h*self.lattice - 1/4*self.J* self.lattice* signal.convolve2d(lattice_pad, self.kernel, mode='valid'))) / self.N**2


    def one_step(self):
        # CHOOSE SUBLATTICE (idx = boolean indexing)
        m0, m1 = np.random.randint(self.sublattice_size+1, size=(2))
        idx = np.zeros(shape=self.lattice.shape, dtype=bool)
        idx[m0::self.sublattice_size, m1::self.sublattice_size] = True

        # METROPOLIS STEP
        delta_H = -2* self.energy_grid()
        
        proba = np.minimum(1, np.exp(-delta_H[idx] / (self.kB *self.T) ))
        flip = np.random.uniform(size=proba.shape) < proba
        self.lattice[idx] *= np.where(flip, -1, 1)  # if flip=True, then *-1

    def simulate(self, n_steps, Ti=None, Tf=None, exp_decay=None, save_data=False, save_frames=False, frames_save_rate=20, verbose=False):

        # define temperatures during simulation
        if Ti is None: Ti = self.T
        if Tf is None: list_T = [Ti]*n_steps
        else:
            if exp_decay:
                list_T = [Tf + (Ti-Tf)*np.exp(-exp_decay* step/n_steps) for step in range(n_steps)]
            else:
                list_T = list(np.linspace(Ti, Tf, n_steps))
        
        # simulation
        for step, T in enumerate(list_T):
            if step%100 == 0 and verbose:
                print(f"Step {step}/{n_steps}: Magnetization = {self.magnetization()}, Temperature = {self.T:.2f}")
            
            # SAVE PARAMETERS
            if save_frames & step%frames_save_rate==0: #pour ne sauver que tous les 10 images
                self.frames.append(self.lattice.copy())
            if save_data:
                self.T_frames.append(self.T)
                self.h_frames.append(self.h)
                self.M_frames.append(self.magnetization())
                self.E_frames.append(self.get_energy())
            
            # ONE STEP
            self.T = T
            self.one_step()

    
    def animate(self):
        """Create an animation of the Ising model simulation."""
        fig, ax = plt.subplots()
        img = ax.imshow(self.frames[0], cmap='gray', interpolation='nearest', vmin=-1, vmax=1)
        ax.set_title('2D Ising Model Animation, T={:.2f}'.format(self.T_frames[0]))
        
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Spin Up'),
            Patch(facecolor='black', label='Spin Down')]
        
        # Position the legend outside the plot area
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=True)
        
        ax.set_xticks([])  # Hide x ticks
        ax.set_yticks([])  # Hide y ticks
        
        def update(frame):
            img.set_array(self.frames[frame])
            ax.set_title('2D Ising Model Animation, T={:.2f}'.format(self.T_frames[frame]))
            return img,
            
        plt.close(fig) #pour ne pas avoir l'image qui s'affiche en plus de la vidéo
        
        return HTML(FuncAnimation(fig, update, frames=len(self.frames), interval=5000/len(self.T_frames), blit=True).to_jshtml())
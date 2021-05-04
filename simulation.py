import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from datetime import datetime
from celluloid import Camera
import os


@jit(nopython=True)
def seed():
    np.random.seed(0)

class Lattice:
    def __init__(self, save_dir=None, initial_state='all_zero', beta=0.05, Nx=20, mu=np.array([0., 0., 0., -20.]),
                 N_samp=20, E=-15., bond_prob=0.5, write_config=True):
        # Defined parameters
        self.beta = beta  # inverse temperature
        self.Nx = Nx  # number of lattice sites in each dimension
        self.mu = mu  # chemical potentials for different phoshporylation states
        self.N_samp = N_samp  # collect a sample for trajectories every N_samp sweeps
        self.E = E  # bond energy, stabilizing bond has E<0
        self.bond_prob = bond_prob  # probability of attemptng to change the state of a bond in each Monte Carlo step

        # Derived parameters
        self.N = Nx ** 2  # total number of lattice sites
        self.positions = self.lattice_positions()
        self.save_dir=save_dir

        # Initialize Containers
        if initial_state == 'all_zero':
            self.lattice_state = np.zeros((self.Nx, self.Nx), dtype=np.int8)
        elif initial_state == 'all_one':
            self.lattice_state = np.ones((self.Nx, self.Nx), dtype=np.int8)
        else:
            raise NotImplementedError('given initial state is not implemented')

        self.lattice_trajectory = np.reshape(np.copy(self.lattice_state),(1, self.Nx, self.Nx))
        self.total_bonds = np.zeros_like(self.lattice_state)
        self.bonds = np.zeros((self.Nx, self.Nx, 3), dtype=np.int8)
        if write_config:
            f=open(self.save_dir+'config.txt','w')
            f.write('beta={} Nx={} mu={} N_samp={} E={} bond_prob={}'.format(self.beta, self.Nx, self.mu, self.N_samp,
                                                                             self.E, self.bond_prob))


    def lattice_positions(self, side_length=1.):
        # Returns an array of the positions off the lattice sites in Cartesian coordinates.
        positions = np.zeros((self.Nx, self.Nx, 2), dtype=np.float64)
        vert_dist = side_length * np.sqrt(3) / 2
        for j in range(self.Nx):
            if j % 2 == 0:
                positions[0, j] = [0., vert_dist * j]
                for i in range(1, self.Nx):
                    if i % 2 == 0:
                        positions[i, j] = positions[i - 1, j] + np.array([side_length * 2, 0.])
                    else:
                        positions[i, j] = positions[i - 1, j] + np.array([side_length, 0.])
            else:
                positions[0, j] = [-side_length / 2, vert_dist * j]
                for i in range(1, self.Nx):
                    if i % 2 == 0:
                        positions[i, j] = positions[i - 1, j] + np.array([side_length, 0.])
                    else:
                        positions[i, j] = positions[i - 1, j] + np.array([side_length * 2, 0.])
        return positions

    def advance(self, iterations):
        # Uses the advance_lattice function to advance the lattice system through iterations Monte Carlo sweeps
        self.lattice_state, self.bonds, self.total_bonds, lattice_trajectory=\
            advance_lattice(self.lattice_state, self.N_samp, iterations, self.Nx, self.N,
                            self.beta, self.E, self.mu, self.bond_prob, self.bonds, self.total_bonds)
        self.lattice_trajectory=np.concatenate((self.lattice_trajectory,lattice_trajectory),axis=0)

    def phospho_trajectory(self):
        # Returns a trajectory of the phosphorylation states of a lattice given configurations of the lattice through
        # time steps. Each row represents a lattice configuration in the lattice trajectory, and each column represents
        # the proportion of a phosphorylation state relative to the lattice size.
        state_trajectory = np.zeros((self.lattice_trajectory.shape[0], 4))
        for i in range(state_trajectory.shape[0]):
            state_trajectory[i] = np.bincount(self.lattice_trajectory[i].flatten(), minlength=4).astype(np.float64) / \
                                  self.N
        return state_trajectory

    def phospho_state_positions(self, lattice):
        # Returns an array of the positions of the lattice sites and the corresponding phosphorylation state at each
        # site given a lattice phosphorylation state.
        state_position = np.zeros((self.N, 3))
        for i in range(self.Nx):
            for j in range(self.Nx):
                index = i * self.Nx + j
                state_position[index][0:2] = self.positions[i, j]
                state_position[index][2] = lattice[i, j]
        return state_position

    def phospho_state_position_trajectory(self):
        state_position_trajectory = np.zeros((len(self.lattice_trajectory), self.N, 3))
        for i in range(len(self.lattice_trajectory)):
            state_position_trajectory[i] = self.phospho_state_positions(self.lattice_trajectory[i])
        return state_position_trajectory

    def plot_phospho_trajectory(self):
        # Plots and saves a given trajectory of phosphorylation states of the lattice.
        print('Plotting Phosphorylation State Trajectory')
        phospho_state_trajectory=self.phospho_trajectory()
        iterations = phospho_state_trajectory.shape[0] * self.N_samp
        plt.clf()
        for n in range(4):
            plt.plot(np.arange(0, iterations, self.N_samp), phospho_state_trajectory[:, n], label='{} pY'.format(n))
        plt.xlabel('MC Step')
        plt.ylabel('Population Fraction')
        plt.legend()
        plt.savefig(self.save_dir + 'phospho_state_trajectory.png')
        return None

    def plot_lattice_trajectory_movie(self):
        # Plots a movie of a lattice trajectory
        print('Plotting Movie')
        state_position_trajectory=self.phospho_state_position_trajectory()
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_aspect('equal')
        camera = Camera(fig)
        # set up artist objects so that each plot shares the same legend regardless of the presence or absence of specific
        # phosphorylation states
        scatter = plt.scatter([0, 0, 0, 0], [0, 0, 0, 0], c=[0, 1, 2, 3])
        artists, _ = scatter.legend_elements()
        for state_position in state_position_trajectory:
            plt.scatter(state_position[:, 0], state_position[:, 1], c=state_position[:, 2], vmin=0, vmax=3)
            plt.legend(handles=artists, labels=['0 pY', '1 pY', '2 pY', '3 pY'], bbox_to_anchor=(1.05, 1))
            plt.tight_layout()
            camera.snap()
        animation = camera.animate(interval=500, blit=True)
        animation.save(self.save_dir + 'movie.gif', writer='imagemagick')

@jit(nopython=True)
def mc_sweep(lattice, Nx, N, beta, E, mu, bond_prob, bonds, total_bonds):
    for _ in range(N):
        lattice_site = np.random.randint(0, N)
        x_i, y_i = (lattice_site // Nx, lattice_site % Nx)
        move = np.random.rand()
        acceptance = np.random.rand()
        if move > bond_prob:  # move will add or remove a phosphate group
            move = (1 - move) / (1 - bond_prob)  # normalize the random number
            state = lattice[x_i, y_i]
            if move >= 0.5 and state < 3:  # only allow adding phosphate when there is an unphosphorylated Y
                threshold = min(1, np.exp(beta * (mu[state + 1] - mu[state])))
                if acceptance < threshold:
                    lattice[x_i, y_i] += 1
            elif move < 0.5 and state > total_bonds[x_i, y_i]:
                # only allow removing phosphate when there is a free phoshporylated Y
                threshold = min(1, np.exp(beta * (mu[state - 1] - mu[state])))
                if acceptance < threshold:
                    lattice[x_i, y_i] -= 1
        else:  # move will add or remove a bond
            move = move / bond_prob
            if move > 2 / 3:
                bond_i = 2
            elif move > 1 / 3:
                bond_i = 1
            else:
                bond_i = 0
            x_j, y_j, bond_j = bond(x_i, y_i, bond_i, Nx)
            if bonds[x_i, y_i, bond_i] == 0 and lattice[x_i, y_i] > total_bonds[x_i, y_i] and \
                    lattice[x_j, y_j] > total_bonds[x_j, y_j]:
                # only allow bond to be added when relevant lattice sites have free phosphates
                threshold = min(1, np.exp(-beta * E))
                if acceptance < threshold:
                    bonds[x_i, y_i, bond_i] = 1
                    bonds[x_j, y_j, bond_j] = 1
                    total_bonds[x_i, y_i] += 1
                    total_bonds[x_j, y_j] += 1
            elif bonds[x_i, y_i, bond_i] == 1:  # only allow bond to be removed when the bond exists
                threshold = min(1, np.exp(beta * E))
                if acceptance < threshold:
                    bonds[x_i, y_i, bond_i] = 0
                    bonds[x_j, y_j, bond_j] = 0
                    total_bonds[x_i, y_i] -= 1
                    total_bonds[x_j, y_j] -= 1
    return lattice, bonds, total_bonds

@jit(nopython=True)
def bond(x_i, y_i, bond_i, Nx):
    # This function returns the information about a bond of interest. Given that we want to know about the bond_i th
    # bond of the lattice site with coordinate x_i, y_i, returns the lattice site x_j, y_j that it bonds with, and that
    # it is the bond_j th bond for that site.
    if (x_i + y_i) % 2 == 0:
        if bond_i == 0:
            x_j, y_j = (x_i + 1, y_i)
        elif bond_i == 1:
            x_j, y_j = (x_i, y_i - 1)
        else:
            x_j, y_j = (x_i, y_i + 1)
    else:
        if bond_i == 0:
            x_j, y_j = (x_i - 1, y_i)
        elif bond_i == 1:
            x_j, y_j = (x_i, y_i + 1)
        else:
            x_j, y_j = (x_i, y_i - 1)
    return x_j % Nx, y_j % Nx, bond_i  # enforce periodic boundary conditions

@jit(nopython=True)
def advance_lattice(lattice, N_samp, iterations, Nx, N, beta, E, mu, bond_prob, bonds, total_bonds):
    # Uses the mc_sweep function to advance a lattice through iterations sweeps
    print('Performing ' + str(iterations) + ' Monte Carlo sweeps')
    samples = iterations // N_samp
    lattice_trajectory = np.zeros((samples, Nx, Nx), dtype=np.int8)
    for iteration in range(iterations):
        lattice, bonds, total_bonds=mc_sweep(lattice, Nx, N, beta, E, mu, bond_prob, bonds, total_bonds)
        if (iteration+1) % N_samp == 0:
            lattice_trajectory[iteration // N_samp] = lattice
    return lattice, bonds, total_bonds, lattice_trajectory


# lattice = initialize_lattice('all_zero')
# lattice_trajectory = mc_sweeps(lattice, bond_prob=0)
# # if plot_trajectory:
# #     phospho_state_trajectory = phospho_trajectory(lattice_trajectory)
# #     plot_phospho_trajectory(phospho_state_trajectory)
# # print('Final State of Lattice')
# # print(lattice_trajectory[-1])
# # # positions=lattice_positions()
# # # state_position=phospho_state_positions(lattice_trajectory[-1],positions)
# plot_lattice_trajectory_movie(lattice_trajectory)

seed()
save_dir = 'experiments/{}/'.format(datetime.now().strftime("%Y_%m_%d_%H_%M"))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
sim_lattice = Lattice(save_dir)
sim_lattice.advance(200)
sim_lattice.plot_phospho_trajectory()
sim_lattice.plot_lattice_trajectory_movie()
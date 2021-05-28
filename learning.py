from simulation import seed, Lattice, plot_lattice
import numpy as np
from datetime import datetime
from sklearn.decomposition import PCA
import os
import pickle

params= {
    'initial_state':'random',
    'beta':1.,
    'Nx':20,
    'mu':np.array([0., 0., 0., 0.]),
    'N_samp':10,
    'E':-3.
}

def chemical_potentials(phi):
    return np.array([0,np.log(3*phi), np.log(3*phi**2), 3*np.log(phi)])

np.random.seed(0)
seed()
save_dir = 'experiments/{}/'.format(datetime.now().strftime("%Y-%m-%d-%H_%M_%S"))
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
sim_lattice = Lattice(save_dir=save_dir, **params)

# beta_range=np.arange(0.1,1.1,0.1)
# for beta in beta_range:
#     sim_lattice.change_temperature(beta)
#     sim_lattice.advance(10000)
# sim_lattice.plot_phospho_trajectory()
# sim_lattice.plot_energy_trajectory()

phi_range=10**np.arange(-2,2,0.2)
for phi in phi_range:
    sim_lattice.change_potentials(chemical_potentials(phi))
    sim_lattice.advance(10000)
sim_lattice.plot_phospho_trajectory()
sim_lattice.plot_energy_trajectory()


plot_lattice(sim_lattice.phospho_state_positions(50), save_dir+'frame2',discrete=True,cmap='Spectral')
plot_lattice(sim_lattice.phospho_state_positions(-50), save_dir+'frame-100',discrete=True,cmap='Spectral')
#
X=sim_lattice.design_matrix()
learner=PCA(n_components=10)
learner.fit(X)
V=learner.components_
print(learner.explained_variance_ratio_)
for i in range(5):
    plot_lattice(sim_lattice.phospho_state_positions(V[i]),save_dir+'V{}'.format(i+1),cmap='Spectral')

with open(save_dir+'data.pickle','wb') as f:
    pickle.dump({'learner':learner, 'X':X, 'temp':sim_lattice.beta_trajectory}, f)

with open(save_dir+'trajectory.pickle','wb') as f:
    pickle.dump({'composition':sim_lattice.phospho_trajectory(), 'temp':sim_lattice.beta_trajectory,
                 'energy':sim_lattice.energy_trajectory()}, f)
